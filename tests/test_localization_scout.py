import unittest
from unittest.mock import patch, MagicMock
import os
import json

from ephemeral_scout import EphemeralScout
from retrieval import TargetedVectorRetriever
from orchestrator import AgyOrchestrator, PromptPrefixStabilizer, STATIC_SYSTEM_DIRECTIVE


class TestLocalizationScout(unittest.TestCase):
    def setUp(self):
        self.retriever = TargetedVectorRetriever()

    @patch("retrieval.CodebaseRetriever.search")
    @patch("retrieval.Mem0Retriever.search")
    @patch("ephemeral_scout.EphemeralScout.resolve_scope")
    def test_mock_high_confidence_vector_hit(self, mock_resolve_scope, mock_mem0_search, mock_codebase_search):
        # Mock vector hit >= 0.75
        mock_codebase_search.return_value = [
            {"path": "src/valid.py", "name": "valid", "summary": "valid content", "similarity": 0.85}
        ]
        mock_mem0_search.return_value = []

        context = self.retriever.retrieve(
            user_prompt="find valid",
            codebase_gate=1.0,
            mem0_gate=0.0
        )

        mock_resolve_scope.assert_not_called()
        self.assertIn("src/valid.py", context["formatted_context"])
        self.assertEqual(context.get("pre_resolved_scope", ""), "")

    @patch("retrieval.CodebaseRetriever.search")
    @patch("retrieval.Mem0Retriever.search")
    @patch("ephemeral_scout.EphemeralScout.is_available")
    @patch("ephemeral_scout.urllib.request.urlopen")
    @patch("ephemeral_scout.EphemeralScout._generate_candidates")
    @patch("os.path.exists")
    def test_mock_low_confidence_vector_miss_active_slm(
        self, mock_exists, mock_generate_candidates, mock_urlopen, mock_is_available, mock_mem0, mock_cb
    ):
        # Codebase search internally filters out scores < 0.75, so it returns empty
        mock_cb.return_value = []
        mock_mem0.return_value = []

        mock_is_available.return_value = True

        # We need mock_exists to return True for valid paths and False for hallucinated
        def exists_side_effect(path):
            return path.endswith("src/valid.py")
        mock_exists.side_effect = exists_side_effect

        mock_generate_candidates.return_value = ["src/valid.py", "src/other.py"]

        mock_resp = MagicMock()
        slm_output = {"targets": ["src/valid.py", "src/hallucinated.py"]}
        # response field inside result containing JSON string
        mock_resp.read.return_value = json.dumps({"response": json.dumps(slm_output)}).encode("utf-8")

        # mock urlopen context manager
        mock_context_manager = MagicMock()
        mock_context_manager.__enter__.return_value = mock_resp
        mock_urlopen.return_value = mock_context_manager

        context = self.retriever.retrieve(
            user_prompt="find valid",
            codebase_gate=1.0,
            mem0_gate=0.0
        )

        pre_resolved = context.get("pre_resolved_scope", "")
        self.assertIn("src/valid.py", pre_resolved)
        self.assertNotIn("src/hallucinated.py", pre_resolved)
        self.assertIn("### Pre-Resolved Execution Scope (Authoritative)", pre_resolved)

    @patch("retrieval.CodebaseRetriever.search")
    @patch("ephemeral_scout.urllib.request.urlopen")
    def test_mock_provider_down_timeout(self, mock_urlopen, mock_cb):
        mock_cb.return_value = []

        import urllib.error
        mock_urlopen.side_effect = urllib.error.URLError("timeout")

        scout = EphemeralScout()
        self.assertFalse(scout.is_available())

        context = self.retriever.retrieve(
            user_prompt="find valid",
            codebase_gate=1.0,
            mem0_gate=0.0
        )

        self.assertEqual(context.get("pre_resolved_scope", ""), "")

    def test_cache_invariant_check(self):
        system_directive = STATIC_SYSTEM_DIRECTIVE
        active_tools = [{"name": "ponytail"}, {"name": "enable_capability"}]

        l4a_static = {"repositories": ["/test/repo"]}
        l4b_volatile = {"is_git": True, "branch": "main", "dirty_files": []}

        assembled_without_scope = PromptPrefixStabilizer.assemble_prompt_layers(
            system_directive=system_directive,
            invariant_tools=active_tools,
            dynamic_active_tools=active_tools,
            l4a_static_invariants=l4a_static,
            l4b_volatile_state=l4b_volatile,
            dynamic_turn_payload="find valid"
        )

        assembled_with_scope = PromptPrefixStabilizer.assemble_prompt_layers(
            system_directive=system_directive,
            invariant_tools=active_tools,
            dynamic_active_tools=active_tools,
            l4a_static_invariants=l4a_static,
            l4b_volatile_state=l4b_volatile,
            dynamic_turn_payload="### Pre-Resolved Execution Scope (Authoritative)\n- src/valid.py\n\nfind valid"
        )

        self.assertEqual(
            assembled_without_scope["prefix_cache_hash"],
            assembled_with_scope["prefix_cache_hash"],
            "Prefix cache hash must remain invariant when pre-resolved scope is added to turn payload."
        )


if __name__ == "__main__":
    unittest.main()
