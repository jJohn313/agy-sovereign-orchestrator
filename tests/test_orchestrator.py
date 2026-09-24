"""
Unit and Integration Tests for AGY Jev Orchestrator Architecture.
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock

# Add module path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from jev_client import JevClient, FALLBACK_PRE_GATING, FALLBACK_POST_GATING
from retrieval import TargetedVectorRetriever, CodebaseRetriever, Mem0Retriever
from meta_tools import LocalToolRegistry, PONYTAIL_SCHEMA
from post_turn_hook import PostTurnHook
from orchestrator import AgyOrchestrator, AgentResponse, ToolCall


class TestJevClient(unittest.TestCase):
    """Test Jev System-1 Client behavior and fallbacks."""

    def test_fallback_when_no_api_key(self):
        client = JevClient(api_key="")
        scores = client.gate_pre_turn(
            user_prompt="Run tests",
            cwd="/home/john",
            available_tools=["git", "terminal"],
        )
        self.assertEqual(scores, FALLBACK_PRE_GATING)
        self.assertEqual(scores["needs_git"], 1.0)
        self.assertEqual(scores["needs_filesystem"], 1.0)
        self.assertEqual(scores["needs_terminal"], 1.0)
        self.assertEqual(scores["needs_codebase_search"], 0.0)
        self.assertEqual(scores["needs_user_memory"], 0.0)

    def test_post_turn_fallback_when_no_api_key(self):
        client = JevClient(api_key="")
        scores = client.filter_post_turn(
            user_prompt="Fix bug",
            agent_actions_summary="Fixed typo in file",
        )
        self.assertEqual(scores, FALLBACK_POST_GATING)
        self.assertEqual(scores["is_durable_knowledge"], 0.0)

    @patch("requests.post")
    def test_successful_pre_gating(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {
            "predictions": {
                "needs_git": 0.92,
                "needs_filesystem": 0.88,
                "needs_terminal": 0.15,
                "needs_codebase_search": 0.72,
                "needs_user_memory": 0.35,
            }
        }
        mock_post.return_value = mock_resp

        client = JevClient(api_key="test-key")
        scores = client.gate_pre_turn("git status", "/home/john", ["git", "filesystem"])
        self.assertAlmostEqual(scores["needs_git"], 0.92)
        self.assertAlmostEqual(scores["needs_filesystem"], 0.88)
        self.assertAlmostEqual(scores["needs_terminal"], 0.15)
        self.assertAlmostEqual(scores["needs_codebase_search"], 0.72)
        self.assertAlmostEqual(scores["needs_user_memory"], 0.35)

    @patch("requests.post")
    def test_timeout_fallback(self, mock_post):
        import requests
        mock_post.side_effect = requests.Timeout("Connection timed out after 1.5s")

        client = JevClient(api_key="test-key", timeout=1.5)
        scores = client.gate_pre_turn("Inspect code", "/home/john", ["git"])
        self.assertEqual(scores, FALLBACK_PRE_GATING)


class TestTargetedVectorRetrieval(unittest.TestCase):
    """Test vector retrieval and strict cosine similarity filtering (>= 0.75)."""

    def test_strict_filtering_threshold(self):
        mock_cb = MagicMock(spec=CodebaseRetriever)
        mock_cb.search.return_value = [
            {"name": "repo1", "similarity": 0.85, "path": "/path/1", "summary": "match 1"},
            # Notice 0.74 must be excluded by retriever implementation
        ]
        mock_mem = MagicMock(spec=Mem0Retriever)
        mock_mem.search.return_value = [
            {"id": "m1", "memory": "Use snake_case", "score": 0.91},
        ]

        retriever = TargetedVectorRetriever(
            codebase_retriever=mock_cb,
            mem0_retriever=mock_mem,
            threshold=0.75,
        )

        # 1. Gates below 0.5: zero queries made
        res1 = retriever.retrieve("query", codebase_gate=0.4, mem0_gate=0.3)
        self.assertEqual(len(res1["codebase_matches"]), 0)
        self.assertEqual(len(res1["mem0_matches"]), 0)
        mock_cb.search.assert_not_called()
        mock_mem.search.assert_not_called()

        # 2. Gates above 0.5: queries executed and filtered
        res2 = retriever.retrieve("query", codebase_gate=0.6, mem0_gate=0.8)
        self.assertEqual(len(res2["codebase_matches"]), 1)
        self.assertEqual(len(res2["mem0_matches"]), 1)
        self.assertIn("Relevant Durable Memory", res2["formatted_context"])
        self.assertIn("Relevant Codebase Workspaces", res2["formatted_context"])


class TestMetaToolsAndRegistry(unittest.TestCase):
    """Test dynamic enable_capability schema and system anchors."""

    def test_baseline_invariants(self):
        registry = LocalToolRegistry()
        schemas = registry.get_baseline_schemas()
        self.assertEqual(len(schemas), 2)
        self.assertEqual(schemas[0]["name"], "ponytail")
        self.assertEqual(schemas[1]["name"], "enable_capability")

        meta_tool = registry.get_meta_tool_schema()
        enum_caps = meta_tool["parameters"]["properties"]["capability"]["enum"]
        for required_cap in ["git", "filesystem", "terminal", "custom_home_tools", "web_search", "python_eval"]:
            self.assertIn(required_cap, enum_caps)

    def test_system_anchors_and_delta_update(self):
        registry = LocalToolRegistry()
        anchors = registry.system_anchors
        self.assertIn("repositories", anchors)
        self.assertIn("config_hubs", anchors)

        initial_len = len(anchors["workspace_paths"])
        registry.update_registry_delta(["/home/john/NewProjectDir"])
        self.assertIn("/home/john/NewProjectDir", registry.system_anchors["workspace_paths"])


class TestPostTurnHook(unittest.TestCase):
    """Test Jev downstream write filter (Mem0 commit strictly when > 0.85)."""

    def test_discard_when_score_lte_085(self):
        mock_jev = MagicMock(spec=JevClient)
        mock_jev.filter_post_turn.return_value = {"is_durable_knowledge": 0.80}

        hook = PostTurnHook(jev_client=mock_jev)
        hook._commit_memory = MagicMock()

        res = hook.process_turn_completion(
            user_prompt="What is 2 + 2?",
            actions_summary="Calculated 4",
        )
        self.assertEqual(res["durable_score"], 0.80)
        self.assertFalse(res["committed"])
        hook._commit_memory.assert_not_called()

    def test_commit_when_score_gt_085(self):
        mock_jev = MagicMock(spec=JevClient)
        mock_jev.filter_post_turn.return_value = {"is_durable_knowledge": 0.95}

        hook = PostTurnHook(jev_client=mock_jev)
        hook._commit_memory = MagicMock(return_value=True)

        res = hook.process_turn_completion(
            user_prompt="Always use uv instead of pip in this repo",
            actions_summary="Updated config to use uv",
        )
        self.assertEqual(res["durable_score"], 0.95)
        self.assertTrue(res["committed"])
        hook._commit_memory.assert_called_once()


class TestOrchestratorEscalationLoop(unittest.TestCase):
    """Test dynamic escalation loop when model invokes enable_capability."""

    def test_dynamic_escalation_loop(self):
        registry = LocalToolRegistry()
        # Register a custom tool for testing
        registry.capabilities["custom_test_tool"] = {
            "name": "custom_test_tool",
            "tools": [{"name": "test_exec", "parameters": {"type": "object"}}],
        }

        orchestrator = AgyOrchestrator(registry=registry)

        # Simulate frontier model behavior:
        # Step 1: Model calls enable_capability("custom_test_tool")
        # Step 2: Model resumes turn with updated tools and returns final response
        step = {"count": 0}

        def mock_frontier_model(messages, tools, context):
            step["count"] += 1
            if step["count"] == 1:
                # First turn: requests capability
                return AgentResponse(
                    content="",
                    tool_calls=[
                        ToolCall(
                            name="enable_capability",
                            arguments={"capability": "custom_test_tool", "reason": "Need test tool"},
                            call_id="call_esc_1",
                        )
                    ],
                )
            else:
                # Second turn: verifies tool was mounted into tools schema
                tool_names = [t.get("name") for t in tools]
                self.assertIn("test_exec", tool_names)
                return AgentResponse(content="Successfully mounted and executed task.")

        active_tools = registry.get_baseline_schemas()
        loop_res = orchestrator.run_agent_loop(
            prompt="Please use the test capability",
            active_tools=active_tools,
            context={},
            call_frontier_model=mock_frontier_model,
        )

        self.assertEqual(loop_res["final_content"], "Successfully mounted and executed task.")
        self.assertIn("Mounted capability: custom_test_tool", loop_res["executed_actions"])
        # Check tool mounted in active_tools
        active_names = [t.get("name") for t in loop_res["active_tools"]]
        self.assertIn("test_exec", active_names)



class TestPromptPrefixStabilizer(unittest.TestCase):
    def test_cache_hash_stability(self):
        from orchestrator import PromptPrefixStabilizer
        system_directive = "Test System Directive"
        invariant_tools = [{"name": "ponytail", "desc": "test"}, {"name": "enable_capability", "desc": "test2"}]
        dynamic_active_tools = [{"name": "b_tool", "desc": "b"}, {"name": "a_tool", "desc": "a"}]
        l4a_static_invariants = {
            "repositories": ["/a/b", "/c/d"],
            "custom_home_scripts": [
                {"name": "test.sh", "path": "/home/user/bin/test.sh"},
            ],
            "mtime": 1234567,
            "pid": 999
        }

        # Turn 1
        l4b_turn1 = {"is_git": True, "branch": "main", "dirty_files": ["foo.py"]}
        res1 = PromptPrefixStabilizer.assemble_prompt_layers(
            system_directive=system_directive,
            invariant_tools=invariant_tools,
            dynamic_active_tools=dynamic_active_tools,
            l4a_static_invariants=l4a_static_invariants,
            l4b_volatile_state=l4b_turn1,
            dynamic_turn_payload="Do something",
        )

        # Turn 2: Mutated L4b, same L4a
        l4b_turn2 = {"is_git": True, "branch": "main", "dirty_files": ["foo.py", "bar.py"]}
        # Also mutate some stripped fields in L4a
        l4a_static_invariants["mtime"] = 7654321
        l4a_static_invariants["pid"] = 888

        res2 = PromptPrefixStabilizer.assemble_prompt_layers(
            system_directive=system_directive,
            invariant_tools=invariant_tools,
            dynamic_active_tools=dynamic_active_tools,
            l4a_static_invariants=l4a_static_invariants,
            l4b_volatile_state=l4b_turn2,
            dynamic_turn_payload="Do something else",
        )

        self.assertEqual(res1["prefix_cache_hash"], res2["prefix_cache_hash"])
        self.assertIn("bar.py", res2["turn_payload"])
        self.assertNotIn("bar.py", res1["turn_payload"])


if __name__ == "__main__":
    unittest.main()
