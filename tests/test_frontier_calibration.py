import os
import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path
import tempfile
import subprocess

from schema_minifier import minify_tool_schema, EXEMPT_TOOLS
from orchestrator import AgyOrchestrator, DEFAULT_MAX_ESCALATIONS, FRONTIER_MAX_ESCALATIONS, ToolCall, AgentResponse
from retrieval import read_file_content_smart, TargetedVectorRetriever
from daemon import OrchestratorDaemon, DaemonClient


class TestFrontierCalibration(unittest.TestCase):
    """Tests for the 5 frontier model calibrations (Neon & Hydrogen)."""

    def test_schema_minifier_exemptions(self):
        # 1. Exempt execution tool: full descriptions retained
        exempt_tool = {
            "name": "edit_file",
            "description": "Use this tool to edit an existing file with precise surgical block replacement rules.",
            "parameters": {
                "type": "object",
                "properties": {
                    "Target": {
                        "type": "string",
                        "description": "The exact multi-line string to be replaced, including leading whitespace and indentation.",
                    }
                }
            }
        }
        res_exempt = minify_tool_schema(exempt_tool)
        self.assertEqual(res_exempt["description"], exempt_tool["description"])
        self.assertEqual(res_exempt["parameters"]["properties"]["Target"]["description"],
                         exempt_tool["parameters"]["properties"]["Target"]["description"])

        # 2. Non-exempt tool: description truncated to 12 words
        non_exempt = {
            "name": "generic_search",
            "description": "One two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "A B C D E F G H I J K L M N O P"
                    }
                }
            }
        }
        res_non_exempt = minify_tool_schema(non_exempt)
        self.assertTrue(res_non_exempt["description"].endswith("..."))
        self.assertEqual(len(res_non_exempt["description"].split()), 12)

    def test_turn_escalation_scaling(self):
        orchestrator = AgyOrchestrator()

        call_count = 0
        def fake_frontier_model(messages, tools, context):
            nonlocal call_count
            call_count += 1
            if call_count <= 5:
                # Issue enable_capability escalation each turn
                return AgentResponse(
                    content="",
                    tool_calls=[ToolCall(name="enable_capability", arguments={"capability": "git"}, call_id=f"c_{call_count}")]
                )
            return AgentResponse(content="Done")

        # Test with default limit (2)
        res_default = orchestrator.run_agent_loop(
            prompt="test",
            active_tools=[],
            context={},
            call_frontier_model=fake_frontier_model,
            max_escalations=DEFAULT_MAX_ESCALATIONS,
        )
        self.assertEqual(res_default["escalation_count"], 2)

        # Reset and test with frontier limit (4)
        call_count = 0
        res_frontier = orchestrator.run_agent_loop(
            prompt="test",
            active_tools=[],
            context={},
            call_frontier_model=fake_frontier_model,
            max_escalations=FRONTIER_MAX_ESCALATIONS,
        )
        self.assertEqual(res_frontier["escalation_count"], 4)

    def test_smart_file_content_clean_vs_dirty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            test_file = tmp_path / "mod.py"
            test_file.write_text(
                "class Engine:\n"
                "    def run(self, speed: int) -> bool:\n"
                "        return True\n"
            )

            # Case A: Clean file (not in git dirty set) -> structural AST skeleton
            with patch("repo_intel_client.RepoIntelClient.get_dirty_files", return_value=set()):
                result_clean = read_file_content_smart(str(test_file), cwd=tmpdir)
                self.assertIn("structural AST skeleton", result_clean)
                self.assertIn("class Engine", result_clean)
                self.assertIn("def run(speed: int) -> bool", result_clean)

            # Case B: Dirty file (in git dirty set) -> raw file content
            with patch("repo_intel_client.RepoIntelClient.get_dirty_files", return_value={"mod.py"}):
                result_dirty = read_file_content_smart(str(test_file), cwd=tmpdir)
                self.assertEqual(result_dirty, test_file.read_text())

            # Case C: Dirty file exceeding 300 lines without target_symbol -> diff-anchored hydration
            large_file = tmp_path / "large.py"
            large_lines = ["# padding line\n"] * 350
            large_lines[50] = "def calculate_hash(data: str) -> str:\n    return 'hash'\n"
            large_file.write_text("".join(large_lines))

            mock_diff = (
                "@@ -45,15 +45,15 @@\n"
                "-    return 'old'\n"
                "+    return 'hash'\n"
            )
            with patch("repo_intel_client.RepoIntelClient.get_dirty_files", return_value={"large.py"}), \
                 patch("retrieval.extract_git_diff_hunks", return_value=mock_diff):
                result_diff_anchored = read_file_content_smart(str(large_file), cwd=tmpdir)
                self.assertIn("diff-anchored view", result_diff_anchored)
                self.assertIn("Active In-Flight Diff Hunks", result_diff_anchored)
                self.assertIn("+    return 'hash'", result_diff_anchored)
                self.assertIn("High-Level Structural AST Outline", result_diff_anchored)

    def test_daemon_socket_timeout(self):
        # Verify DaemonClient sets 60.0s socket timeout
        client = DaemonClient()
        # Verify OrchestratorDaemon has 60.0s in code
        import inspect
        src_daemon = inspect.getsource(OrchestratorDaemon.start)
        self.assertIn("client_sock.settimeout(60.0)", src_daemon)
        src_client = inspect.getsource(DaemonClient.send_request)
        self.assertIn("sock.settimeout(60.0)", src_client)

    def test_retriever_max_k_scaling(self):
        retriever = TargetedVectorRetriever(
            codebase_retriever=MagicMock(),
            mem0_retriever=MagicMock(),
        )
        # Mock 10 candidates
        candidates = [
            {"similarity": 0.90 - i * 0.01, "path": f"path_{i}.py", "name": f"f_{i}", "summary": f"summary {i}", "embedding": [0.1 * i, 0.2]}
            for i in range(10)
        ]
        retriever.codebase_retriever.search.return_value = candidates

        with patch("retrieval.compute_cosine_similarity", return_value=0.1):  # No MMR deduplication
            # Default scout k=4
            ctx_4 = retriever.retrieve(user_prompt="query", codebase_gate=1.0, mem0_gate=0.0, max_k=4)
            # Frontier k=6
            ctx_6 = retriever.retrieve(user_prompt="query", codebase_gate=1.0, mem0_gate=0.0, max_k=6)

            lines_4 = ctx_4["formatted_context"].strip().splitlines()
            lines_6 = ctx_6["formatted_context"].strip().splitlines()
            self.assertEqual(len(lines_4), 4)
            self.assertEqual(len(lines_6), 6)


if __name__ == "__main__":
    unittest.main()
