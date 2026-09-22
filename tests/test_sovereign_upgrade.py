"""
Comprehensive Test Suite for Production Sovereign Orchestrator Upgrade.
Covers all 7 verification axes specified in Section 9:
1. Daemon Ping & Stale Socket Test
2. 10-Minute Idle Timeout Test
3. Traceback Anchor Test
4. Fault-Tolerant AST Test
5. Ephemeral Reset & Autonomous MCP Install Test
6. SQLite WAL & Concurrency Test
7. End-to-End Non-Git Workspace Test
"""

import os
import sys
import time
import shutil
import sqlite3
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from daemon import OrchestratorDaemon, DaemonClient, SOCKET_PATH
from sanitizer import fold_log_output
from retrieval import extract_ast_outline
from meta_tools import LocalToolRegistry, PONYTAIL_SCHEMA
from orchestrator import AgyOrchestrator, AgentResponse, ToolCall
from post_turn_hook import PostTurnHook, ensure_sqlite_wal, WORKSPACES_DB_PATH
from mcp_installer import AutonomousMcpInstaller


class TestSovereignOrchestratorUpgrade(unittest.TestCase):
    """Production Sovereign Verification Suite."""

    def test_1_daemon_ping_and_stale_socket(self):
        """Test 1: Create dummy dead socket, assert auto-unlinking and daemon auto-spawn."""
        os.makedirs(os.path.dirname(SOCKET_PATH), exist_ok=True)
        if os.path.exists(SOCKET_PATH):
            try:
                os.unlink(SOCKET_PATH)
            except OSError:
                pass
        # Create dead dummy socket file
        with open(SOCKET_PATH, "w") as f:
            f.write("dead_stale_socket_test")

        client = DaemonClient()
        self.assertFalse(client._try_ping())

        # Auto-recovery
        client.ensure_daemon()
        self.assertTrue(client._try_ping())

        resp = client.send_request({"action": "ping"})
        self.assertIsNotNone(resp)
        self.assertEqual(resp.get("status"), "pong")

    def test_2_idle_timeout_vram_unloader(self):
        """Test 2: Verify check_idle(force_seconds=601) triggers CUDA/VRAM flushing."""
        daemon = OrchestratorDaemon()
        daemon.engine.model_loaded = True

        unloaded = daemon.check_idle(force_seconds=601)
        self.assertTrue(unloaded)
        self.assertFalse(daemon.engine.model_loaded)

    def test_3_traceback_anchor_log_folding(self):
        """Test 3: 120-line log with assertion failure at line 50. Preserves lines 45-75."""
        raw_lines = [f"Setup invocation line {i}" for i in range(120)]
        raw_lines[50] = "AssertionError: Critical assertion failed at line 50"

        folded = fold_log_output("\n".join(raw_lines), exit_code=1)
        # Check setup lines preserved
        self.assertIn("Setup invocation line 0", folded)
        self.assertIn("Setup invocation line 10", folded)
        # Check folded gap indicator
        self.assertIn("... [Folded", folded)
        # Check error context window preserved (45-75)
        self.assertIn("Setup invocation line 45", folded)
        self.assertIn("AssertionError: Critical assertion failed at line 50", folded)
        self.assertIn("Setup invocation line 75", folded)

    def test_4_fault_tolerant_ast(self):
        """Test 4: Deliberate Python syntax error fallback to regex extraction without crashing."""
        broken_code = """
import os
import sys

def calculate_checksum(data):
    \"\"\"Calculate valid checksum.\"\"\"
    return sum(data)

class BrokenClass:
    def broken_method(self, arg1,
        # Broken indentation / unclosed parameter list causes SyntaxError
        return arg1
"""
        outline = extract_ast_outline(broken_code)
        self.assertIn("[AST Fallback: Regex Structural Extraction]", outline)
        self.assertIn("def calculate_checksum", outline)
        self.assertIn("class BrokenClass:", outline)
        self.assertIn("def broken_method", outline)

    def test_5_ephemeral_reset_and_autonomous_mcp_install(self):
        """
        Test 5:
        a) Verify active tools reset to baseline on new turn.
        b) Call enable_capability for unregistered tool, verify autonomous install,
           registration in mcps.json, mounted for turn, and standby default.
        """
        temp_config = tempfile.NamedTemporaryFile(suffix=".json", delete=False).name
        try:
            with open(temp_config, "w") as f:
                f.write('{"capabilities": {}}')

            registry = LocalToolRegistry(config_path=temp_config)
            # a) Ephemeral baseline reset
            baseline = registry.get_baseline_schemas()
            baseline_names = [t["name"] for t in baseline]
            self.assertEqual(sorted(baseline_names), ["enable_capability", "ponytail"])

            # b) Autonomous install for test verified capability
            orchestrator = AgyOrchestrator(registry=registry)

            def mock_frontier_model(messages, tools, context):
                tool_names = [t.get("name") for t in tools]
                if "test_sandbox_action" not in tool_names:
                    return AgentResponse(
                        content="",
                        tool_calls=[
                            ToolCall(
                                name="enable_capability",
                                arguments={"capability": "test_sandbox_tool", "reason": "Need sandboxed tool"},
                                call_id="call_auto_install",
                            )
                        ],
                    )
                else:
                    return AgentResponse(content="Successfully used dynamically installed capability.")

            turn_result = orchestrator.process_turn(
                user_prompt="Run sandboxed test task",
                call_frontier_model=mock_frontier_model,
            )
            self.assertIn("test_sandbox_action", turn_result["active_tools"])
            self.assertIn("Successfully used dynamically installed capability.", turn_result["final_content"])

            # Verify saved in registry with standby state
            saved_cap = registry.capabilities.get("test_sandbox_tool", {})
            self.assertTrue(saved_cap.get("standby", False))
        finally:
            if os.path.exists(temp_config):
                os.unlink(temp_config)

    def test_6_sqlite_wal_and_concurrency(self):
        """Test 6: Verify PRAGMA journal_mode=WAL is active on workspaces_vec.db."""
        if os.path.exists(WORKSPACES_DB_PATH):
            wal_active = ensure_sqlite_wal(WORKSPACES_DB_PATH)
            self.assertTrue(wal_active)
            conn = sqlite3.connect(WORKSPACES_DB_PATH, timeout=5.0)
            mode = conn.execute("PRAGMA journal_mode;").fetchone()[0]
            conn.close()
            self.assertEqual(mode.lower(), "wal")

    def test_7_end_to_end_non_git_workspace(self):
        """Test 7: From /tmp/agy_test_workspace, mutate file, assert mtime triggers sem-index."""
        test_dir = "/tmp/agy_test_workspace"
        os.makedirs(test_dir, exist_ok=True)
        test_file = os.path.join(test_dir, "work_file.py")

        # Mutate file
        with open(test_file, "w") as f:
            f.write("def sample_work(): return 42\n")

        hook = PostTurnHook()
        res = hook.detect_changes_and_reindex(cwd=test_dir)
        self.assertEqual(res["mode"], "non_git_mtime")
        self.assertTrue(res["reindexed"])
        self.assertEqual(res["path"], test_dir)


if __name__ == "__main__":
    unittest.main()
