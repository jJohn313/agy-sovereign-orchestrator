import unittest
import tempfile
import os
import shutil
import sqlite3
from unittest.mock import patch, MagicMock
from orchestrator import AgyOrchestrator
from post_turn_hook import WORKSPACES_DB_PATH

class TestPrefixCacheTelemetry(unittest.TestCase):
    def setUp(self):
        # Create a temporary workspace and database
        self.test_dir = tempfile.mkdtemp()
        self.db_fd, self.db_path = tempfile.mkstemp()

        # Patch WORKSPACES_DB_PATH in post_turn_hook
        self.patcher1 = patch('post_turn_hook.WORKSPACES_DB_PATH', self.db_path)
        self.patcher1.start()

        # Avoid direct Mem0 instantiation errors
        self.patcher2 = patch('post_turn_hook.PostTurnHook._get_memory', return_value=None)
        self.patcher2.start()

        # Initialize Orchestrator
        self.orchestrator = AgyOrchestrator()

        # Setup session ID
        self.session_id = "test_session_123"

    def tearDown(self):
        self.patcher1.stop()
        self.patcher2.stop()
        os.close(self.db_fd)
        if os.path.exists(self.db_path):
            os.unlink(self.db_path)
        shutil.rmtree(self.test_dir)

    def test_3_turn_interactive_session(self):
        # --- TURN 1 (Cold Start) ---
        res_t1 = self.orchestrator.process_turn(
            user_prompt="Turn 1",
            cwd=self.test_dir,
            session_id=self.session_id,
        )
        telemetry_t1 = res_t1.get("cache_telemetry", {})

        self.assertFalse(telemetry_t1.get("cache_hit"))
        self.assertEqual(telemetry_t1.get("session_hit_rate"), 0.0)
        hash_t1 = telemetry_t1.get("current_hash")
        self.assertIsNotNone(hash_t1)

        # --- TURN 2 (L4b Volatile Mutation - Cache Hit) ---
        # Mocking get_volatile_workspace_state to return dirty files
        # The prefix should remain invariant because volatile state goes into L5, not L1-L4a
        with patch('orchestrator.PromptPrefixStabilizer.get_volatile_workspace_state',
                  return_value={"is_git": True, "branch": "main", "dirty_files": ["new_file.txt"], "raw_porcelain": "?? new_file.txt"}):
            res_t2 = self.orchestrator.process_turn(
                user_prompt="Turn 2",
                cwd=self.test_dir,
                session_id=self.session_id,
            )

        telemetry_t2 = res_t2.get("cache_telemetry", {})

        self.assertTrue(telemetry_t2.get("cache_hit"))
        self.assertEqual(telemetry_t2.get("current_hash"), hash_t1)
        self.assertEqual(telemetry_t2.get("session_hit_rate"), 1.0)

        # --- TURN 3 (L1-L4a Invariant Change - Cache Bust) ---
        # Simulate changing baseline invariants
        original_get_baseline = self.orchestrator.registry.get_baseline_schemas

        def mocked_get_baseline():
            baselines = original_get_baseline()
            baselines.append({"name": "new_invariant_tool", "description": "busting cache"})
            return baselines

        with patch.object(self.orchestrator.registry, 'get_baseline_schemas', side_effect=mocked_get_baseline):
            res_t3 = self.orchestrator.process_turn(
                user_prompt="Turn 3",
                cwd=self.test_dir,
                session_id=self.session_id,
            )

        telemetry_t3 = res_t3.get("cache_telemetry", {})

        self.assertFalse(telemetry_t3.get("cache_hit"))
        self.assertNotEqual(telemetry_t3.get("current_hash"), hash_t1)
        # Turn 1: None -> Hit=False
        # Turn 2: Hit=True (1 hit)
        # Turn 3: Hit=False (1 hit, 1 bust -> total 3 turns -> rate = 1 / (3-1) = 0.5)
        self.assertEqual(telemetry_t3.get("session_hit_rate"), 0.5)

if __name__ == '__main__':
    unittest.main()
