"""
Test DeBERTa NLI Spin-Up & Auto-Unload Lifecycle.
Ensures:
1. Model only spins up into VRAM during message evaluation.
2. Model is immediately unloaded from VRAM once the turn/evaluation completes.
3. True DeBERTa neural scoring discriminates prompts, mounting only required tools to prune token usage.
"""

import os
import unittest
from local_decision_engine import LocalDecisionEngine
from daemon import OrchestratorDaemon


class TestDebertaLifecycle(unittest.TestCase):

    def setUp(self):
        self.engine = LocalDecisionEngine()

    def tearDown(self):
        if self.engine.model_loaded:
            self.engine.unload_vram()

    def test_1_active_session_spins_up_and_unloads(self):
        """Verify active_session context manager loads model and unloads upon exit."""
        self.assertFalse(self.engine.model_loaded)
        self.assertIsNone(self.engine.model)

        with self.engine.active_session():
            self.assertTrue(self.engine.model_loaded)
            self.assertIsNotNone(self.engine.model)
            score = self.engine.noul("The user said: \"git status\"", "The task requires using git or version control.")
            self.assertGreater(score, 0.70)

        # After block exit, VRAM must be completely freed
        self.assertFalse(self.engine.model_loaded)
        self.assertIsNone(self.engine.model)

    def test_2_standalone_noul_auto_unloads(self):
        """Verify standalone noul call automatically spins up and unloads."""
        self.assertFalse(self.engine.model_loaded)
        score = self.engine.noul("The user said: \"read README.md\"", "The task requires reading, writing, or viewing files or directories.")
        self.assertGreater(score, 0.70)
        self.assertFalse(self.engine.model_loaded)
        self.assertIsNone(self.engine.model)

    def test_3_daemon_gate_pre_turn_lifecycle(self):
        """Verify daemon gate_pre_turn spins model up, evaluates batch, and leaves model unloaded."""
        daemon = OrchestratorDaemon()
        self.assertFalse(daemon.engine.model_loaded)

        req = {
            "action": "gate_pre_turn",
            "user_prompt": "What is the capital of France?",
            "cwd": "/home/john",
            "available_tools": ["git", "filesystem", "terminal"]
        }
        res = daemon.handle_request(req)
        self.assertIn("scores", res)
        scores = res["scores"]

        # Ensure general knowledge prompt keeps heavy tools unmounted (token-pruning)
        self.assertLess(scores.get("needs_git", 1.0), 0.60)
        self.assertLess(scores.get("needs_filesystem", 1.0), 0.60)
        self.assertLess(scores.get("needs_terminal", 1.0), 0.60)

        # Ensure daemon immediately unloaded model
        self.assertFalse(daemon.engine.model_loaded)
        self.assertIsNone(daemon.engine.model)

    def test_4_daemon_selective_tool_gating(self):
        """Verify git prompt selectively enables git and keeps others pruned."""
        daemon = OrchestratorDaemon()
        req = {
            "action": "gate_pre_turn",
            "user_prompt": "git commit -m 'feat: add feature' and push",
            "cwd": "/home/john",
            "available_tools": ["git", "filesystem", "terminal"]
        }
        res = daemon.handle_request(req)
        scores = res["scores"]
        self.assertGreater(scores.get("needs_git", 0.0), 0.70)

        # Daemon unloaded immediately
        self.assertFalse(daemon.engine.model_loaded)
        self.assertIsNone(daemon.engine.model)


if __name__ == "__main__":
    unittest.main()
