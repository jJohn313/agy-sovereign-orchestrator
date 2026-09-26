import unittest
import os
from exec_tool import ExecTool

class TestBashGuard(unittest.TestCase):
    def test_interactive_binary_blocking(self):
        blocked_cmds = [
            "rofi",
            "nano foo.txt",
            "cat file | fzf",
            "top",
            "vim file.py",
            "less data.txt",
            "watch ls"
        ]
        for cmd in blocked_cmds:
            is_interactive, msg = ExecTool.is_interactive_command(cmd)
            self.assertTrue(is_interactive, f"Expected '{cmd}' to be blocked.")
            self.assertIn("blocked", msg)

    def test_flagged_exceptions_allowed(self):
        allowed_cmds = [
            "rofi -dump-config",
            "fzf --filter \"test\"",
            "man -P cat ls",
            "rofi -version"
        ]
        for cmd in allowed_cmds:
            is_interactive, msg = ExecTool.is_interactive_command(cmd)
            self.assertFalse(is_interactive, f"Expected '{cmd}' to be allowed, got error: {msg}")

    def test_lexical_false_positive_protection(self):
        safe_cmds = [
            "grep -E \"nano\" config.txt",
            "echo \"installed vim\"",
            "python3 -c \"import vim\"",
            "cat rofi.conf"
        ]
        for cmd in safe_cmds:
            is_interactive, msg = ExecTool.is_interactive_command(cmd)
            self.assertFalse(is_interactive, f"Expected '{cmd}' to be allowed, got error: {msg}")

    def test_bare_ssh_vs_remote_command(self):
        # Blocked
        blocked_ssh = [
            "ssh acer-server",
            "ssh user@host",
            "ssh -p 22 host",
            "ssh -i key.pem -o StrictHostKeyChecking=no host"
        ]
        for cmd in blocked_ssh:
            is_interactive, msg = ExecTool.is_interactive_command(cmd)
            self.assertTrue(is_interactive, f"Expected '{cmd}' to be blocked.")
            self.assertIn("Bare interactive SSH sessions are prohibited", msg)

        # Permitted
        allowed_ssh = [
            "ssh acer-server \"ls -la\"",
            "ssh user@host 'cat /etc/os-release'",
            "ssh -p 22 host uptime",
            "ssh host 'bash -s' << 'EOF'\necho ok\nEOF"
        ]
        for cmd in allowed_ssh:
            is_interactive, msg = ExecTool.is_interactive_command(cmd)
            self.assertFalse(is_interactive, f"Expected '{cmd}' to be allowed, got error: {msg}")

    def test_headless_execution_verification(self):
        # We test that execute_shell injects headless env vars and doesn't block on pager.
        # `git diff` should exit cleanly. Even if not in a git repo, it should exit and not hang.
        res = ExecTool.execute_shell("echo -e 'a\\nb' | cat")
        self.assertIn("ok", res)

        # Test an actual executable that we know uses PAGER
        # Just running env | grep PAGER might be flaky depending on system, but we can try
        res_env = ExecTool.execute_shell("env | grep PAGER")
        self.assertIn("PAGER=cat", res_env)
        self.assertIn("GIT_PAGER=cat", res_env)

    def test_repl_thrash_warning_local(self):
        ExecTool.reset_eval_counter()

        # 1st call
        res = ExecTool.execute_shell("python3 -c 'print(1)'")
        self.assertNotIn("SYSTEM WARNING: REPL micro-probing detected", res)

        # 2nd call
        res = ExecTool.execute_shell("python3 -c 'print(2)'")
        self.assertNotIn("SYSTEM WARNING: REPL micro-probing detected", res)

        # 3rd call - should trigger warning
        res = ExecTool.execute_shell("python3 -c 'print(3)'")
        self.assertIn("SYSTEM WARNING: REPL micro-probing detected", res)
        self.assertIn('"status": "ok"', res) # Should not fail

        # Reset counter
        ExecTool.reset_eval_counter()

    def test_repl_thrash_warning_remote(self):
        ExecTool.reset_eval_counter()

        # Note: We need a command that succeeds to properly test this without SSH hanging or failing.
        # However, we only care about the command parsing logic detecting it as an evaluator.
        # Since we use execute_shell, it will actually run the command.
        # A fake ssh command will fail. We can mock is_inline_evaluator or just test is_inline_evaluator directly.
        # Let's test the is_inline_evaluator directly for the wrapping logic.

        self.assertTrue(ExecTool.is_inline_evaluator("ssh host 'python3 -c \"print(1)\"'"))
        self.assertTrue(ExecTool.is_inline_evaluator("sudo bash -c 'node -e \"console.log(1)\"'"))
        self.assertTrue(ExecTool.is_inline_evaluator("doas sh -c 'python -c \"print(1)\"'"))
        self.assertTrue(ExecTool.is_inline_evaluator("ssh -p 22 host 'ruby -e \"puts 1\"'"))

        # Also let's just simulate the counter using execute_shell with a harmless local bash -c wrapping python -c
        res = ExecTool.execute_shell("bash -c 'python3 -c \"print(1)\"'")
        self.assertNotIn("SYSTEM WARNING", res)

        res = ExecTool.execute_shell("bash -c 'python3 -c \"print(2)\"'")
        self.assertNotIn("SYSTEM WARNING", res)

        res = ExecTool.execute_shell("bash -c 'python3 -c \"print(3)\"'")
        self.assertIn("SYSTEM WARNING", res)

        ExecTool.reset_eval_counter()

    def test_eval_counter_reset(self):
        ExecTool.reset_eval_counter()

        # 1st call eval
        ExecTool.execute_shell("python3 -c 'print(1)'")
        self.assertEqual(ExecTool.consecutive_eval_count, 1)

        # 2nd call eval
        ExecTool.execute_shell("python3 -c 'print(2)'")
        self.assertEqual(ExecTool.consecutive_eval_count, 2)

        # Non-evaluator resets counter
        ExecTool.execute_shell("ls")
        self.assertEqual(ExecTool.consecutive_eval_count, 0)

        # 3rd eval call doesn't trigger warning
        res = ExecTool.execute_shell("python3 -c 'print(3)'")
        self.assertNotIn("SYSTEM WARNING", res)
        self.assertEqual(ExecTool.consecutive_eval_count, 1)

if __name__ == '__main__':
    unittest.main()
