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

if __name__ == '__main__':
    unittest.main()
