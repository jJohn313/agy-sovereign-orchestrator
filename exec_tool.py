import os
import re
import shlex
import subprocess
import json
from typing import Optional, Tuple

class ExecTool:
    DENYLIST = {
        "rofi", "fzf", "nano", "vim", "vi", "emacs",
        "less", "more", "man", "top", "htop", "btop",
        "watch", "tmux", "screen"
    }

    ALLOWED_FLAGS = {
        "rofi": {"-dump-config", "-dump-theme", "-help", "-version"},
        "fzf": {"--filter", "-f", "--version", "--help"},
        "man": {"-P"},
    }

    @staticmethod
    def is_interactive_command(cmd_str: str) -> Tuple[bool, str]:
        # Split by &&, ||, ;, |, and newlines
        segments = re.split(r"(?:&&|\|\||[;\n|])", cmd_str)

        for segment in segments:
            segment = segment.strip()
            if not segment:
                continue

            try:
                tokens = shlex.split(segment)
            except ValueError:
                # Fallback on unclosed quotes or raw heredocs
                tokens = segment.split()

            # Strip leading environment variable assignments (e.g., VAR=val cmd)
            cmd_tokens = [t for t in tokens if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", t)]
            if not cmd_tokens:
                continue

            binary = os.path.basename(cmd_tokens[0])

            if binary in ExecTool.DENYLIST:
                allowed = False
                if binary in ExecTool.ALLOWED_FLAGS:
                    # Check if any of the allowed flags are present in the arguments
                    allowed_flags = ExecTool.ALLOWED_FLAGS[binary]
                    for arg in cmd_tokens[1:]:
                        if arg in allowed_flags or any(arg.startswith(f) for f in allowed_flags):
                            # Allow things like -P cat for man, or --filter for fzf
                            # man -P cat -> arg is -P, which is in allowed_flags.
                            allowed = True
                            break
                        # Also handle --filter=something
                        if "=" in arg and arg.split("=")[0] in allowed_flags:
                            allowed = True
                            break

                    # Handle man -P cat specifically if -P is not split
                    if binary == "man" and "cat" in cmd_tokens and "-P" in cmd_tokens:
                        allowed = True

                if not allowed:
                    return True, f"Interactive command '{binary}' blocked. Must be run with non-interactive flags or export options."

            if binary == "ssh":
                # Strip SSH flags (e.g. -p 22, -i key, -o StrictHostKeyChecking=no, -v, -T, -t)
                # and check if a remote command string or heredoc payload is passed.
                ssh_args = cmd_tokens[1:]

                filtered_args = []
                skip_next = False
                for i, arg in enumerate(ssh_args):
                    if skip_next:
                        skip_next = False
                        continue
                    # Common ssh flags that take an argument
                    if arg in ("-p", "-i", "-o", "-l", "-L", "-R", "-D", "-c", "-m", "-F", "-E", "-w", "-b"):
                        skip_next = True
                        continue
                    # Flags that don't take an argument
                    if arg.startswith("-"):
                        # If it's a combined flag like -vT, we skip it
                        continue
                    filtered_args.append(arg)

                # After filtering flags, the remaining arguments should be: [user@]hostname [command]
                # We need at least 2 arguments remaining (hostname + command)
                # Note: if heredoc is used (ssh host << EOF), the operator isn't in cmd_tokens
                # but segment would look like "ssh host" and it's problematic if we don't look at original cmd_str.
                # However, re.split splits on newlines and pipes. Heredoc '<<' might not be split.
                # Let's check the raw segment for '<<'
                if len(filtered_args) <= 1 and "<<" not in segment:
                    return True, "Bare interactive SSH sessions are prohibited. Pass a remote command string or use heredoc: ssh host 'bash -s' << 'EOF' ... EOF"

        return False, ""

    @staticmethod
    def execute_shell(cmd_str: str, cwd: Optional[str] = None, timeout: int = 60) -> str:
        is_interactive, error_msg = ExecTool.is_interactive_command(cmd_str)
        if is_interactive:
            return json.dumps({"status": "err", "error": error_msg})

        HEADLESS_ENV = {
            **os.environ,
            "PAGER": "cat",
            "SYSTEMD_PAGER": "cat",
            "GIT_PAGER": "cat",
            "EDITOR": "cat",
            "VISUAL": "cat",
            "TERM": "dumb",
            "DEBIAN_FRONTEND": "noninteractive",
        }

        try:
            res = subprocess.run(
                cmd_str,
                shell=True,
                cwd=cwd or os.getcwd(),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout,
                env=HEADLESS_ENV,
            )

            if res.returncode != 0:
                return json.dumps({
                    "status": "err",
                    "exit_code": res.returncode,
                    "error": res.stderr.strip() or res.stdout.strip(),
                    "stdout": res.stdout.strip()
                })
            else:
                return json.dumps({
                    "status": "ok",
                    "stdout": res.stdout.strip()
                })
        except subprocess.TimeoutExpired:
            return json.dumps({"status": "err", "error": f"Command timed out after {timeout} seconds"})
        except Exception as e:
            return json.dumps({"status": "err", "error": str(e)})
