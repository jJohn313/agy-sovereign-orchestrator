"""
Sovereign Orchestrator Daemon & Unix Domain Socket Server.
Socket Path: ~/.config/agy/orchestrator.sock
Features:
- Sub-millisecond IPC for NLI decisions
- 10-Minute VRAM Auto-Idle Unloader (Fix 5 - Option A)
- Stale Socket Recovery with auto-spawn (Fix 3)
"""

import os
import sys
import time
import json
import socket
import select
import signal
import logging
import threading
import subprocess
from typing import Dict, Any, Optional

from local_decision_engine import LocalDecisionEngine

logger = logging.getLogger("agy.daemon")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [DAEMON] %(message)s")

SOCKET_PATH = os.path.expanduser("~/.config/agy/orchestrator.sock")
IDLE_TIMEOUT_SECONDS = 600  # 10 minutes


class OrchestratorDaemon:
    """Unix Domain Socket Daemon hosting the sovereign decision engine."""

    def __init__(self, socket_path: str = SOCKET_PATH):
        self.socket_path = socket_path
        self.engine = LocalDecisionEngine()
        self.running = False
        self.server_socket: Optional[socket.socket] = None
        self.last_activity_time = time.time()
        self.lock = threading.Lock()

    def check_idle(self, force_seconds: Optional[int] = None) -> bool:
        """
        Check if idle time exceeded limit.
        If idle > 10 minutes (or force_seconds > 600), offload VRAM.
        """
        elapsed = force_seconds if force_seconds is not None else (time.time() - self.last_activity_time)
        if elapsed >= IDLE_TIMEOUT_SECONDS:
            if self.engine.model_loaded:
                self.engine.unload_vram()
                return True
        return False

    def _idle_monitor_loop(self):
        """Background thread checking for 10-minute idle inactivity."""
        while self.running:
            time.sleep(10)
            with self.lock:
                self.check_idle()

    def handle_request(self, req: Dict[str, Any]) -> Dict[str, Any]:
        """Dispatch IPC requests."""
        action = req.get("action", "")
        self.last_activity_time = time.time()

        if action == "ping":
            return {"status": "pong", "model_loaded": self.engine.model_loaded}

        elif action == "noul":
            state = req.get("state", "")
            prop = req.get("proposition", "")
            score = self.engine.noul(state, prop)
            return {"result": score}

        elif action == "choice":
            state = req.get("state", "")
            options = req.get("options", [])
            instr = req.get("instructions", "")
            choices = self.engine.choice(state, options, instructions=instr)
            return {"result": choices}

        elif action == "score":
            state = req.get("state", "")
            levels = req.get("levels", {})
            val = self.engine.score(state, levels)
            return {"result": val}

        elif action == "gate_pre_turn":
            prompt = req.get("user_prompt", "")
            cwd = req.get("cwd", "")
            tools = req.get("available_tools", [])
            state = f"WORKING_DIR: {cwd}\nAVAILABLE_LOCAL_CAPABILITIES: {', '.join(tools)}\nUSER_PROMPT: {prompt}"

            needs_git = self.engine.noul(state, "This prompt requires git commands, commits, diffs, or repository status.")
            needs_fs = self.engine.noul(state, "This prompt requires inspecting, reading, or writing specific files or directories.")
            needs_term = self.engine.noul(state, "This prompt requires running shell commands, build diagnostics, or scripts.")
            needs_cb = self.engine.noul(state, "This task requires searching the semantic codebase index for symbol or function definitions.")
            needs_mem = self.engine.noul(state, "This task relies on past user preferences, architectural rules, or setup history.")

            return {
                "scores": {
                    "needs_git": round(needs_git, 4),
                    "needs_filesystem": round(needs_fs, 4),
                    "needs_terminal": round(needs_term, 4),
                    "needs_codebase_search": round(needs_cb, 4),
                    "needs_user_memory": round(needs_mem, 4),
                }
            }

        elif action == "filter_post_turn":
            prompt = req.get("user_prompt", "")
            summary = req.get("summary", "")
            state = f"USER_PROMPT: {prompt}\nRESPONSE_SUMMARY: {summary}"
            durable = self.engine.noul(
                state,
                "This exchange establishes a permanent user preference, key architectural invariant, or durable system config."
            )
            return {"is_durable_knowledge": round(durable, 4)}

        elif action == "check_idle":
            force_secs = req.get("force_seconds", 601)
            unloaded = self.check_idle(force_seconds=force_secs)
            return {"status": "idle_checked", "unloaded": unloaded, "model_loaded": self.engine.model_loaded}

        elif action == "reload_tools":
            return {"status": "reloaded"}

        elif action == "shutdown":
            self.running = False
            return {"status": "shutting_down"}

        return {"error": f"Unknown action: {action}"}

    def start(self):
        """Bind Unix Domain Socket and enter event loop."""
        os.makedirs(os.path.dirname(self.socket_path), exist_ok=True)
        if os.path.exists(self.socket_path):
            try:
                os.unlink(self.socket_path)
            except OSError:
                pass

        self.server_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.server_socket.bind(self.socket_path)
        self.server_socket.listen(16)
        self.running = True

        # Start idle monitor
        idle_thread = threading.Thread(target=self._idle_monitor_loop, daemon=True)
        idle_thread.start()

        logger.info("Orchestrator daemon bound to %s (PID: %d)", self.socket_path, os.getpid())

        while self.running:
            try:
                rlist, _, _ = select.select([self.server_socket], [], [], 1.0)
                if not rlist:
                    continue

                client_sock, _ = self.server_socket.accept()
                with client_sock:
                    client_sock.settimeout(5.0)
                    data = client_sock.recv(65536)
                    if not data:
                        continue
                    try:
                        req = json.loads(data.decode("utf-8"))
                        with self.lock:
                            resp = self.handle_request(req)
                        client_sock.sendall(json.dumps(resp).encode("utf-8"))
                    except Exception as err:
                        err_resp = {"error": str(err)}
                        client_sock.sendall(json.dumps(err_resp).encode("utf-8"))
            except Exception as loop_err:
                if self.running:
                    logger.error("Error in daemon loop: %s", loop_err)

        self.cleanup()

    def cleanup(self):
        """Cleanup socket and unload VRAM."""
        if self.server_socket:
            try:
                self.server_socket.close()
            except Exception:
                pass
        if os.path.exists(self.socket_path):
            try:
                os.unlink(self.socket_path)
            except Exception:
                pass
        self.engine.unload_vram()
        logger.info("Daemon cleanly stopped.")


class DaemonClient:
    """Client communicating with OrchestratorDaemon with Stale Socket Recovery (Fix 3)."""

    def __init__(self, socket_path: str = SOCKET_PATH):
        self.socket_path = socket_path

    def _try_ping(self) -> bool:
        """Attempt to ping daemon with 100ms timeout."""
        if not os.path.exists(self.socket_path):
            return False

        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(0.1)  # 100ms ping timeout
        try:
            sock.connect(self.socket_path)
            sock.sendall(json.dumps({"action": "ping"}).encode("utf-8"))
            resp = json.loads(sock.recv(4096).decode("utf-8"))
            return resp.get("status") == "pong"
        except Exception:
            return False
        finally:
            try:
                sock.close()
            except Exception:
                pass

    def ensure_daemon(self):
        """Stale Socket Recovery (Fix 3): ping; if stale/unresponsive, unlink, spawn, and wait."""
        if self._try_ping():
            return

        # Stale socket or daemon not running
        if os.path.exists(self.socket_path):
            try:
                os.unlink(self.socket_path)
                logger.info("Unlinked stale socket: %s", self.socket_path)
            except OSError:
                pass

        # Spawn daemon detached in background
        daemon_script = os.path.join(os.path.dirname(__file__), "daemon.py")
        venv_python = sys.executable

        logger.info("Spawning orchestrator daemon...")
        subprocess.Popen(
            [venv_python, daemon_script],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )

        # Wait up to 1.0s for daemon to bind
        deadline = time.time() + 1.0
        while time.time() < deadline:
            time.sleep(0.04)
            if self._try_ping():
                logger.info("Daemon ready and responding to ping.")
                return

        logger.warning("Daemon did not respond within 1.0s; will fallback to in-process engine.")

    def send_request(self, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Send request to daemon with automatic retry and recovery."""
        self.ensure_daemon()

        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(3.0)
        try:
            sock.connect(self.socket_path)
            sock.sendall(json.dumps(payload).encode("utf-8"))
            raw = sock.recv(65536)
            if raw:
                return json.loads(raw.decode("utf-8"))
        except Exception as err:
            logger.warning("IPC to daemon failed (%s); using in-process execution", err)
            return None
        finally:
            try:
                sock.close()
            except Exception:
                pass
        return None


if __name__ == "__main__":
    daemon = OrchestratorDaemon()

    def sig_handler(sig, frame):
        daemon.running = False

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    daemon.start()
