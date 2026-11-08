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
import fcntl
from typing import Dict, Any, Optional, List

from local_decision_engine import LocalDecisionEngine

logger = logging.getLogger("agy.daemon")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [DAEMON] %(message)s")

SOCKET_PATH = os.path.expanduser("~/.config/agy/orchestrator.sock")
LOCK_PATH = os.path.expanduser("~/.config/agy/orchestrator.lock")
VENV_PYTHON = "/home/john/.config/agy/orchestrator/.venv/bin/python3"
IDLE_TIMEOUT_SECONDS = int(os.getenv("AGY_IDLE_TIMEOUT", "20"))  # Auto-exit after 20s idle to free RAM


class OrchestratorDaemon:
    """Unix Domain Socket Daemon hosting the sovereign decision engine."""

    def __init__(self, socket_path: str = SOCKET_PATH, lock_path: str = LOCK_PATH):
        self.socket_path = socket_path
        self.lock_path = lock_path
        self.lock_file = None
        self.engine = LocalDecisionEngine()
        self.running = False
        self.server_socket: Optional[socket.socket] = None
        self.last_activity_time = time.time()
        self.lock = threading.Lock()

    def check_idle(self, force_seconds: Optional[int] = None) -> bool:
        """
        Check if idle time exceeded limit.
        If idle > IDLE_TIMEOUT_SECONDS, offload VRAM and terminate daemon to release host RAM.
        """
        elapsed = force_seconds if force_seconds is not None else (time.time() - self.last_activity_time)
        if elapsed >= IDLE_TIMEOUT_SECONDS:
            if self.engine.model_loaded:
                self.engine.unload_vram()
            logger.info("Idle timeout reached (%.1fs >= %ds). Terminating daemon to reclaim RAM/VRAM.", elapsed, IDLE_TIMEOUT_SECONDS)
            self.running = False
            return True
        return False

    def _idle_monitor_loop(self):
        """Background thread checking for idle inactivity."""
        while self.running:
            time.sleep(2)
            with self.lock:
                if self.check_idle():
                    break

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
            state = f'The user asked: "{prompt}"' if prompt else f"WORKING_DIR: {cwd}"

            with self.engine.active_session():
                propositions = [
                    "The task requires using git or version control.",
                    "The task requires reading, writing, or viewing files or directories.",
                    "The task requires running terminal shell commands or tests.",
                    "The task requires searching the codebase or symbol definitions.",
                    "The user is stating a personal preference such as a preferred theme or habit.",
                ]
                scores = self.engine.noul_batch(state, propositions)

            return {
                "scores": {
                    "needs_git": round(scores[0], 4),
                    "needs_filesystem": round(scores[1], 4),
                    "needs_terminal": round(scores[2], 4),
                    "needs_codebase_search": round(scores[3], 4),
                    "needs_user_memory": round(scores[4], 4),
                }
            }

        elif action == "filter_post_turn":
            prompt = req.get("user_prompt", "")
            summary = req.get("summary", "")
            state = f'Exchange: User asked "{prompt}" and system responded: "{summary}"'
            with self.engine.active_session():
                durable = self.engine.noul(
                    state,
                    "This exchange establishes a permanent user preference, key architectural invariant, or durable rule."
                )
            return {"is_durable_knowledge": round(durable, 4)}

        elif action == "query_fixes":
            query = req.get("query", "")
            domain = req.get("domain", None)
            limit = req.get("limit", 2)
            threshold = req.get("threshold", 0.80)
            try:
                from fix_ledger import FixLedger
                fl = FixLedger()
                matches = fl.search_fixes(query, domain=domain, limit=limit, score_threshold=threshold)
                directive = fl.format_directive(matches) if matches else ""
                return {"status": "ok", "matches": matches, "directive": directive}
            except Exception as e:
                logger.error("Daemon query_fixes failed: %s", e)
                return {"status": "error", "message": str(e), "matches": [], "directive": ""}

        elif action == "record_fix":
            domain = req.get("domain", "")
            symptom = req.get("symptom", "")
            root_cause = req.get("root_cause", "")
            verified_fix = req.get("verified_fix", "")
            tags = req.get("tags", None)
            try:
                from fix_ledger import FixLedger
                fl = FixLedger()
                ok = fl.record_fix(domain, symptom, root_cause, verified_fix, tags=tags)
                return {"status": "ok", "recorded": ok}
            except Exception as e:
                logger.error("Daemon record_fix failed: %s", e)
                return {"status": "error", "message": str(e), "recorded": False}

        elif action == "check_idle":
            force_secs = req.get("force_seconds", 601)
            unloaded = self.check_idle(force_seconds=force_secs)
            return {"status": "idle_checked", "unloaded": unloaded, "model_loaded": self.engine.model_loaded}

        elif action == "async_reindex":
            repo_path = req.get("repo_path", "")
            if repo_path and os.path.exists(repo_path):
                def _bg_index():
                    try:
                        from code_indexer import IncrementalGitIndexer
                        indexer = IncrementalGitIndexer()
                        res = indexer.index_repository(repo_path)
                        logger.info("Async reindex completed for %s: %s", repo_path, res)
                    except Exception as e:
                        logger.error("Async reindex failed for %s: %s", repo_path, e)

                threading.Thread(target=_bg_index, daemon=True).start()
                return {"status": "enqueued", "repo_path": repo_path}
            return {"status": "error", "message": f"Invalid repo_path: {repo_path}"}

        elif action == "reload_tools":
            return {"status": "reloaded"}

        elif action == "shutdown":
            self.running = False
            return {"status": "shutting_down"}

        return {"error": f"Unknown action: {action}"}

    def start(self):
        """Bind Unix Domain Socket and enter event loop."""
        os.makedirs(os.path.dirname(self.lock_path), exist_ok=True)
        self.lock_file = open(self.lock_path, "a+")
        try:
            fcntl.flock(self.lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (BlockingIOError, OSError):
            logger.info("Another orchestrator daemon is already running (locked %s). Exiting.", self.lock_path)
            sys.exit(0)

        # Record PID in lock file
        self.lock_file.seek(0)
        self.lock_file.truncate()
        self.lock_file.write(f"{os.getpid()}\n")
        self.lock_file.flush()

        os.makedirs(os.path.dirname(self.socket_path), exist_ok=True)
        if os.path.exists(self.socket_path):
            try:
                os.unlink(self.socket_path)
            except OSError:
                pass

        self.server_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.server_socket.bind(self.socket_path)
        self.server_socket.listen(64)
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
        if self.lock_file:
            try:
                fcntl.flock(self.lock_file.fileno(), fcntl.LOCK_UN)
                self.lock_file.close()
            except Exception:
                pass
            if os.path.exists(self.lock_path):
                try:
                    os.unlink(self.lock_path)
                except Exception:
                    pass
        self.engine.unload_vram()
        logger.info("Daemon cleanly stopped.")


class DaemonClient:
    """Client communicating with OrchestratorDaemon with Singleton Safety & Stale Recovery."""

    def __init__(self, socket_path: str = SOCKET_PATH, lock_path: str = LOCK_PATH):
        self.socket_path = socket_path
        self.lock_path = lock_path

    def _try_ping(self, timeout: float = 1.5) -> bool:
        """Attempt to ping daemon with timeout."""
        if not os.path.exists(self.socket_path):
            return False

        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(timeout)
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

    def _is_locked(self) -> bool:
        """Check if another process holds the exclusive lock."""
        if not os.path.exists(self.lock_path):
            return False
        try:
            with open(self.lock_path, "r") as f:
                try:
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                    return False
                except (BlockingIOError, OSError):
                    return True
        except Exception:
            return False

    def ensure_daemon(self):
        """Ensure exactly one orchestrator daemon is running without duplicating instances."""
        if self._try_ping(timeout=1.0):
            return

        # If daemon holds the lock, it might be starting up or busy
        if self._is_locked():
            deadline = time.time() + 1.2
            while time.time() < deadline:
                time.sleep(0.15)
                if self._try_ping(timeout=0.5):
                    return
            # If still not pingable despite holding lock, socket file may be corrupt or desynced
            logger.warning("Daemon holds lock but fails ping; clearing socket and restarting service...")

        # Not responding and lock is free: try systemd service if available
        try:
            res = subprocess.run(
                ["systemctl", "--user", "is-active", "--quiet", "agy-orchestrator.service"],
                capture_output=True,
                timeout=2.0
            )
            if res.returncode == 0:
                # Service is marked active but socket failed to respond: restart to clear stale socket
                subprocess.run(
                    ["systemctl", "--user", "restart", "agy-orchestrator.service"],
                    capture_output=True,
                    timeout=3.0
                )
                deadline = time.time() + 6.0
                while time.time() < deadline:
                    time.sleep(0.2)
                    if self._try_ping(timeout=1.0):
                        return
            else:
                subprocess.run(
                    ["systemctl", "--user", "start", "agy-orchestrator.service"],
                    capture_output=True,
                    timeout=3.0
                )
                deadline = time.time() + 6.0
                while time.time() < deadline:
                    time.sleep(0.2)
                    if self._try_ping(timeout=1.0):
                        return
        except Exception:
            pass

        # Standalone fallback: clean up dead socket and spawn detached daemon
        if os.path.exists(self.socket_path):
            try:
                os.unlink(self.socket_path)
            except OSError:
                pass

        daemon_script = os.path.join(os.path.dirname(__file__), "daemon.py")
        venv_python = VENV_PYTHON if os.path.exists(VENV_PYTHON) else sys.executable

        logger.info("Spawning single orchestrator daemon...")
        subprocess.Popen(
            [venv_python, daemon_script],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )

        deadline = time.time() + 6.0
        while time.time() < deadline:
            time.sleep(0.1)
            if self._try_ping(timeout=1.0):
                logger.info("Daemon ready and responding to ping.")
                return

        logger.warning("Daemon did not respond within 6.0s; will fallback to in-process engine.")

    def send_request(self, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Send request to daemon with automatic retry and recovery."""
        self.ensure_daemon()

        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(15.0)
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

    def query_fixes(self, query: str, domain: Optional[str] = None, threshold: float = 0.80) -> str:
        """Query fix ledger via daemon or local fallback."""
        try:
            resp = self.send_request({
                "action": "query_fixes",
                "query": query,
                "domain": domain,
                "threshold": threshold,
            })
            if resp and resp.get("status") == "ok":
                return resp.get("directive", "")
        except Exception:
            pass

        # In-process fallback
        try:
            from fix_ledger import FixLedger
            fl = FixLedger()
            matches = fl.search_fixes(query, domain=domain, score_threshold=threshold)
            return fl.format_directive(matches) if matches else ""
        except Exception as e:
            logger.warning("query_fixes fallback failed: %s", e)
            return ""

    def record_fix(
        self,
        domain: str,
        symptom: str,
        root_cause: str,
        verified_fix: str,
        tags: Optional[List[str]] = None,
    ) -> bool:
        """Record verified fix via daemon or local fallback."""
        try:
            resp = self.send_request({
                "action": "record_fix",
                "domain": domain,
                "symptom": symptom,
                "root_cause": root_cause,
                "verified_fix": verified_fix,
                "tags": tags,
            })
            if resp and resp.get("status") == "ok":
                return resp.get("recorded", False)
        except Exception:
            pass

        # In-process fallback
        try:
            from fix_ledger import FixLedger
            fl = FixLedger()
            return fl.record_fix(domain, symptom, root_cause, verified_fix, tags=tags)
        except Exception as e:
            logger.warning("record_fix fallback failed: %s", e)
            return False


if __name__ == "__main__":
    daemon = OrchestratorDaemon()

    def sig_handler(sig, frame):
        daemon.running = False

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    daemon.start()
