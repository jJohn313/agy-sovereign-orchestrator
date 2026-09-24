"""
Post-Turn Hook, Dual-Mode Change Detection & SQLite WAL Protection (Fix 2 & Fix 9).
Features:
- Dual-mode workspace change detection (Git + mtime)
- SQLite WAL concurrency protection on ~/.cache/workspaces_vec.db
- Mem0 write-gate strictly enforcing > 0.85 threshold
"""

import os
import time
import shutil
import sqlite3
import subprocess
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from jev_client import JevClient
from meta_tools import LocalToolRegistry

logger = logging.getLogger("agy.post_turn_hook")

DURABLE_KNOWLEDGE_THRESHOLD = 0.85
WORKSPACES_DB_PATH = os.path.expanduser("~/.cache/workspaces_vec.db")
LAST_TURN_TS_FILE = os.path.expanduser("~/.cache/agy/last_turn_ts")


def ensure_sqlite_wal(db_path: str = WORKSPACES_DB_PATH) -> bool:
    """Ensure SQLite database is configured with WAL mode and 5000ms busy_timeout."""
    if not os.path.exists(db_path):
        return False
    try:
        conn = sqlite3.connect(db_path, timeout=5.0)
        mode = conn.execute("PRAGMA journal_mode = WAL;").fetchone()[0]
        conn.execute("PRAGMA busy_timeout = 5000;")
        conn.close()
        return mode.lower() == "wal"
    except Exception as e:
        logger.warning("Could not set WAL pragma on %s: %s", db_path, e)
        return False


class PostTurnHook:
    """Evaluates turn outputs, executes dual-mode change detection, and gates Mem0 commits."""

    def __init__(
        self,
        jev_client: Optional[JevClient] = None,
        registry: Optional[LocalToolRegistry] = None,
        user_id: str = "john",
    ):
        self._custom_jev = jev_client is not None
        self.jev_client = jev_client or JevClient()
        self.registry = registry or LocalToolRegistry()
        self.user_id = user_id
        self._memory = None
        ensure_sqlite_wal(WORKSPACES_DB_PATH)

    def _get_memory(self):
        if self._memory is None:
            os.environ["MEM0_TELEMETRY"] = "false"
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            try:
                from mem0 import Memory
                mem_config = {
                    "vector_store": {
                        "provider": "qdrant",
                        "config": {
                            "path": os.path.expanduser("~/.local/share/mem0/qdrant"),
                            "collection_name": "agent_memories",
                            "embedding_model_dims": 384,
                        },
                    },
                    "history_db_path": os.path.expanduser("~/.local/share/mem0/history.db"),
                    "embedder": {
                        "provider": "fastembed",
                        "config": {"model": "BAAI/bge-small-en-v1.5"},
                    },
                    "llm": {
                        "provider": "openai",
                        "config": {"api_key": "dummy-local-key"},
                    },
                    "version": "v1.1",
                }
                self._memory = Memory.from_config(mem_config)
            except Exception as e:
                logger.warning("Could not instantiate direct Mem0 client: %s", e)
                self._memory = None
        return self._memory

    def _commit_memory(self, text: str) -> bool:
        """Commit memory to Mem0 via direct API or mem0-cli fallback."""
        memory = self._get_memory()
        if memory:
            try:
                memory.add(text, user_id=self.user_id, infer=False)
                logger.info("Committed durable memory to Mem0: %s", text)
                return True
            except Exception as err:
                logger.error("Direct Mem0 commit failed: %s", err)

        try:
            res = subprocess.run(
                ["mem0-cli", "add", text, "--user", self.user_id],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if res.returncode == 0:
                logger.info("Committed durable memory via mem0-cli: %s", text)
                return True
        except Exception as err:
            logger.warning("mem0-cli commit fallback failed: %s", err)

        return False

    def detect_changes_and_reindex(self, cwd: Optional[str] = None) -> Dict[str, Any]:
        """
        Dual-Mode Change Detection (Fix 2):
        1. Check: git rev-parse --is-inside-work-tree
        2. Git Workspace: If git status --porcelain has output -> sem-index $(git rev-parse --show-toplevel)
        3. Non-Git Workspace: Check file mtime against ~/.cache/agy/last_turn_ts -> sem-index $(pwd)
        """
        target_dir = os.path.abspath(cwd or os.getcwd())
        ensure_sqlite_wal(WORKSPACES_DB_PATH)

        # 1. Check Git
        is_git = False
        toplevel = None
        try:
            res_git = subprocess.run(
                ["git", "rev-parse", "--is-inside-work-tree"],
                cwd=target_dir,
                capture_output=True,
                text=True,
            )
            is_git = res_git.returncode == 0 and "true" in res_git.stdout.strip().lower()
            if is_git:
                top_res = subprocess.run(
                    ["git", "rev-parse", "--show-toplevel"],
                    cwd=target_dir,
                    capture_output=True,
                    text=True,
                )
                toplevel = top_res.stdout.strip()
        except Exception:
            is_git = False

        if is_git and toplevel:
            status_res = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=target_dir,
                capture_output=True,
                text=True,
            )
            if status_res.stdout.strip():
                if shutil.which("sem-index"):
                    logger.info("Git modifications detected in %s; triggering sem-index", toplevel)
                    subprocess.run(["sem-index", toplevel], capture_output=True)
                else:
                    logger.warning("Git modifications detected in %s, but 'sem-index' is not installed or not in PATH.", toplevel)
                return {"mode": "git", "reindexed": True, "path": toplevel}
            return {"mode": "git", "reindexed": False, "path": toplevel}

        # 2. Non-Git Workspace (mtime detection)
        os.makedirs(os.path.dirname(LAST_TURN_TS_FILE), exist_ok=True)
        last_ts = 0.0
        if os.path.exists(LAST_TURN_TS_FILE):
            try:
                with open(LAST_TURN_TS_FILE, "r", encoding="utf-8") as f:
                    last_ts = float(f.read().strip())
            except Exception:
                last_ts = 0.0

        max_mtime = 0.0
        try:
            for root, dirs, files in os.walk(target_dir):
                dirs[:] = [d for d in dirs if not d.startswith(".") and d != "__pycache__"]
                for fname in files:
                    if fname.startswith("."):
                        continue
                    fpath = os.path.join(root, fname)
                    try:
                        mtime = os.path.getmtime(fpath)
                        if mtime > max_mtime:
                            max_mtime = mtime
                    except OSError:
                        pass
        except Exception as e:
            logger.warning("Error checking mtime in %s: %s", target_dir, e)

        current_time = time.time()
        # If files were modified after last_ts, or first turn in non-git dir
        if max_mtime > last_ts or last_ts == 0.0:
            try:
                with open(LAST_TURN_TS_FILE, "w", encoding="utf-8") as f:
                    f.write(str(current_time))
            except Exception:
                pass

            if shutil.which("sem-index"):
                logger.info("Non-Git workspace changes detected via mtime in %s; triggering sem-index", target_dir)
                subprocess.run(["sem-index", target_dir], capture_output=True)
            else:
                logger.warning("Non-Git workspace changes detected via mtime in %s, but 'sem-index' is not installed or not in PATH.", target_dir)

            return {"mode": "non_git_mtime", "reindexed": True, "path": target_dir}

        try:
            with open(LAST_TURN_TS_FILE, "w", encoding="utf-8") as f:
                f.write(str(current_time))
        except Exception:
            pass
        return {"mode": "non_git_mtime", "reindexed": False, "path": target_dir}

    def process_turn_completion(
        self,
        user_prompt: str,
        actions_summary: str,
        mutated_paths: Optional[List[str]] = None,
        cwd: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Phase 4: Downstream Write-Gate & Dual-Mode Change Detection.
        """
        # 1. Update registry delta
        if mutated_paths:
            self.registry.update_registry_delta(mutated_paths)

        # 2. Dual-mode change detection
        change_info = self.detect_changes_and_reindex(cwd=cwd)

        # 3. Downstream Jev/NLI Write Filter (via IPC daemon if available and not custom mocked)
        durable_score = 0.0
        if not self._custom_jev:
            try:
                from daemon import DaemonClient
                client = DaemonClient()
                resp = client.send_request({
                    "action": "filter_post_turn",
                    "user_prompt": user_prompt,
                    "summary": actions_summary,
                })
                if resp and "is_durable_knowledge" in resp:
                    durable_score = float(resp["is_durable_knowledge"])
            except Exception:
                pass

        if durable_score == 0.0:
            scores = self.jev_client.filter_post_turn(
                user_prompt=user_prompt,
                agent_actions_summary=actions_summary,
            )
            durable_score = scores.get("is_durable_knowledge", 0.0)

        result = {
            "durable_score": durable_score,
            "committed": False,
            "memory_text": None,
            "change_detection": change_info,
        }

        if durable_score > DURABLE_KNOWLEDGE_THRESHOLD:
            memory_text = f"Preference/Rule established in prompt '{user_prompt}': {actions_summary}"
            committed = self._commit_memory(memory_text)
            result["committed"] = committed
            result["memory_text"] = memory_text

        return result
