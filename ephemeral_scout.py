import os
import json
import logging
import subprocess
import urllib.request
import urllib.error
import fcntl
import time
from contextlib import contextmanager
from typing import List

import re

logger = logging.getLogger("agy.ephemeral_scout")
LOCK_PATH = "/tmp/agy_qwen_scout.lock"


@contextmanager
def _acquire_qwen_lock(timeout: float = 3.0):
    """Ensure strictly single-instance execution of Qwen across AGY instances."""
    lock_file = None
    acquired = False
    start_time = time.time()
    try:
        lock_file = open(LOCK_PATH, "a+")
        while time.time() - start_time < timeout:
            try:
                fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
                break
            except (BlockingIOError, OSError):
                time.sleep(0.05)
        if not acquired:
            logger.warning("Another AGY instance is executing Qwen scout; skipping to prevent concurrent VRAM contention")
            yield False
            return
        yield True
    finally:
        if acquired and lock_file:
            try:
                fcntl.flock(lock_file, fcntl.LOCK_UN)
            except Exception:
                pass
        if lock_file:
            try:
                lock_file.close()
            except Exception:
                pass

def compute_jaccard_similarity(text_a: str, text_b: str) -> float:
    """Computes word-level Jaccard similarity between two strings."""
    tokens_a = set(re.findall(r"\w+", text_a.lower()))
    tokens_b = set(re.findall(r"\w+", text_b.lower()))
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)
    return intersection / union if union > 0 else 0.0

class EphemeralScout:
    """Two-tier localization scout with SLM for finding the correct files."""

    def __init__(self, model_name: str = "qwen2.5-coder:1.5b", base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url

    def is_available(self) -> bool:
        """Probe Ollama with 100ms timeout."""
        try:
            req = urllib.request.Request(f"{self.base_url}/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=0.1) as response:
                return response.status == 200
        except Exception as e:
            logger.debug(f"Ollama not available: {e}")
            return False

    def _score_candidate(self, query: str, path: str) -> float:
        """Weight basename higher than the full directory path."""
        filename = os.path.basename(path)
        return compute_jaccard_similarity(query, filename) * 2.0 + compute_jaccard_similarity(query, path)

    def _generate_candidates(self, query: str, cwd: str) -> List[str]:
        """Generate up to 15 candidates using cascading tiers."""
        candidates = set()

        # Tier 1: git ls-files
        try:
            res = subprocess.run(
                ["git", "ls-files"],
                cwd=cwd, capture_output=True, text=True, timeout=1.0
            )
            if res.returncode == 0 and res.stdout:
                candidates.update(res.stdout.splitlines())
        except Exception as e:
            logger.debug(f"git ls-files failed: {e}")

        # Tier 2: rg --files
        if not candidates:
            try:
                import shutil
                if shutil.which("rg"):
                    res = subprocess.run(
                        ["rg", "--files"],
                        cwd=cwd, capture_output=True, text=True, timeout=1.0
                    )
                    if res.returncode == 0 and res.stdout:
                        candidates.update(res.stdout.splitlines())
            except Exception as e:
                logger.debug(f"rg --files failed: {e}")

        # Tier 3: os.walk bounded
        if not candidates:
            exclude_dirs = {".git", "node_modules", "target", "dist", ".venv", "__pycache__", ".cache", "build"}
            for root, dirs, files in os.walk(cwd):
                dirs[:] = [d for d in dirs if d not in exclude_dirs]
                for f in files:
                    rel_path = os.path.relpath(os.path.join(root, f), cwd)
                    candidates.add(rel_path)

        candidates_list = [c for c in candidates if c]

        if len(candidates_list) <= 15:
            return candidates_list

        # Score and sort if > 15
        scored = [(c, self._score_candidate(query, c)) for c in candidates_list]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [c for c, _ in scored[:15]]

    def purge_vram(self) -> None:
        """Force Ollama to immediately purge the model from VRAM."""
        try:
            req = urllib.request.Request(
                f"{self.base_url}/api/generate",
                data=json.dumps({"model": self.model_name, "keep_alive": 0}).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=1.5):
                pass
        except Exception as e:
            logger.debug(f"Failed to purge {self.model_name} from VRAM: {e}")

    def resolve_scope(self, query: str, cwd: str) -> List[str]:
        """Use SLM to pick 1-3 most critical paths, anti-hallucination filtered."""
        candidates = self._generate_candidates(query, cwd)
        if not candidates:
            return []

        prompt = (
            f"User Query: {query}\n\n"
            f"Candidate Files:\n"
            + "\n".join(f"- {c}" for c in candidates) +
            "\n\nBased on the user query, select 1 to 3 most critical file paths from the candidate list above. "
            "Return the paths strictly in JSON format as a list of strings under the key 'targets'. "
            "Example: {\"targets\": [\"path/to/file.py\"]}"
        )

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "format": "json",
            "stream": False,
            "keep_alive": 0
        }

        with _acquire_qwen_lock(timeout=3.0) as locked:
            if not locked:
                return []

            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                f"{self.base_url}/api/generate",
                data=data,
                headers={"Content-Type": "application/json"}
            )

            try:
                with urllib.request.urlopen(req, timeout=2.5) as response:
                    result = json.loads(response.read().decode("utf-8"))

                    resp_obj = json.loads(result.get("response", "{}"))
                    slm_targets = resp_obj.get("targets", [])

                    # Anti-hallucination filter
                    authoritative_targets = [
                        p for p in slm_targets
                        if p in candidates and os.path.exists(os.path.join(cwd, p))
                    ]

                    return authoritative_targets
            except Exception as e:
                logger.warning(f"EphemeralScout resolution skipped or failed: {e}; falling back to standard discovery")
                return []
            finally:
                self.purge_vram()
