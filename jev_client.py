"""
TypeSafe Jev System-1 Client
Sub-200ms deterministic routing and gating layer for AGY CLI environment.
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional
import requests

logger = logging.getLogger("agy.jev_client")

DEFAULT_JEV_URL = "https://thejevai.com/v1/systemone"
DEFAULT_MODEL = "typesafe/jev-latest"
DEFAULT_TIMEOUT = 1.5  # Strict 1.5s timeout requirement

# Fallback presets when Jev network fails or times out
FALLBACK_PRE_GATING = {
    "needs_git": 1.0,
    "needs_filesystem": 1.0,
    "needs_terminal": 1.0,
    "needs_codebase_search": 0.0,
    "needs_user_memory": 0.0,
}

FALLBACK_POST_GATING = {
    "is_durable_knowledge": 0.0,
}


def _load_config_env():
    """Load environment variables from ~/.config/agy/env if present."""
    env_file = os.path.expanduser("~/.config/agy/env")
    if os.path.exists(env_file):
        try:
            with open(env_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip('"').strip("'")
                        if k and k not in os.environ and not v.startswith("${"):
                            os.environ[k] = v
        except Exception:
            pass


def _heuristic_fallback_pre_gating(user_prompt: str) -> Dict[str, float]:
    p = user_prompt.lower()
    has_git = any(w in p for w in ["git", "branch", "commit", "diff", "repo", "stash", "checkout", "log"])
    has_fs = any(w in p for w in ["file", "dir", "read", "write", "edit", "folder"])
    has_term = any(w in p for w in ["terminal", "bash", "shell", "run ", "execute", "command", "script", "build", "diagnostics"])
    has_cb = any(w in p for w in ["codebase", "symbol", "function", "class", "semantic", "architecture"])
    has_mem = any(w in p for w in ["remember", "preference", "rule", "profile", "setting", "past "])

    # If prompt is a specific git request without fs or terminal, mount git dynamically
    if has_git and not has_fs and not has_term:
        return {
            "needs_git": 0.95,
            "needs_filesystem": 0.0,
            "needs_terminal": 0.0,
            "needs_codebase_search": 0.85 if has_cb else 0.0,
            "needs_user_memory": 0.85 if has_mem else 0.0,
        }

    # Otherwise fallback to mounting baseline dev tools (git, filesystem, terminal)
    return dict(FALLBACK_PRE_GATING)


class JevClient:
    """Client for TypeSafe Jev System-1 routing and gating."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        timeout: float = DEFAULT_TIMEOUT,
    ):
        _load_config_env()
        self.api_key = api_key or os.getenv("TYPESAFE_API_KEY") or os.getenv("JEV_API_KEY", "")
        self.api_url = api_url or os.getenv("TYPESAFE_API_URL") or os.getenv("JEV_API_URL", DEFAULT_JEV_URL)
        self.model = model
        self.timeout = timeout

    def _post(self, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Perform HTTP POST with strict timeout and fallback handling."""
        if not self.api_key:
            return None

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "x-api-key": self.api_key,
        }

        try:
            response = requests.post(
                self.api_url,
                json=payload,
                headers=headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, requests.Timeout, ValueError) as err:
            logger.warning("Jev request failed or timed out (%s): %s", type(err).__name__, err)
            return None

    def _parse_scores(self, data: Optional[Dict[str, Any]], expected_keys: List[str]) -> Dict[str, float]:
        """Extract float scores from various potential Jev response formats."""
        if not data:
            return {}

        results: Dict[str, float] = {}
        candidates = [
            data.get("predictions"),
            data.get("results"),
            data.get("answers"),
            data.get("decisions"),
            data,
        ]

        found_dict: Dict[str, Any] = {}
        for c in candidates:
            if isinstance(c, dict) and any(k in c for k in expected_keys):
                found_dict = c
                break

        for key in expected_keys:
            if key in found_dict:
                val = found_dict[key]
                if isinstance(val, (int, float)):
                    results[key] = float(val)
                elif isinstance(val, dict):
                    score = val.get("score", val.get("probability", val.get("confidence", val.get("value", 0.0))))
                    results[key] = float(score)
                elif isinstance(val, bool):
                    results[key] = 1.0 if val else 0.0
                elif isinstance(val, str):
                    try:
                        results[key] = float(val)
                    except ValueError:
                        results[key] = 1.0 if val.lower() in ("yes", "true", "y") else 0.0

        return results

    def gate_pre_turn(
        self,
        user_prompt: str,
        cwd: str,
        available_tools: List[str],
    ) -> Dict[str, float]:
        """
        Execute Pre-Execution Turn Gater.
        Evaluates prompt for git, filesystem, terminal, codebase search, and memory lookup.
        """
        comma_separated_tools = ", ".join(available_tools)
        state_text = (
            f"WORKING_DIR: {cwd}\n"
            f"AVAILABLE_LOCAL_CAPABILITIES: {comma_separated_tools}\n"
            f"USER_PROMPT: {user_prompt}"
        )

        payload = {
            "model": self.model,
            "state": state_text,
            "questions": {
                "needs_git": {
                    "type": "noul",
                    "instructions": "Does this prompt require git commands, repository status, commits, or branch diffs?",
                },
                "needs_filesystem": {
                    "type": "noul",
                    "instructions": "Does this prompt require inspecting, reading, or writing specific files or directories?",
                },
                "needs_terminal": {
                    "type": "noul",
                    "instructions": "Does this prompt require running shell commands, compilation steps, system diagnostics, or scripts from ~/tools?",
                },
                "needs_codebase_search": {
                    "type": "noul",
                    "instructions": "Does this task require searching the semantic codebase index for symbol or function definitions?",
                },
                "needs_user_memory": {
                    "type": "noul",
                    "instructions": "Does this task rely on past user preferences, architectural rules, or personalized setup history?",
                },
            },
        }

        keys = ["needs_git", "needs_filesystem", "needs_terminal", "needs_codebase_search", "needs_user_memory"]
        data = self._post(payload)
        parsed = self._parse_scores(data, keys)

        # Merge with fallback if missing
        if parsed:
            scores = dict(FALLBACK_PRE_GATING)
            for k in keys:
                if k in parsed:
                    scores[k] = parsed[k]
        else:
            scores = _heuristic_fallback_pre_gating(user_prompt)

        return scores

    def filter_post_turn(
        self,
        user_prompt: str,
        agent_actions_summary: str,
    ) -> Dict[str, float]:
        """
        Execute Post-Execution Write Filter (Mem0 Guard).
        Evaluates if the exchange established permanent preferences or architectural rules.
        """
        state_text = (
            f"USER_PROMPT: {user_prompt}\n"
            f"RESPONSE_SUMMARY: {agent_actions_summary}"
        )

        payload = {
            "model": self.model,
            "state": state_text,
            "questions": {
                "is_durable_knowledge": {
                    "type": "noul",
                    "instructions": (
                        "Does this exchange establish a permanent user preference, a key architectural invariant, "
                        "or a durable system config update that should be remembered permanently?"
                    ),
                }
            },
        }

        keys = ["is_durable_knowledge"]
        data = self._post(payload)
        parsed = self._parse_scores(data, keys)

        scores = dict(FALLBACK_POST_GATING)
        if parsed and "is_durable_knowledge" in parsed:
            scores["is_durable_knowledge"] = parsed["is_durable_knowledge"]

        return scores
