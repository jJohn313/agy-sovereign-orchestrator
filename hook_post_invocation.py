#!/usr/bin/env python3
"""
Sovereign Orchestrator - PostInvocation Hook for Antigravity CLI / AGY.
Fires after model & tool execution completes on every turn.
Runs change detection, triggers sem-index for mutated workspace files,
and evaluates durable knowledge filtering with DeBERTa-v3 to write to Mem0.
"""

import sys
import os
import json
import re
import subprocess

ORCH_DIR = "/home/john/.config/agy/orchestrator"
if ORCH_DIR not in sys.path:
    sys.path.insert(0, ORCH_DIR)

from daemon import DaemonClient


def extract_user_request(raw_text: str) -> str:
    if not raw_text:
        return ""
    match = re.search(r"<USER_REQUEST>(.*?)</USER_REQUEST>", raw_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return raw_text.strip()


def inspect_recent_transcript(transcript_path: str):
    """Extract latest prompt and actions from transcript."""
    if not transcript_path or not os.path.exists(transcript_path):
        return None, []
    try:
        with open(transcript_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
        user_prompt = ""
        actions = []
        for line in reversed(lines):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                stype = data.get("type")
                if stype == "USER_INPUT" and not user_prompt:
                    user_prompt = extract_user_request(data.get("content", ""))
                    break
                elif stype == "PLANNER_RESPONSE":
                    if "tool_calls" in data:
                        for tc in data.get("tool_calls", []):
                            actions.append(tc.get("name", ""))
            except Exception:
                continue
        return user_prompt, actions
    except Exception:
        return None, []


def main():
    try:
        raw_input = sys.stdin.read()
        if not raw_input:
            print("{}")
            return
        payload = json.loads(raw_input)
    except Exception:
        print("{}")
        return

    transcript_path = payload.get("transcriptPath", "")
    workspace_paths = payload.get("workspacePaths", [])
    cwd = workspace_paths[0] if workspace_paths else os.getcwd()

    user_prompt, actions = inspect_recent_transcript(transcript_path)
    if not user_prompt:
        print("{}")
        return

    try:
        # Check if durable memory filter is needed (when user prompt contains directive/remember/preference keywords)
        keywords = ["remember", "always", "never", "preference", "rule", "policy", "config", "architectural", "invariant"]
        if any(kw in user_prompt.lower() for kw in keywords):
            client = DaemonClient()
            client.ensure_daemon()
            resp = client.send_request({
                "action": "filter_post_turn",
                "user_prompt": user_prompt,
                "summary": f"Executed actions: {', '.join(actions) if actions else 'Discussion'}",
            })
            if resp and resp.get("is_durable_knowledge", 0.0) >= 0.85:
                # Add to Mem0
                subprocess.run(
                    ["mem0-cli", "add", f"User Directive: {user_prompt}"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=5,
                )
    except Exception:
        pass

    print("{}")


if __name__ == "__main__":
    main()
