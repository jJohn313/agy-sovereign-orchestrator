#!/usr/bin/env python3
"""
Sovereign Orchestrator - PreInvocation Hook for Antigravity CLI / AGY.
Fires before model invocation on every turn.
Runs DeBERTa-v3-large neural gating on CUDA (on-demand VRAM, instant unload),
runs targeted vector retrieval (Mem0 + codebase), and injects dynamic directives.
"""

import sys
import os
import json
import re

# Add orchestrator directory to python path
ORCH_DIR = "/home/john/.config/agy/orchestrator"
if ORCH_DIR not in sys.path:
    sys.path.insert(0, ORCH_DIR)

from daemon import DaemonClient
from retrieval import TargetedVectorRetriever


def extract_user_request(raw_text: str) -> str:
    """Extract clean user request text from raw transcript content."""
    if not raw_text:
        return ""
    match = re.search(r"<USER_REQUEST>(.*?)</USER_REQUEST>", raw_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return raw_text.strip()


def get_latest_user_prompt(transcript_path: str) -> str:
    """Read backwards from transcript to find the latest USER_INPUT."""
    if not transcript_path or not os.path.exists(transcript_path):
        return ""
    try:
        with open(transcript_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
        for line in reversed(lines):
            line = line.strip()
            if not line:
                continue
            try:
                if '"type":"USER_INPUT"' in line:
                    data = json.loads(line)
                    if data.get("type") == "USER_INPUT":
                        content = data.get("content", "")
                        return extract_user_request(content)
            except Exception:
                continue
    except Exception:
        pass
    return ""


def main():
    try:
        raw_input = sys.stdin.read()
        if not raw_input:
            print(json.dumps({"injectSteps": []}))
            return

        payload = json.loads(raw_input)
    except Exception:
        print(json.dumps({"injectSteps": []}))
        return

    invocation_num = payload.get("invocationNum", 1)

    # Only run DeBERTa gating on the initial model call for a user prompt (invocation 1)
    if invocation_num > 1:
        print(json.dumps({"injectSteps": []}))
        return

    transcript_path = payload.get("transcriptPath", "")
    workspace_paths = payload.get("workspacePaths", [])
    cwd = workspace_paths[0] if workspace_paths else os.getcwd()

    user_prompt = get_latest_user_prompt(transcript_path)
    if not user_prompt:
        print(json.dumps({"injectSteps": []}))
        return

    try:
        # Phase 1: DeBERTa Neural Gating via Daemon
        client = DaemonClient()
        client.ensure_daemon()

        resp = client.send_request({
            "action": "gate_pre_turn",
            "user_prompt": user_prompt,
            "cwd": cwd,
            "available_tools": ["git", "filesystem", "terminal", "codebase", "memory"],
        })

        scores = resp.get("scores", {}) if resp else {}

        # Phase 2: Targeted Vector Retrieval
        cb_gate = scores.get("needs_codebase_search", 0.0)
        mem_gate = scores.get("needs_user_memory", 0.0)

        retrieval_text = ""
        if cb_gate >= 0.75 or mem_gate >= 0.75:
            try:
                retriever = TargetedVectorRetriever()
                ret_ctx = retriever.retrieve(
                    user_prompt=user_prompt,
                    codebase_gate=cb_gate,
                    mem0_gate=mem_gate,
                )
                parts = []
                mem_matches = ret_ctx.get("mem0_matches", [])
                if mem_matches:
                    parts.append("### Retrieved User Memory / Directives:\n" + "\n".join(mem_matches))
                cb_matches = ret_ctx.get("codebase_matches", [])
                if cb_matches:
                    parts.append("### Retrieved Codebase References:\n" + "\n".join(cb_matches))
                retrieval_text = "\n\n".join(parts)
            except Exception:
                pass

        # Phase 3: Build Injected Directives
        gating_summary = ", ".join(f"{k.replace('needs_', '')}={v:.2f}" for k, v in scores.items())
        directive_parts = [
            f"[Sovereign Orchestrator | DeBERTa-v3 Neural Gating Active]",
            f"Gating: {gating_summary}",
            "Mandatory Policies: Ponytail minimal diff policy strictly enforced (no redundant prose, minimal targeted edits, stdlib/native first). Silent reuse-first gate. Search before reading whole files.",
        ]

        if retrieval_text:
            directive_parts.append(retrieval_text)

        injected_message = "\n".join(directive_parts)

        result = {
            "injectSteps": [
                {
                    "ephemeralMessage": injected_message
                }
            ]
        }
        print(json.dumps(result))

    except Exception:
        # Fail safe: never crash the CLI
        print(json.dumps({"injectSteps": []}))


if __name__ == "__main__":
    main()
