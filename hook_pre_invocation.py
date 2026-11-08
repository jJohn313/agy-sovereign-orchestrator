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
                if '"USER_INPUT"' in line:
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
        prompt_lower = user_prompt.lower()
        if any(tag in prompt_lower for tag in ["mem0_bypass", "zero_persistence", "mem0 bypass", "zero persistence", "mem0 directive", "do not log this session"]):
            mem_gate = 0.0

        retrieval_text = ""
        if cb_gate >= 0.75 or mem_gate >= 0.75:
            try:
                retriever = TargetedVectorRetriever()
                ret_ctx = retriever.retrieve(
                    user_prompt=user_prompt,
                    codebase_gate=cb_gate,
                    mem0_gate=mem_gate,
                )
                formatted = ret_ctx.get("formatted_context", "")
                if formatted:
                    retrieval_text = f"### Retrieved Codebase References & Directives:\n{formatted}"
            except Exception as e:
                pass
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

        # Phase 2.5: Episodic Troubleshooting Fix Retrieval
        try:
            fix_directive = client.query_fixes(user_prompt, threshold=0.80)
            if fix_directive:
                directive_parts.append(f"### Verified Prior Troubleshooting Solutions:\n{fix_directive}")
        except Exception:
            pass

        # Detect Away / Headless / SSH Mode
        is_away = False
        away_reasons = []

        # 1. Semantic away intent (avoids false positives like 'refactor this away' or 'throw away')
        away_patterns = [
            r"\b(?:i\s+am|i\'?m|i\s+will\s+be|i\'?ll\s+be|going|heading|stepping|step)\s+away\b",
            r"\baway\s+from\s+(?:my\s+)?(?:desk|pc|computer|screen)\b",
            r"\bnot\s+(?:at|on|in\s+front\s+of)\s+(?:my\s+)?(?:desk|pc|computer|screen)\b",
            r"\b(?:leave|leaving)\s+(?:my\s+)?(?:desk|pc|computer)\b",
            r"\bafk\b",
        ]
        if any(re.search(pat, user_prompt, re.IGNORECASE) for pat in away_patterns):
            is_away = True
            away_reasons.append("Prompt indicates user is stepping away / not at desk")

        # 2. Super+Shift+H Headless mode flag
        if os.path.exists("/tmp/headless_away_mode"):
            is_away = True
            away_reasons.append("Headless Away power-saving mode active (/tmp/headless_away_mode)")

        # 3. All DPMS monitors off
        try:
            res_dpms = subprocess.run(["hyprctl", "monitors", "-j"], capture_output=True, text=True, timeout=2)
            if res_dpms.returncode == 0 and res_dpms.stdout.strip():
                mon_data = json.loads(res_dpms.stdout)
                if mon_data and all(not m.get("dpmsStatus", True) for m in mon_data):
                    is_away = True
                    away_reasons.append("All Hyprland monitors powered off (DPMS off)")
        except Exception:
            pass

        # 4. SSH session detected
        if os.getenv("SSH_CONNECTION") or os.getenv("SSH_CLIENT") or os.getenv("SSH_TTY"):
            is_away = True
            away_reasons.append("Connected via remote SSH session")

        if is_away:
            reasons_str = "; ".join(away_reasons)
            directive_parts.append(
                f"[REMOTE / AWAY MODE DETECTED: {reasons_str}]\n"
                f"John is NOT at his desk. When finishing a task, reaching a milestone, or requiring input:\n"
                f"1. Post summaries and screenshots into his dedicated Telegram Forum Topic using `/home/john/scripts/telegram_voice.py --summary ...`\n"
                f"2. High-priority voice briefing/decision can be dispatched via Telegram voice audio note or Discord voice.\n"
                f"Do not wait for terminal input at the desk."
            )

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
