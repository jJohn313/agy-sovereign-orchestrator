#!/usr/bin/env python3
"""
Argon Invocation Utility & Gemini 4 Frontier Gateway.
Routes surgical reasoning tasks directly to Gemini 4 Argon with minimal token usage.
Credentials loaded from ~/.config/gemini/api_key (mode 0600) or GEMINI_API_KEY env var.
"""

import sys
import os
import re
import json
import argparse
import subprocess
import urllib.request
import urllib.error
from typing import Optional, Dict, Any, List, Tuple

API_KEY_PATH = os.path.expanduser("~/.config/gemini/api_key")
ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/gemini-4-argon:generateContent"


def get_api_key() -> str:
    """Retrieve API key from env or ~/.config/gemini/api_key."""
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if key:
        return key
    if os.path.exists(API_KEY_PATH):
        try:
            with open(API_KEY_PATH, "r", encoding="utf-8") as f:
                key = f.read().strip()
                if key:
                    return key
        except Exception as e:
            sys.stderr.write(f"Error reading {API_KEY_PATH}: {e}\n")
    raise ValueError(
        f"Gemini API key not found. Ensure GEMINI_API_KEY is set or saved in {API_KEY_PATH}"
    )


def extract_model_text(candidate: Dict[str, Any]) -> str:
    """Safely extract and concatenate visible text parts, skipping internal thought blocks."""
    parts = candidate.get("content", {}).get("parts", [])
    text_chunks = [p["text"] for p in parts if "text" in p and not p.get("thought", False)]
    return "".join(text_chunks)


def apply_search_replace_block(content: str, search_block: str, replace_block: str) -> Tuple[bool, str, Optional[str]]:
    """
    Applies search/replace with a robust 3-tier fallback, ambiguity protection,
    and line-ending preservation (CRLF/LF).

    Returns:
        (success: bool, updated_content: str, error_reason: Optional[str])
    """
    newline = "\r\n" if "\r\n" in content else "\n"
    content_norm = content.replace("\r\n", "\n")
    search_norm = search_block.replace("\r\n", "\n")
    replace_norm = replace_block.replace("\r\n", "\n")

    # Tier 1: Exact match with ambiguity detection
    exact_count = content_norm.count(search_norm)
    if exact_count > 1:
        return False, content, f"Ambiguous match: SEARCH block appears {exact_count} times in target file."
    if exact_count == 1:
        res = content_norm.replace(search_norm, replace_norm, 1)
        return True, res.replace("\n", newline), None

    content_lines = content_norm.split("\n")
    search_lines = search_norm.split("\n")
    n_search = len(search_lines)

    if n_search > len(content_lines):
        return False, content, "SEARCH block is longer than target content."

    # Tier 2: Trailing whitespace tolerance (per-line rstrip)
    search_rstripped = [l.rstrip() for l in search_lines]
    matching_indices = []
    for i in range(len(content_lines) - n_search + 1):
        window = [l.rstrip() for l in content_lines[i : i + n_search]]
        if window == search_rstripped:
            matching_indices.append(i)

    if len(matching_indices) > 1:
        return False, content, f"Ambiguous match: SEARCH block matches {len(matching_indices)} locations under whitespace tolerance."
    if len(matching_indices) == 1:
        idx = matching_indices[0]
        new_lines = content_lines[:idx] + replace_norm.split("\n") + content_lines[idx + n_search:]
        return True, "\n".join(new_lines).replace("\n", newline), None

    # Tier 3: Indentation-relative shift
    def get_indent(l: str) -> int:
        return len(l) - len(l.lstrip())

    non_empty_search = [l for l in search_lines if l.strip()]
    if non_empty_search:
        search_base_indent = min(get_indent(l) for l in non_empty_search)
        search_relative = [l[search_base_indent:] if l.strip() else "" for l in search_lines]

        indent_matches = []
        for i in range(len(content_lines) - n_search + 1):
            window = content_lines[i : i + n_search]
            non_empty_window = [l for l in window if l.strip()]
            if not non_empty_window:
                continue
            window_base_indent = min(get_indent(l) for l in non_empty_window)
            window_relative = [l[window_base_indent:] if l.strip() else "" for l in window]

            if window_relative == search_relative:
                indent_matches.append((i, window_base_indent - search_base_indent))

        if len(indent_matches) > 1:
            return False, content, f"Ambiguous match: SEARCH block matches {len(indent_matches)} locations with relative indentation."
        if len(indent_matches) == 1:
            idx, delta = indent_matches[0]
            adjusted_replace = []
            for rl in replace_norm.split("\n"):
                if rl.strip():
                    adjusted_replace.append((" " * max(0, get_indent(rl) + delta)) + rl.lstrip())
                else:
                    adjusted_replace.append("")
            new_lines = content_lines[:idx] + adjusted_replace + content_lines[idx + n_search:]
            return True, "\n".join(new_lines).replace("\n", newline), None

    return False, content, "No matching block found in target content."


def call_argon_conversation(
    contents: List[Dict[str, Any]],
    system_instruction: Optional[str] = None,
    max_output_tokens: int = 8192,
    temperature: float = 0.7,
) -> Dict[str, Any]:
    """Execute multi-turn generateContent against Gemini 4 Argon endpoint."""
    api_key = get_api_key()
    url = f"{ENDPOINT}?key={api_key}"

    payload: Dict[str, Any] = {
        "contents": contents,
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_output_tokens,
        }
    }

    if system_instruction:
        payload["systemInstruction"] = {
            "parts": [{"text": system_instruction.strip()}]
        }

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = resp.read().decode("utf-8")
            return json.loads(body)
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"Argon API HTTP Error {e.code}: {error_body}") from e
    except Exception as e:
        raise RuntimeError(f"Argon API Request failed: {e}") from e


def call_argon(
    prompt: str,
    context: Optional[str] = None,
    system_instruction: Optional[str] = None,
    max_output_tokens: int = 8192,
    temperature: float = 0.7,
) -> Dict[str, Any]:
    """Single-turn helper routing through call_argon_conversation."""
    combined_prompt = prompt
    if context:
        combined_prompt = f"### CONTEXT & CODE SNIPPETS:\n{context.strip()}\n\n### TASK / REASONING REQUEST:\n{prompt.strip()}"

    contents = [{"role": "user", "parts": [{"text": combined_prompt}]}]
    return call_argon_conversation(
        contents=contents,
        system_instruction=system_instruction,
        max_output_tokens=max_output_tokens,
        temperature=temperature,
    )


def resolve_context_slice(ref: str) -> str:
    """Resolve a target file or symbol reference to an exact code slice."""
    ref = ref.strip().strip("'\"")
    path_part = ref
    line_start, line_end = None, None
    symbol = None

    if "#L" in ref:
        path_part, lines_spec = ref.split("#L", 1)
        if "-" in lines_spec:
            parts = lines_spec.split("-", 1)
            try:
                line_start = int(parts[0])
                line_end = int(parts[1].lstrip("L"))
            except ValueError:
                pass
        else:
            try:
                line_start = int(lines_spec)
                line_end = line_start + 30
            except ValueError:
                pass
    elif "::" in ref:
        path_part, symbol = ref.split("::", 1)

    target_path = os.path.expanduser(path_part)
    if not os.path.exists(target_path):
        try:
            proc = subprocess.run(["sem-find", path_part], text=True, capture_output=True, timeout=3)
            matches = re.findall(r"\[\d+\]\s+([^\s\(\)]+)", proc.stdout)
            if matches and os.path.exists(matches[0]):
                target_path = matches[0]
        except Exception:
            pass

    if not os.path.exists(target_path):
        return f"[ERROR: File not found: {path_part}]"

    try:
        with open(target_path, "r", encoding="utf-8", errors="ignore") as f:
            all_lines = f.readlines()

        if line_start is not None and line_end is not None:
            s_idx = max(0, line_start - 1)
            e_idx = min(len(all_lines), line_end)
            return "".join(all_lines[s_idx:e_idx])

        if symbol:
            pattern = re.compile(rf"\b(def|class|function|interface|type)\s+{re.escape(symbol)}\b")
            for idx, line in enumerate(all_lines):
                if pattern.search(line):
                    s_idx = max(0, idx - 2)
                    e_idx = min(len(all_lines), idx + 45)
                    return "".join(all_lines[s_idx:e_idx])

        if len(all_lines) > 120:
            return "".join(all_lines[:100]) + f"\n... [Truncated: {len(all_lines) - 100} lines omitted]\n"
        return "".join(all_lines)
    except Exception as e:
        return f"[ERROR reading {target_path}: {e}]"


def run_interactive_escalation(
    initial_prompt: str,
    initial_context: str,
    system_instruction: Optional[str] = None,
    max_turns: int = 4,
    temperature: float = 0.7,
) -> Tuple[str, Dict[str, Any]]:
    """
    Executes a multi-turn conversation loop with Argon.
    Batches multi-symbol context requests via re.findall, preserves candidate['content'],
    and accumulates full token usage.
    """
    first_user_text = f"### CONTEXT:\n{initial_context}\n\n### TASK:\n{initial_prompt}" if initial_context else initial_prompt
    messages: List[Dict[str, Any]] = [
        {"role": "user", "parts": [{"text": first_user_text}]}
    ]

    total_prompt_tokens = 0
    total_candidate_tokens = 0
    last_resp: Dict[str, Any] = {}

    for turn in range(max_turns):
        resp = call_argon_conversation(
            contents=messages,
            system_instruction=system_instruction,
            temperature=temperature,
        )
        last_resp = resp
        usage = resp.get("usageMetadata", {})
        total_prompt_tokens += usage.get("promptTokenCount", 0)
        total_candidate_tokens += usage.get("candidatesTokenCount", 0)

        candidates = resp.get("candidates", [])
        if not candidates:
            break

        candidate = candidates[0]
        # Append full candidate content directly to preserve native multi-part structure
        messages.append(candidate.get("content", {}))
        candidate_text = extract_model_text(candidate)

        # Batch resolve all requested context references in one turn
        needed_refs = re.findall(r"\[NEED_CONTEXT:\s*([^\s\]]+)\]", candidate_text)
        if not needed_refs:
            return candidate_text, {
                "prompt_tokens": total_prompt_tokens,
                "candidate_tokens": total_candidate_tokens,
                "total_tokens": total_prompt_tokens + total_candidate_tokens,
                "turns": turn + 1,
            }

        sys.stderr.write(f"[*] Turn {turn + 1}: Argon requested {len(needed_refs)} context slices: {', '.join(needed_refs)}\n")

        bundled_slices = []
        for ref in needed_refs:
            slice_data = resolve_context_slice(ref)
            bundled_slices.append(f"#### SLICE: {ref}\n```\n{slice_data}\n```")

        messages.append({
            "role": "user",
            "parts": [{"text": "### REQUESTED CONTEXT SLICES:\n" + "\n\n".join(bundled_slices)}]
        })

    final_text = extract_model_text(last_resp.get("candidates", [{}])[0])
    return final_text, {
        "prompt_tokens": total_prompt_tokens,
        "candidate_tokens": total_candidate_tokens,
        "total_tokens": total_prompt_tokens + total_candidate_tokens,
        "turns": max_turns,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Invoke Gemini 4 Argon for surgical reasoning tasks."
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        default="",
        help="Task prompt or reasoning question"
    )
    parser.add_argument(
        "--context-file",
        "-c",
        help="Path to file containing gathered code/context"
    )
    parser.add_argument(
        "--system",
        "-s",
        help="Custom system instruction"
    )
    parser.add_argument(
        "--max-tokens",
        "-m",
        type=int,
        default=8192,
        help="Maximum candidate output tokens (default: 8192)"
    )
    parser.add_argument(
        "--temperature",
        "-t",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON response from Argon"
    )
    args = parser.parse_args()

    # Read context if provided
    context_text = ""
    if args.context_file:
        if args.context_file == "-":
            context_text = sys.stdin.read()
        elif os.path.exists(args.context_file):
            with open(args.context_file, "r", encoding="utf-8", errors="ignore") as f:
                context_text = f.read()
        else:
            sys.stderr.write(f"Warning: Context file {args.context_file} not found.\n")

    prompt = args.prompt.strip()
    if not prompt:
        if not sys.stdin.isatty() and not args.context_file:
            prompt = sys.stdin.read().strip()
        else:
            parser.error("Prompt argument is required (or pipe via stdin)")

    try:
        resp = call_argon(
            prompt=prompt,
            context=context_text,
            system_instruction=args.system,
            max_output_tokens=args.max_tokens,
            temperature=args.temperature,
        )

        usage = resp.get("usageMetadata", {})
        prompt_tokens = usage.get("promptTokenCount", 0)
        cand_tokens = usage.get("candidatesTokenCount", 0)
        total_tokens = usage.get("totalTokenCount", 0)
        sys.stderr.write(
            f"[Argon Usage] Prompt: {prompt_tokens} | Output: {cand_tokens} | Total: {total_tokens} tokens\n"
        )

        if args.json:
            print(json.dumps(resp, indent=2))
            return

        candidates = resp.get("candidates", [])
        if not candidates:
            sys.stderr.write("No candidate returned from Argon.\n")
            sys.exit(1)

        parts = candidates[0].get("content", {}).get("parts", [])
        text_outputs = [p.get("text", "") for p in parts if "text" in p]
        print("".join(text_outputs))

    except Exception as e:
        sys.stderr.write(f"Error invoking Argon: {e}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
