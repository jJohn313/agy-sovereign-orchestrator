"""
Log Sanitizer and Semantic Error-Anchored Traceback Folder (Fix 10).
Strips ANSI codes, detects critical error anchors, and preserves failure context.
"""

import re
from typing import List, Optional, Tuple

ANSI_REGEX = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")

ERROR_ANCHORS = [
    "Traceback (most recent call last):",
    "Error:",
    "FAILED",
    "Exception:",
    "panic!",
    "assert",
    "fatal:",
]


def strip_ansi(text: str) -> str:
    """Strip ANSI escape sequences and color codes."""
    return ANSI_REGEX.sub("", text)


def find_first_anchor(lines: List[str]) -> Optional[int]:
    """Find the 0-indexed line number of the first critical error anchor."""
    for idx, line in enumerate(lines):
        line_clean = line.strip()
        for anchor in ERROR_ANCHORS:
            if anchor.lower() in line_clean.lower():
                return idx
    return None


def fold_log_output(
    output: str,
    exit_code: int = 0,
    max_lines: int = 45,
    setup_lines_count: int = 11,
    context_before: int = 5,
    context_after: int = 26,
) -> str:
    """
    Semantic Error-Anchored Folding:
    1. If exit_code == 0 and output indicates a clean success banner, collapse to 1 line.
    2. If lines <= 45, return full stripped output.
    3. If lines > 45 and anchor detected:
       Keep lines 0..10, fold gap, keep (L-5)..(L+25).
    4. If lines > 45 and no anchor:
       Keep first 10 and last 30 lines.
    """
    clean_text = strip_ansi(output).strip()
    if not clean_text:
        return "[Clean exit: 0 (no output)]" if exit_code == 0 else f"[Process exited with code {exit_code}]"

    lines = clean_text.splitlines()
    total_lines = len(lines)

    # Clean success pass
    if exit_code == 0 and total_lines <= 5:
        lower_full = clean_text.lower()
        if any(term in lower_full for term in ["ok", "success", "passed", "complete", "finished"]):
            return f"✓ {lines[-1].strip() or 'Command succeeded (exit code 0)'}"

    if total_lines <= max_lines:
        return "\n".join(lines)

    anchor_idx = find_first_anchor(lines)

    if anchor_idx is not None:
        # Keep lines 0..10 (setup lines)
        head_end = min(setup_lines_count, total_lines)
        head = lines[:head_end]

        # Calculate error window
        err_start = max(head_end, anchor_idx - context_before)
        err_end = min(total_lines, anchor_idx + context_after)
        error_chunk = lines[err_start:err_end]

        result = list(head)
        folded_count = err_start - head_end
        if folded_count > 0:
            result.append(f"... [Folded {folded_count} lines] ...")
        result.extend(error_chunk)

        if err_end < total_lines:
            remaining = total_lines - err_end
            result.append(f"... [Folded {remaining} trailing lines] ...")

        return "\n".join(result)
    else:
        # No anchor found: keep first 10 and last 30 lines
        head = lines[:10]
        tail = lines[-30:]
        folded_count = total_lines - 40
        return "\n".join(head + [f"... [Folded {folded_count} lines] ..."] + tail)
