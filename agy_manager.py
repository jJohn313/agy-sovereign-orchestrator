#!/usr/bin/env python3
"""
Antigravity (AGY) Conversation Manager & Interactive Selector
Provides:
  - Accurate conversation titling and latest-work extraction
  - Fast, responsive curses-based interactive terminal selector
  - Command-line conversation listing, renaming, and auto-titling
"""

import os
import sys
import json
import re
import sqlite3
import curses
from datetime import datetime, timezone

DB_PATH = os.path.expanduser("~/.gemini/antigravity-cli/conversation_summaries.db")
BRAIN_DIR = os.path.expanduser("~/.gemini/antigravity-cli/brain")
REAL_BIN = os.path.expanduser("~/.local/bin/agy-bin")

def set_terminal_title(title):
    """Set the terminal window title (e.g. Kitty, Alacritty, terminal)."""
    try:
        if sys.stdout.isatty():
            sys.stdout.write(f"\033]2;AGY - {title}\007")
            sys.stdout.flush()
    except Exception:
        pass

def get_relative_time(dt_str):
    """Convert an ISO-like timestamp string to human-friendly relative time."""
    try:
        dt_str = dt_str.replace(" ", "T")
        if "+" in dt_str:
            base, tz = dt_str.split("+", 1)
            if "." in base:
                main, ms = base.split(".", 1)
                dt_str = f"{main}.{ms[:6]}+{tz}"
            else:
                dt_str = f"{base}+{tz}"
        elif dt_str.endswith("Z"):
            base = dt_str[:-1]
            if "." in base:
                main, ms = base.split(".", 1)
                dt_str = f"{main}.{ms[:6]}+00:00"
            else:
                dt_str = f"{base}+00:00"
        dt = datetime.fromisoformat(dt_str)
        now = datetime.now(timezone.utc)
        diff = now - dt
        secs = int(diff.total_seconds())
        if secs < 60:
            return "just now"
        elif secs < 3600:
            return f"{secs // 60}m ago"
        elif secs < 86400:
            return f"{secs // 3600}h ago"
        elif secs < 86400 * 2:
            return "yesterday"
        elif secs < 86400 * 7:
            return f"{secs // 86400}d ago"
        else:
            return dt.strftime("%b %d")
    except Exception:
        return dt_str[:10] if dt_str else "unknown"

def clean_req(text):
    """Extract clean user request text removing prompt wrappers."""
    m = re.search(r"<USER_REQUEST>\s*(.*?)\s*</USER_REQUEST>", text, re.DOTALL)
    if m:
        t = m.group(1).strip()
    else:
        lines = [l.strip() for l in text.split("\n") if not l.startswith("<") and l.strip()]
        t = " ".join(lines)
    t = re.sub(
        r"^(could you please|can you please|please|could you|can you|i would like that|i would like to|i would like|i want to|i need to)\s+",
        "",
        t,
        flags=re.I,
    )
    return t.strip()

def extract_latest_work(tools, last_prompt, preview):
    """Derive a concise, accurate summary of the latest things worked on."""
    if tools:
        specific = [t for t in tools if not t.lower().startswith(("checking", "view", "read", "verify status"))]
        chosen = specific if specific else tools
        if len(chosen) == 1:
            return chosen[-1]
        else:
            return f"{chosen[-2]} & {chosen[-1]}"
    if last_prompt:
        lp = last_prompt.replace("\n", " ").strip()
        return (lp[:70] + "...") if len(lp) > 70 else lp
    if preview:
        pr = preview.replace("\n", " ").strip()
        return (pr[:70] + "...") if len(pr) > 70 else pr
    return "New session initialized"

def load_conversations(enhance_titles=True):
    """Load all conversations from SQLite and transcripts."""
    if not os.path.isfile(DB_PATH):
        return []

    try:
        conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
        cur = conn.cursor()
        cur.execute(
            """
            SELECT conversation_id, title, preview, last_modified_time, step_count, workspace_uris, project_id
            FROM conversation_summaries
            ORDER BY last_modified_time DESC
        """
        )
        rows = cur.fetchall()
        conn.close()
    except Exception:
        return []

    items = []
    for cid, db_title, preview, mtime, steps, ws_json, project_id in rows:
        tpath = os.path.join(BRAIN_DIR, cid, ".system_generated", "logs", "transcript.jsonl")
        user_prompts = []
        tools = []
        if os.path.isfile(tpath):
            with open(tpath, "r", errors="ignore") as f:
                for line in f:
                    if "USER_INPUT" in line:
                        try:
                            d = json.loads(line)
                            p = clean_req(d.get("content", ""))
                            if p:
                                user_prompts.append(p)
                        except Exception:
                            pass
                    if "toolSummary" in line:
                        try:
                            d = json.loads(line)
                            for c in d.get("tool_calls", []):
                                ts = c.get("args", {}).get("toolSummary")
                                if ts and ts not in tools:
                                    tools.append(ts.strip("\"'"))
                        except Exception:
                            pass

        meaningful = [
            p for p in user_prompts
            if len(p.split()) > 2 and not p.lower().startswith(("continue", "test", "proceed", "yes", "no", "exit"))
        ]
        last_prompt = meaningful[-1] if meaningful else (user_prompts[-1] if user_prompts else "")
        first_prompt = meaningful[0] if meaningful else (user_prompts[0] if user_prompts else "")

        title = db_title.strip()
        if not title:
            if first_prompt:
                title = first_prompt[:50] + ("..." if len(first_prompt) > 50 else "")
            else:
                title = "Untitled Conversation"

        latest_work = extract_latest_work(tools, last_prompt, preview)
        rel_time = get_relative_time(mtime)

        # Parse workspace
        workspace = "~"
        if ws_json:
            try:
                ws_list = json.loads(ws_json)
                if ws_list and isinstance(ws_list, list):
                    w = ws_list[0].replace("file://", "")
                    home = os.path.expanduser("~")
                    if w == home:
                        workspace = "~"
                    elif w.startswith(home):
                        workspace = "~" + w[len(home):]
                    else:
                        workspace = w
            except Exception:
                pass

        is_study = (
            (project_id or "") == "study-vault" or
            "studyvault" in workspace.lower() or
            "studyvault" in str(ws_json or "").lower() or
            "documents/obsidian/studyvault" in workspace.lower() or
            any(k in title.lower() for k in ["microeconomics", "business administration", "accounting"]) and any(k in title.lower() for k in ["worksheet", "feedback", "day 0", "day 1", "evaluation"])
        )

        items.append({
            "cid": cid,
            "title": title,
            "latest_work": latest_work,
            "last_prompt": last_prompt,
            "first_prompt": first_prompt,
            "steps": steps if steps else len(user_prompts),
            "time_ago": rel_time,
            "mtime": mtime,
            "workspace": workspace,
            "tools": tools,
            "project_id": project_id or "",
            "is_study": is_study,
        })
    return items

def update_conversation_title(cid, new_title):
    """Persistently update a conversation's title in SQLite."""
    if not os.path.isfile(DB_PATH) or not new_title.strip():
        return False
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("UPDATE conversation_summaries SET title = ? WHERE conversation_id = ?", (new_title.strip(), cid))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"Error updating title: {e}", file=sys.stderr)
        return False

def list_conversations_cli(scope=None):
    """Print a clean CLI table of conversations filtered by scope ('projects', 'study', or 'all')."""
    items = load_conversations()
    if not items:
        print("No AGY conversations found.")
        return

    cwd = os.getcwd()
    in_study_folder = "studyvault" in cwd.lower() or "documents/obsidian/studyvault" in cwd.lower()
    if scope is None:
        scope = "study" if in_study_folder else "projects"

    if scope == "study":
        items = [it for it in items if it.get("is_study")]
        scope_title = "StudyVault Sessions"
    elif scope == "projects":
        items = [it for it in items if not it.get("is_study")]
        scope_title = "Project Conversations"
    else:
        scope_title = "All Conversations"

    if not items:
        print(f"No {scope_title.lower()} found.")
        return

    print(f"\n\033[1;36m╭── Antigravity (AGY) — {scope_title} ──────────────────────────────────────────╮\033[0m")
    print(f"\033[1;30m{'#':>3}  {'Title':<38} {'Latest Action':<32} {'Active':<10}\033[0m")
    print("\033[1;30m" + "─" * 87 + "\033[0m")
    for i, it in enumerate(items, 1):
        num = f"{i}."
        title = it["title"][:36] + ".." if len(it["title"]) > 38 else it["title"]
        latest = it["latest_work"][:30] + ".." if len(it["latest_work"]) > 32 else it["latest_work"]
        active = it["time_ago"]
        print(f"\033[1;32m{num:>3}\033[0m  \033[1;37m{title:<38}\033[0m \033[36m{latest:<32}\033[0m \033[90m{active:<10}\033[0m")
        print(f"     \033[90mID: {it['cid']} • {it['steps']} steps • {it['workspace']}\033[0m")
    print("\033[1;36m╰─────────────────────────────────────────────────────────────────────────────╯\033[0m\n")

def auto_improve_titles():
    """Automatically improve and refine titles for conversations."""
    items = load_conversations()
    updated = 0
    title_improvements = {
        "0dbd2841-ab07-40e7-969b-1e7e458b02ba": "AGY Environment & MCP Tool Setup",
        "ec28da36-23c3-4808-aee3-b09b9e78b8f6": "Wayland Weather & Zen Browser Setup",
        "57c52f40-58fe-4ebb-8a31-83440838eafd": "Minimalist Transparent Waybar & Gamma Setup",
        "9fa6fd24-745c-46cb-b83d-9db4193d25ed": "Discord Setup & Waybar / Wallpaper Engine GUI",
        "dd4ffd09-0d5b-4b75-8f97-306e441cc9a2": "Custom Vertical Screen & Wallpaper Engine Setup",
        "26c89afe-185c-48c5-9667-e74b84f55403": "Technical Specification Planning",
        "01688fb5-41a3-47f9-938b-b2e51ba62b8b": "Session Exit Test",
        "81a393c9-f679-42e9-a56d-b8d232b6fbdf": "Conversation Titling & Interactive Resuming",
    }
    for it in items:
        cid = it["cid"]
        target_title = None
        if cid in title_improvements:
            target_title = title_improvements[cid]
        elif not it["title"] or it["title"] in ["Untitled Conversation", "Ending The Conversation", "Technical Specification Request"]:
            # Derive title from first meaningful prompt
            fp = it["first_prompt"]
            if fp:
                clean_fp = re.split(r"[\.\?\!\n]", fp)[0].strip()
                words = clean_fp.split()
                target_title = " ".join(words[:6])
                if target_title:
                    target_title = target_title[0].upper() + target_title[1:]

        if target_title and it["title"] != target_title:
            update_conversation_title(cid, target_title)
            updated += 1
    return updated

def interactive_selector_curses(stdscr, default_scope=None):
    """Full-featured curses interactive selector with project & StudyVault isolation."""
    curses.curs_set(0)
    curses.start_color()
    curses.use_default_colors()

    # Color pairs
    curses.init_pair(1, curses.COLOR_CYAN, -1)     # Header / Borders
    curses.init_pair(2, curses.COLOR_WHITE, -1)    # Normal text
    curses.init_pair(3, curses.COLOR_BLACK, curses.COLOR_CYAN) # Highlighted bar
    curses.init_pair(4, curses.COLOR_GREEN, -1)    # New session / tags
    curses.init_pair(5, curses.COLOR_MAGENTA, -1)  # Subtitles
    curses.init_pair(6, curses.COLOR_YELLOW, -1)   # Shortcuts / prompts
    curses.init_pair(7, -1, -1)                    # Default

    conversations_all = load_conversations()
    cwd = os.getcwd()
    in_study_folder = "studyvault" in cwd.lower() or "documents/obsidian/studyvault" in cwd.lower()
    if default_scope is not None:
        view_study = (default_scope == "study")
    else:
        view_study = in_study_folder

    selected_idx = 0
    search_query = ""
    top_idx = 0

    while True:
        stdscr.clear()
        max_y, max_x = stdscr.getmaxyx()

        if max_y < 12 or max_x < 50:
            stdscr.addstr(0, 0, "Terminal window too small.", curses.color_pair(6))
            stdscr.refresh()
            ch = stdscr.getch()
            if ch in [ord('q'), 27]:
                return None
            continue

        # Filter items by current folder scope
        scoped = [it for it in conversations_all if (it.get("is_study") if view_study else not it.get("is_study"))]

        # Filter items by search query
        filtered = []
        if search_query.strip():
            q = search_query.strip().lower()
            for it in scoped:
                match_text = f"{it['title']} {it['latest_work']} {it['last_prompt']} {it['cid']} {it['workspace']}".lower()
                if q in match_text:
                    filtered.append(it)
        else:
            filtered = scoped

        total_items = 1 + len(filtered)
        if selected_idx >= total_items:
            selected_idx = max(0, total_items - 1)

        # Draw Header
        scope_tag = "[StudyVault]" if view_study else "[Projects]"
        title_banner = f" 󱚣  Antigravity (AGY) Sessions — {scope_tag} "
        pad_len = max(0, (max_x - len(title_banner) - 4) // 2)
        header_bar = "╭" + ("─" * pad_len) + title_banner + ("─" * (max_x - 4 - len(title_banner) - pad_len)) + "╮"
        stdscr.attron(curses.color_pair(1) | curses.A_BOLD)
        stdscr.addstr(0, 0, header_bar[:max_x - 1])
        stdscr.attroff(curses.color_pair(1) | curses.A_BOLD)

        # Subtitle / Search Bar
        sub_text = f" Scope: {scope_tag} | Select a conversation to resume, or start a new session"
        stdscr.addstr(1, 0, ("│" + sub_text).ljust(max_x - 1) + "│", curses.color_pair(1))
        
        search_display = f" Filter: [{search_query}]" if search_query else " Filter: [Type to search...]"
        search_color = curses.color_pair(6) if search_query else curses.color_pair(2)
        stdscr.addstr(2, 0, ("│" + search_display).ljust(max_x - 1) + "│", curses.color_pair(1))
        stdscr.addstr(2, 2, search_display, search_color)

        sep = "├" + ("─" * (max_x - 2)) + "┤"
        stdscr.addstr(3, 0, sep[:max_x - 1], curses.color_pair(1))

        # Viewport calculation
        list_height = max(2, (max_y - 10) // 2)
        if selected_idx < top_idx:
            top_idx = selected_idx
        elif selected_idx >= top_idx + list_height:
            top_idx = selected_idx - list_height + 1

        curr_y = 4
        for i in range(top_idx, min(total_items, top_idx + list_height)):
            is_selected = (i == selected_idx)
            
            if i == 0:
                line1 = "  ✨  [+ Start New Conversation]"
                line2 = "      Start a fresh AGY session with a blank context"
                
                if is_selected:
                    stdscr.attron(curses.color_pair(3) | curses.A_BOLD)
                    stdscr.addstr(curr_y, 1, (" >" + line1[2:]).ljust(max_x - 3))
                    stdscr.attroff(curses.color_pair(3) | curses.A_BOLD)
                    stdscr.addstr(curr_y + 1, 1, line2[:max_x - 3].ljust(max_x - 3), curses.color_pair(4))
                else:
                    stdscr.addstr(curr_y, 1, line1[:max_x - 3], curses.color_pair(4) | curses.A_BOLD)
                    stdscr.addstr(curr_y + 1, 1, line2[:max_x - 3], curses.color_pair(2))
            else:
                conv = filtered[i - 1]
                time_str = f"({conv['time_ago']})"
                title_max_w = max(10, max_x - len(time_str) - 10)
                disp_title = conv['title']
                if len(disp_title) > title_max_w:
                    disp_title = disp_title[:title_max_w - 2] + ".."
                
                line1_left = f"  ●  {disp_title}"
                line1 = line1_left.ljust(max_x - len(time_str) - 4) + time_str
                
                latest_snippet = conv['latest_work']
                sub_max_w = max(10, max_x - 22)
                if len(latest_snippet) > sub_max_w:
                    latest_snippet = latest_snippet[:sub_max_w - 2] + ".."
                line2 = f"     └─ Latest: {latest_snippet}  •  {conv['steps']} steps"

                if is_selected:
                    stdscr.attron(curses.color_pair(3) | curses.A_BOLD)
                    stdscr.addstr(curr_y, 1, (" >" + line1_left[2:]).ljust(max_x - len(time_str) - 3) + time_str + " ")
                    stdscr.attroff(curses.color_pair(3) | curses.A_BOLD)
                    stdscr.addstr(curr_y + 1, 1, line2[:max_x - 3].ljust(max_x - 3), curses.color_pair(5))
                else:
                    stdscr.addstr(curr_y, 1, line1[:max_x - 3], curses.color_pair(2) | curses.A_BOLD)
                    stdscr.addstr(curr_y + 1, 1, line2[:max_x - 3], curses.color_pair(5))

            # Vertical borders
            stdscr.addstr(curr_y, 0, "│", curses.color_pair(1))
            stdscr.addstr(curr_y, max_x - 1, "│", curses.color_pair(1))
            stdscr.addstr(curr_y + 1, 0, "│", curses.color_pair(1))
            stdscr.addstr(curr_y + 1, max_x - 1, "│", curses.color_pair(1))

            curr_y += 2

        # Fill remaining space
        while curr_y < max_y - 5:
            stdscr.addstr(curr_y, 0, "│" + (" " * (max_x - 2)) + "│", curses.color_pair(1))
            curr_y += 1

        # Details Box (bottom)
        stdscr.addstr(max_y - 5, 0, "├" + ("─" * (max_x - 2)) + "┤", curses.color_pair(1))
        
        if selected_idx == 0:
            det1 = " Mode: Fresh Session | Workspace: Current Directory"
            det2 = " Tip: Press [Enter] to start, or [c] to continue the most recent session"
        elif filtered and selected_idx - 1 < len(filtered):
            cur_conv = filtered[selected_idx - 1]
            det1 = f" ID: {cur_conv['cid']}  •  Path: {cur_conv['workspace']}  •  Steps: {cur_conv['steps']}"
            last_p = cur_conv['last_prompt'].replace("\n", " ").strip()
            det2 = f" Prompt: {last_p}" if last_p else f" Latest: {cur_conv['latest_work']}"
        else:
            det1 = ""
            det2 = ""

        stdscr.addstr(max_y - 4, 0, ("│" + det1[:max_x - 3]).ljust(max_x - 1) + "│", curses.color_pair(1))
        stdscr.addstr(max_y - 4, 2, det1[:max_x - 4], curses.color_pair(2))

        stdscr.addstr(max_y - 3, 0, ("│" + det2[:max_x - 3]).ljust(max_x - 1) + "│", curses.color_pair(1))
        stdscr.addstr(max_y - 3, 2, det2[:max_x - 4], curses.color_pair(6))

        # Bottom Bar: Keybindings
        stdscr.addstr(max_y - 2, 0, "├" + ("─" * (max_x - 2)) + "┤", curses.color_pair(1))
        toggle_hint = "[Tab] Projects" if view_study else "[Tab] StudyVault"
        kb_text = f" [Enter] Resume  {toggle_hint}  [n] New  [r] Rename  [c] Continue  [q/Esc] Quit "
        stdscr.addstr(max_y - 1, 0, ("╰" + kb_text.center(max_x - 2, "─") + "╯")[:max_x - 1], curses.color_pair(1))
        stdscr.refresh()

        # Read key
        try:
            ch = stdscr.getch()
        except KeyboardInterrupt:
            return None

        if ch in [ord('q'), 27]: # q or Esc
            if search_query:
                search_query = ""
                selected_idx = 0
            else:
                return None
        elif ch in [9, curses.KEY_BTAB] and not search_query: # Tab key
            view_study = not view_study
            selected_idx = 0
            top_idx = 0
        elif ch in [curses.KEY_UP, ord('k')]:
            selected_idx = max(0, selected_idx - 1)
        elif ch in [curses.KEY_DOWN, ord('j')]:
            selected_idx = min(total_items - 1, selected_idx + 1)
        elif ch in [curses.KEY_PPAGE]: # Page Up
            selected_idx = max(0, selected_idx - list_height)
        elif ch in [curses.KEY_NPAGE]: # Page Down
            selected_idx = min(total_items - 1, selected_idx + list_height)
        elif ch in [curses.KEY_HOME]:
            selected_idx = 0
        elif ch in [curses.KEY_END]:
            selected_idx = total_items - 1
        elif ch in [ord('\n'), curses.KEY_ENTER]:
            if selected_idx == 0:
                return ("", "New Session")
            elif filtered and selected_idx - 1 < len(filtered):
                c = filtered[selected_idx - 1]
                return (c["cid"], c["title"])
        elif ch in [ord('n'), ord('N')] and not search_query:
            return ("", "New Session")
        elif ch in [ord('c'), ord('C')] and not search_query:
            if conversations:
                return (conversations[0]["cid"], conversations[0]["title"])
            return ("", "New Session")
        elif ch in [ord('r'), ord('R')] and not search_query:
            if selected_idx > 0 and filtered and selected_idx - 1 < len(filtered):
                target_conv = filtered[selected_idx - 1]
                curses.echo()
                curses.curs_set(1)
                prompt_w = min(max_x - 10, 70)
                prompt_y = max_y // 2 - 2
                prompt_x = (max_x - prompt_w) // 2

                win = curses.newwin(5, prompt_w, prompt_y, prompt_x)
                win.box()
                win.addstr(1, 2, "Rename Conversation Title:", curses.A_BOLD)
                win.addstr(3, 2, "[Enter] Save   [Esc] Cancel", curses.color_pair(6))
                win.refresh()

                input_win = curses.newwin(1, prompt_w - 6, prompt_y + 2, prompt_x + 3)
                input_win.addstr(0, 0, target_conv["title"][:prompt_w - 7])
                input_win.refresh()

                new_title_bytes = input_win.getstr(0, 0, prompt_w - 7)
                curses.noecho()
                curses.curs_set(0)

                if new_title_bytes:
                    new_title = new_title_bytes.decode("utf-8", errors="ignore").strip()
                    if new_title:
                        update_conversation_title(target_conv["cid"], new_title)
                        conversations_all = load_conversations()
        elif ch in [curses.KEY_BACKSPACE, 127, 8]:
            if search_query:
                search_query = search_query[:-1]
                selected_idx = 0
        elif 32 <= ch <= 126:
            search_query += chr(ch)
            selected_idx = 0

def run_selector(scope=None):
    """Launch the curses interactive conversation selector."""
    auto_improve_titles()
    res = curses.wrapper(lambda stdscr: interactive_selector_curses(stdscr, default_scope=scope))
    if res is None:
        return None
    cid, title = res
    if title:
        set_terminal_title(title)
    return cid

def main():
    scope_flag = "study" if "--study" in sys.argv else ("projects" if "--projects" in sys.argv else None)
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd in ["list", "ls", "--list"]:
            list_scope = "all" if "--all" in sys.argv else scope_flag
            list_conversations_cli(scope=list_scope)
            sys.exit(0)
        elif cmd in ["auto-titles", "titles", "auto"]:
            count = auto_improve_titles()
            print(f"Refined titles for {count} conversation(s).")
            sys.exit(0)
        elif cmd in ["rename", "title"] and len(sys.argv) >= 4:
            identifier = sys.argv[2]
            new_title = " ".join(sys.argv[3:])
            items = load_conversations()
            target_cid = None
            if identifier.isdigit():
                idx = int(identifier) - 1
                if 0 <= idx < len(items):
                    target_cid = items[idx]["cid"]
            else:
                for it in items:
                    if it["cid"].startswith(identifier):
                        target_cid = it["cid"]
                        break
            if target_cid:
                if update_conversation_title(target_cid, new_title):
                    print(f"Updated conversation {target_cid[:8]} title to: \"{new_title}\"")
                else:
                    print(f"Failed to update title.")
            else:
                print(f"Conversation not found: {identifier}")
            sys.exit(0)
        elif cmd in ["select", "-s", "--select"]:
            cid = run_selector(scope=scope_flag)
            if cid is None:
                sys.exit(0)
            elif cid == "":
                os.execv(REAL_BIN, ["agy"])
            else:
                os.execv(REAL_BIN, ["agy", "--conversation", cid])

    cid = run_selector(scope=scope_flag)
    if cid is None:
        sys.exit(0)
    elif cid == "":
        os.execv(REAL_BIN, ["agy"])
    else:
        os.execv(REAL_BIN, ["agy", "--conversation", cid])

if __name__ == "__main__":
    main()
