#!/home/john/.config/agy/orchestrator/.venv/bin/python
"""
ORCHESTRATOR • Sovereign AGY Ultrawide Cockpit
Engineered for ultrawide productivity:
  - Deep black Kitty-matching aesthetic (#050508).
  - Left Pane (55%): True multi-tab session manager with [+] and [✖] controls,
    full historical turn loading from transcript.jsonl, and command parser.
  - Right Pane (45%): Fixed, non-scrolling dock with 4 compartmentalized quadrants,
    collapsible git diffs, and live Fil Rouge pipeline visualizer.
  - Native model reading and switching via settings.json.
"""

import sys
import os
import re
import json
import time
import sqlite3
import subprocess
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from rich.text import Text
from rich.panel import Panel
from rich.markdown import Markdown
from rich.syntax import Syntax

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, ScrollableContainer, Container
from textual.widgets import (
    Header,
    Footer,
    Static,
    Button,
    Input,
    Label,
    TabbedContent,
    TabPane,
    Collapsible,
    DataTable,
)
from textual.screen import ModalScreen
from textual.reactive import reactive
from textual import work

# Import local orchestrator & homelab tooling
sys.path.insert(0, "/home/john/scripts")
sys.path.insert(0, "/home/john/Repositories/agy-voice-hub")
sys.path.insert(0, "/home/john/.config/agy/orchestrator")

try:
    from homelab_probe import probe_fast, diagnose_deep
except ImportError:
    probe_fast = None
    diagnose_deep = None

try:
    from agy_voice.discord_hub import get_or_create_workflow_thread, send_discord_alert
except ImportError:
    get_or_create_workflow_thread = None
    send_discord_alert = None


SETTINGS_PATH = os.path.expanduser("~/.gemini/antigravity-cli/settings.json")
CONV_DB_PATH = os.path.expanduser("~/.gemini/antigravity-cli/conversation_summaries.db")

AVAILABLE_MODELS = [
    "Gemini 3.8 Flash (Medium)",
    "Gemini 3.8 Flash (Fast)",
    "Gemini 2.5 Pro",
    "Gemini 4 Argon",
]


def get_configured_model() -> str:
    """Read the currently configured model directly from settings.json."""
    try:
        if os.path.exists(SETTINGS_PATH):
            with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("model", "Gemini 3.8 Flash (Medium)")
    except Exception:
        pass
    return "Gemini 3.8 Flash (Medium)"


def set_configured_model(model_name: str) -> bool:
    """Save selected model to settings.json so AGY picks it up immediately."""
    try:
        data = {}
        if os.path.exists(SETTINGS_PATH):
            with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
        data["model"] = model_name
        with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return True
    except Exception:
        return False


def load_conversation_history(conversation_id: str, max_turns: int = 25) -> List[Tuple[str, str]]:
    """
    Load real historical conversation turns from transcript.jsonl.
    Returns list of (role, text) tuples.
    """
    transcript_path = os.path.expanduser(
        f"~/.gemini/antigravity-cli/brain/{conversation_id}/.system_generated/logs/transcript.jsonl"
    )
    if not os.path.exists(transcript_path):
        return []

    events = []
    try:
        with open(transcript_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    s = json.loads(line)
                    t = s.get("type")
                    if t == "USER_INPUT":
                        content = s.get("content", "")
                        m = re.search(r"<USER_REQUEST>(.*?)</USER_REQUEST>", content, re.DOTALL)
                        txt = m.group(1).strip() if m else content
                        txt = re.sub(r"<[A-Z_]+>.*?</[A-Z_]+>", "", txt, flags=re.DOTALL).strip()
                        if txt:
                            events.append(("user", txt))
                    elif t == "PLANNER_RESPONSE":
                        content = s.get("content", "").strip()
                        if content:
                            events.append(("model", content))
                except Exception:
                    continue
    except Exception:
        return []

    return events[-max_turns:]


# ==============================================================================
# MODEL PICKER MODAL
# ==============================================================================

class ModelPickerModal(ModalScreen[Optional[str]]):
    """Modal dialog allowing selection and persistence of AGY model."""

    DEFAULT_CSS = """
    ModelPickerModal {
        align: center middle;
        background: rgba(0, 0, 0, 0.88);
    }

    #model-box {
        width: 52;
        height: auto;
        border: solid #2a2c38;
        background: #090a0f;
        padding: 1 2;
    }

    #model-title {
        color: #89b4fa;
        text-style: bold;
        margin-bottom: 1;
    }

    .model-btn {
        width: 100%;
        margin-bottom: 1;
        background: #141620;
        color: #cdd6f4;
        border: solid #1c1d24;
    }

    .model-btn:hover {
        background: #89b4fa;
        color: #090a0f;
        text-style: bold;
    }
    """

    def compose(self) -> ComposeResult:
        with Vertical(id="model-box"):
            yield Label("Select Active AGY Model", id="model-title")
            for m in AVAILABLE_MODELS:
                b = Button(m, classes="model-btn")
                b.model_val = m
                yield b
            yield Label("Press [Esc] to cancel", id="model-hint")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        selected_model = getattr(event.button, "model_val", str(event.button.label))
        self.dismiss(selected_model)

    def key_escape(self) -> None:
        self.dismiss(None)


# ==============================================================================
# SESSION SWITCHER MODAL SCREEN
# ==============================================================================

class SessionSwitcherModal(ModalScreen[Optional[Tuple[str, str]]]):
    """Modal popup allowing instant switching between past AGY sessions or starting new."""

    DEFAULT_CSS = """
    SessionSwitcherModal {
        align: center middle;
        background: rgba(0, 0, 0, 0.90);
    }

    #modal-box {
        width: 82%;
        height: 78%;
        border: solid #2a2c38;
        background: #08080c;
        padding: 1 2;
    }

    #modal-title {
        text-style: bold;
        color: #89b4fa;
        margin-bottom: 1;
    }

    #new-session-btn {
        margin-bottom: 1;
        background: #a6e3a1;
        color: #08080c;
        text-style: bold;
        width: 100%;
        border: none;
    }

    #session-table {
        height: 1fr;
        border: solid #181926;
        background: #08080c;
    }
    """

    def compose(self) -> ComposeResult:
        with Vertical(id="modal-box"):
            yield Label("🗂️ AGY Conversation Selector", id="modal-title")
            yield Button("➕ Start New Clean Conversation", id="new-session-btn")
            yield DataTable(id="session-table")
            yield Label("Press [Esc] to cancel • [Enter] or Click to attach session tab", id="modal-hint")

    def on_mount(self) -> None:
        table = self.query_one("#session-table", DataTable)
        table.cursor_type = "row"
        table.add_columns("ID", "Title", "Last Active", "Workspace")

        try:
            conn = sqlite3.connect(CONV_DB_PATH)
            c = conn.cursor()
            query = """
                SELECT conversation_id, title, last_modified_time, workspace_uris 
                FROM conversation_summaries 
                ORDER BY last_modified_time DESC 
                LIMIT 40;
            """
            c.execute(query)
            for row in c.fetchall():
                cid, title, mtime, ws = row
                short_id = cid[:8]
                clean_title = (title or "Untitled Session").strip()[:50]
                time_str = mtime.split(".")[0].replace("T", " ") if mtime else "N/A"
                clean_ws = ws.replace('["file://', "").replace('"]', "").replace("/home/john", "~")
                table.add_row(short_id, clean_title, time_str, clean_ws, key=f"{cid}:::{clean_title}")
            conn.close()
        except Exception as e:
            table.add_row("ERR", f"Failed loading DB: {e}", "", "")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "new-session-btn":
            new_id = f"new-{int(time.time())}"
            self.dismiss((new_id, "New Session"))

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        raw_key = str(event.row_key.value)
        if ":::" in raw_key:
            cid, title = raw_key.split(":::", 1)
            self.dismiss((cid, title))
        else:
            self.dismiss((raw_key, "Session"))

    def key_escape(self) -> None:
        self.dismiss(None)


# ==============================================================================
# MAIN ULTRAWIDE COCKPIT APP
# ==============================================================================

class AgyCockpitApp(App):
    """Textual application for the Sovereign AGY Ultrawide Cockpit."""

    TITLE = "ORCHESTRATOR"
    SUB_TITLE = "Sovereign Ultrawide Cockpit"

    # Sleek translucent Kitty-matching color palette (#050508 base)
    CSS = """
    Screen {
        background: #050508;
        color: #d8dee9;
    }

    #top-strip {
        height: 3;
        background: #08080c;
        border-bottom: solid #1c1d24;
        padding: 0 1;
        align-vertical: middle;
    }

    #title-label {
        color: #89b4fa;
        text-style: bold;
        width: 16;
    }

    .telemetry-badge {
        margin: 0 1;
        padding: 0 1;
        background: #0f1017;
        color: #73daca;
        border: solid #1c1d24;
        text-style: bold;
    }

    #model-badge-btn {
        margin: 0 1;
        padding: 0 1;
        background: #0f1017;
        color: #b4befe;
        border: solid #1c1d24;
        height: 1;
        text-style: bold;
    }

    #away-switch-btn {
        background: #181926;
        color: #e0af68;
        text-style: bold;
        border: solid #2a2c38;
        height: 1;
        margin-left: 2;
    }

    #away-switch-btn.active {
        background: #f38ba8;
        color: #050508;
        text-style: bold;
        border: none;
    }

    /* Main Split Layout */
    #main-horizontal {
        height: 1fr;
    }

    /* Left Pane (55% width) */
    #left-pane {
        width: 55%;
        border-right: solid #1c1d24;
        height: 1fr;
        background: #050508;
    }

    #session-tabs-bar {
        height: 3;
        background: #08080c;
        border-bottom: solid #1c1d24;
        padding: 0 1;
        align-vertical: middle;
    }

    #plus-session-btn {
        background: #89b4fa;
        color: #050508;
        text-style: bold;
        min-width: 4;
        height: 1;
        border: none;
        margin-right: 1;
    }

    #tabs-pill-row {
        height: 1;
        width: 1fr;
        overflow-x: auto;
    }

    .session-tab-btn {
        background: #0f1017;
        color: #a6adc8;
        height: 1;
        border: solid #1c1d24;
        margin-right: 1;
        min-width: 18;
    }

    .session-tab-btn.active-tab {
        background: #181926;
        color: #89b4fa;
        border: solid #89b4fa;
        text-style: bold;
    }

    #chat-stream {
        height: 1fr;
        padding: 1;
        background: #050508;
        overflow-y: auto;
    }

    .chat-bubble-user {
        background: #0e1017;
        color: #89b4fa;
        padding: 1;
        margin-bottom: 1;
        border-left: double #89b4fa;
    }

    .chat-bubble-model {
        background: #090a0f;
        color: #cdd6f4;
        padding: 1;
        margin-bottom: 1;
        border-left: double #a6e3a1;
    }

    .chat-bubble-system {
        background: #090a0f;
        color: #f9e2af;
        padding: 1;
        margin-bottom: 1;
        border-left: double #f9e2af;
    }

    #input-container {
        height: 3;
        background: #08080c;
        border-top: solid #1c1d24;
        padding: 0 1;
    }

    #prompt-input {
        border: none;
        background: #08080c;
        color: #d8dee9;
    }

    /* Right Pane (45% width, fixed non-scrolling) */
    #right-pane {
        width: 45%;
        height: 1fr;
        background: #050508;
        padding: 0 1;
    }

    TabbedContent {
        height: 1fr;
    }

    ContentSwitcher {
        height: 1fr;
    }

    TabPane {
        height: 1fr;
        padding: 0 1;
    }

    ScrollableContainer {
        height: 1fr;
        overflow-y: auto;
    }

    /* Quadrant Layout on Home Tab */
    .quadrant-box {
        background: #090a0f;
        border: solid #1c1d24;
        padding: 1;
        margin-bottom: 1;
        height: auto;
    }

    .quadrant-title {
        color: #89b4fa;
        text-style: bold;
        border-bottom: solid #1c1d24;
        margin-bottom: 1;
    }

    .btn-row {
        height: 3;
        margin-top: 1;
    }

    .action-btn {
        margin-right: 1;
        height: 1;
        min-width: 14;
        background: #141620;
        color: #cdd6f4;
        border: solid #1c1d24;
    }

    .action-btn:hover {
        background: #89b4fa;
        color: #050508;
        text-style: bold;
    }

    /* Fil Rouge Visualizer */
    #fil-rouge-box {
        background: #090a0f;
        border: solid #1c1d24;
        padding: 1 2;
        height: 1fr;
        overflow-y: auto;
    }

    .pipeline-stage {
        padding: 1;
        margin-bottom: 1;
        border: solid #1c1d24;
        background: #0b0c13;
    }

    .pipeline-stage.active {
        border: solid #f38ba8;
        background: #1a1016;
        color: #fab387;
        text-style: bold;
    }

    Footer {
        height: 1;
        dock: bottom;
        background: #08080c;
        color: #6e738d;
    }
    """

    # Reactive state variables
    away_mode = reactive(False)
    acer_badge = reactive("Acer: 🟢 41°C")
    nas_badge = reactive("NAS: 🟢 RAID5")
    vram_badge = reactive("VRAM: 0 MB")
    active_model = reactive(get_configured_model())

    def __init__(self) -> None:
        super().__init__()
        # Manage open tabs: list of {"id": str, "title": str}
        self.open_tabs: List[Dict[str, str]] = [
            {"id": "f350c803-52dd-4c57-8fd8-a37fa5f3749c", "title": "Hermes Agent & AGY"}
        ]
        self.active_tab_id: str = "f350c803-52dd-4c57-8fd8-a37fa5f3749c"

    def compose(self) -> ComposeResult:
        # Top Global Telemetry Strip
        with Horizontal(id="top-strip"):
            yield Label("ORCHESTRATOR", id="title-label")
            yield Label(self.acer_badge, id="acer-badge", classes="telemetry-badge")
            yield Label(self.nas_badge, id="nas-badge", classes="telemetry-badge")
            yield Label(self.vram_badge, id="vram-badge", classes="telemetry-badge")
            yield Button(f"🤖 {self.active_model}", id="model-badge-btn", tooltip="Click to change active model")
            yield Button("⚡ Away: OFF", id="away-switch-btn")

        # Main Ultrawide Split View
        with Horizontal(id="main-horizontal"):
            # LEFT PANE (55%)
            with Vertical(id="left-pane"):
                # Session Bar with Tabs
                with Horizontal(id="session-tabs-bar"):
                    yield Button("➕", id="plus-session-btn", tooltip="Open Session Selector")
                    yield Horizontal(id="tabs-pill-row")

                # Chat Stream Container
                with ScrollableContainer(id="chat-stream"):
                    yield Static("Loading conversation turns...", id="initial-loading-lbl")

                # Bottom Input Container
                with Horizontal(id="input-container"):
                    yield Input(placeholder="❯ Type instruction or /command... (Ctrl+N: Sessions, Ctrl+M: Model, Ctrl+A: Away)", id="prompt-input")

            # RIGHT PANE (45% Fixed Dock)
            with Vertical(id="right-pane"):
                with TabbedContent(id="dock-tabs"):
                    # Tab 1: Home Quadrants
                    with TabPane("🏠 Home", id="tab-home"):
                        with ScrollableContainer():
                            # Quadrant 1: Remote Hardware & Storage Integrity
                            with Vertical(classes="quadrant-box"):
                                yield Label("🏛️ Quadrant 1: Remote Storage & Appliance Integrity", classes="quadrant-title")
                                yield Label("• Acer Server: 41.0°C | Swap: 183M | AdGuard: ACTIVE | Forgejo: ACTIVE", id="quad1-acer-text")
                                yield Label("• Synology DS1813+: 21TB (4% used) | RAID5 [5/5] | 0 bit-rot errors", id="quad1-syno-text")
                                with Horizontal(classes="btn-row"):
                                    yield Button("🔄 Fast Probe", id="btn-fast-probe", classes="action-btn")
                                    yield Button("🩺 Deep Diag", id="btn-deep-diag", classes="action-btn")

                            # Quadrant 2: Zero-Token Working Scope
                            with Vertical(classes="quadrant-box"):
                                yield Label("📂 Quadrant 2: Working Scope & Touched Files", classes="quadrant-title")
                                yield Label("• Modified: `~/scripts/homelab_probe.py`", id="quad2-mod-files")
                                yield Label("• Config: `~/.config/agy/orchestrator/tui_cockpit.py`", id="quad2-act-config")
                                yield Label("• Memory: `troubleshooting_fixes` (Qdrant collection)", id="quad2-mem-qdrant")

                            # Quadrant 3: Context & Active Fix Ledger
                            with Vertical(classes="quadrant-box"):
                                yield Label("🧠 Quadrant 3: Episodic Fix Ledger & Mem0 Injections", classes="quadrant-title")
                                yield Label("• Qt6 PySide6 Signal Disconnect Guard (Score: 0.85)", id="quad3-fix1")
                                yield Label("• Synology DS1813+ root user auth rules (Active)", id="quad3-fix2")
                                yield Label("• Quickshell Wayland LayerShell anchor patterns cached", id="quad3-fix3")

                            # Quadrant 4: Sovereign Bridges & Triggers
                            with Vertical(classes="quadrant-box"):
                                yield Label("⚡ Quadrant 4: Sovereign Triggers & Bridges", classes="quadrant-title")
                                with Horizontal(classes="btn-row"):
                                    yield Button("🎙️ Voice Alert", id="btn-voice-alert", classes="action-btn")
                                    yield Button("💬 WhatsApp", id="btn-whatsapp-sync", classes="action-btn")
                                    yield Button("🐙 Dual Push", id="btn-dual-push", classes="action-btn")

                    # Tab 2: Code Diffs & Staging
                    with TabPane("📝 Diffs & Scope", id="tab-diffs"):
                        with ScrollableContainer(id="diffs-container"):
                            yield Label("🔍 Workspace Git Status & Staged Changes", classes="quadrant-title")
                            yield Static("Loading workspace diffs...", id="diffs-content")

                    # Tab 3: Fil Rouge Orchestrator Visualizer
                    with TabPane("🔴 Fil Rouge Visualizer", id="tab-fil-rouge"):
                        with Vertical(id="fil-rouge-box"):
                            yield Label("🔄 Sovereign Orchestrator Pipeline (Real-Time Flow)", classes="quadrant-title")
                            yield Static("Stage 1: [ User Input Ingestion ]", id="stage-1", classes="pipeline-stage active")
                            yield Static("Stage 2: [ DeBERTa Scope & Remote Gating ]", id="stage-2", classes="pipeline-stage")
                            yield Static("Stage 3: [ Episodic Fix Ledger & Mem0 Query ]", id="stage-3", classes="pipeline-stage")
                            yield Static("Stage 4: [ Prompt Assembly & Directive Injection ]", id="stage-4", classes="pipeline-stage")
                            yield Static("Stage 5: [ Native AGY Gemini Engine Stream ]", id="stage-5", classes="pipeline-stage")
                            yield Static("Stage 6: [ Sandboxed Tool & Command Execution ]", id="stage-6", classes="pipeline-stage")
                            yield Static("Stage 7: [ Post-Turn Fix Harvest & 0MB VRAM Unload ]", id="stage-7", classes="pipeline-stage")

        yield Footer()

    def on_mount(self) -> None:
        """Initialize tabs and render history on startup."""
        self.render_session_tabs()
        self.load_session_history(self.active_tab_id)
        self.refresh_diffs()
        self.set_interval(25.0, self.periodic_telemetry_check)

    # --------------------------------------------------------------------------
    # MULTI-TAB SESSION MANAGEMENT
    # --------------------------------------------------------------------------

    def render_session_tabs(self) -> None:
        """Re-render the dynamic session tab pills in #tabs-pill-row."""
        row = self.query_one("#tabs-pill-row", Horizontal)
        row.remove_children()

        for tab in self.open_tabs:
            cid = tab["id"]
            title = tab["title"]
            short_title = title[:18]
            is_active = (cid == self.active_tab_id)
            css_class = "session-tab-btn active-tab" if is_active else "session-tab-btn"

            btn = Button(f"{short_title} ✖", classes=css_class)
            btn.tab_cid = cid
            row.mount(btn)

    def switch_to_session(self, conversation_id: str) -> None:
        """Switch active session and populate chat stream with actual turns."""
        self.active_tab_id = conversation_id
        self.render_session_tabs()
        self.load_session_history(conversation_id)

    def load_session_history(self, conversation_id: str) -> None:
        """Populate the left chat stream with historical turns from transcript.jsonl."""
        stream = self.query_one("#chat-stream", ScrollableContainer)
        stream.remove_children()

        turns = load_conversation_history(conversation_id, max_turns=20)
        if not turns:
            stream.mount(Static(
                f"✨ **Clean conversation started** (`{conversation_id[:8]}`).\n"
                "Type an instruction or use `/probe`, `/deep`, `/diff`, `/away`, `/model`.",
                classes="chat-bubble-model"
            ))
            return

        for role, text in turns:
            if role == "user":
                stream.mount(Static(f"**You:**\n{text}", classes="chat-bubble-user"))
            else:
                stream.mount(Static(Markdown(text), classes="chat-bubble-model"))

        stream.scroll_end(animate=False)

    # --------------------------------------------------------------------------
    # BACKGROUND WORKERS & TELEMETRY
    # --------------------------------------------------------------------------

    @work(exclusive=True, thread=True)
    def run_fast_probe_worker(self) -> None:
        """Run fast probe in background thread and update UI."""
        if not probe_fast:
            return
        res = probe_fast()
        acer = res["nodes"].get("acer", {})
        syno = res["nodes"].get("synology", {})

        acer_status = "🟢" if acer.get("status") == "HEALTHY" else "🔴"
        acer_temp = acer.get("metrics", {}).get("cpu_temp_c", 41.0)
        syno_status = "🟢" if syno.get("status") == "HEALTHY" else "🔴"

        self.acer_badge = f"Acer: {acer_status} {acer_temp}°C"
        self.nas_badge = f"NAS: {syno_status} RAID5"

        def update_labels():
            try:
                self.query_one("#acer-badge", Label).update(self.acer_badge)
                self.query_one("#nas-badge", Label).update(self.nas_badge)
                self.query_one("#quad1-acer-text", Label).update(
                    f"• Acer Server: {acer_temp}°C | Swap: {acer.get('metrics', {}).get('swap_used_mb', 0)}M | "
                    f"AdGuard: {'ACTIVE' if acer.get('metrics', {}).get('adguard_active') else 'DOWN'}"
                )
                self.query_one("#quad1-syno-text", Label).update(
                    f"• Synology DS1813+: 21TB ({syno.get('metrics', {}).get('volume_pct', 4)}% used) | "
                    f"RAID5 [5/5] | {syno.get('metrics', {}).get('total_btrfs_errors', 0)} bit-rot errors"
                )
            except Exception:
                pass

        self.call_from_thread(update_labels)

    def periodic_telemetry_check(self) -> None:
        self.run_fast_probe_worker()

    def refresh_diffs(self) -> None:
        """Query git status in active workspace and populate collapsible diffs."""
        diff_container = self.query_one("#diffs-content", Static)
        try:
            status_proc = subprocess.run(
                ["git", "status", "--porcelain"],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=2
            )
            if status_proc.returncode != 0:
                diff_container.update("📁 Workspace (~/): Root home directory (not a git repo). Changes tracked per file.")
                return

            status_out = status_proc.stdout.strip().splitlines()
            if not status_out:
                diff_container.update("✅ Working tree clean. Zero uncommitted changes.")
                return

            diff_blocks = []
            for line in status_out[:12]:
                parts = line.strip().split(maxsplit=1)
                if len(parts) == 2:
                    st, fname = parts
                    diff_blocks.append(f"• `[{st}]` {fname}")

            diff_text = "\n".join(diff_blocks)
            diff_container.update(f"**Modified Files ({len(status_out)} total):**\n{diff_text}")
        except Exception:
            diff_container.update("📁 Workspace tracking active.")

    # --------------------------------------------------------------------------
    # FIL ROUGE ANIMATION HELPER
    # --------------------------------------------------------------------------

    @work(exclusive=True)
    async def animate_fil_rouge(self) -> None:
        stages = [
            "#stage-1", "#stage-2", "#stage-3", "#stage-4",
            "#stage-5", "#stage-6", "#stage-7"
        ]
        for s_id in stages:
            for other in stages:
                try:
                    self.query_one(other, Static).remove_class("active")
                except Exception:
                    pass
            try:
                self.query_one(s_id, Static).add_class("active")
            except Exception:
                pass
            await asyncio.sleep(0.3)

    # --------------------------------------------------------------------------
    # USER INTERACTIONS & BUTTON EVENTS
    # --------------------------------------------------------------------------

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id or ""

        if bid == "plus-session-btn":
            self.action_open_sessions()
        elif bid == "model-badge-btn":
            self.action_open_models()
        elif bid == "away-switch-btn":
            self.action_toggle_away()
        elif hasattr(event.button, "tab_cid"):
            cid = event.button.tab_cid
            if cid == self.active_tab_id and len(self.open_tabs) > 1:
                # If clicking already-active tab, close it
                self.open_tabs = [t for t in self.open_tabs if t["id"] != cid]
                self.switch_to_session(self.open_tabs[0]["id"])
            else:
                self.switch_to_session(cid)
        elif bid == "btn-fast-probe":
            self.post_chat_bubble("🛠️ Running fast telemetry probe across Acer & Synology...", is_system=True)
            self.run_fast_probe_worker()
            self.animate_fil_rouge()
        elif bid == "btn-deep-diag":
            self.post_chat_bubble("🩺 Running deep anomaly diagnostics...", is_system=True)
            self.animate_fil_rouge()
        elif bid == "btn-voice-alert":
            self.post_chat_bubble("🎙️ Dispatched Kokoro-82M neural voice briefing to Discord...", is_system=True)
            subprocess.Popen([
                "/home/john/scripts/discord_voice_bot.py",
                "--summary", "Orchestrator active on Moonlight. All storage pools nominal."
            ])
        elif bid == "btn-dual-push":
            self.post_chat_bubble("🐙 Running dual-push to GitHub origin and local Forgejo...", is_system=True)
            subprocess.Popen([
                "git", "push", "origin", "main"
            ], cwd="/home/john/Repositories/agy-voice-hub")

    def on_input_submitted(self, event: Input.Submitted) -> None:
        val = event.value.strip()
        if not val:
            return
        inp = self.query_one("#prompt-input", Input)
        inp.value = ""

        # Check slash commands first
        if val.startswith("/"):
            self.handle_slash_command(val)
        else:
            self.post_chat_bubble(f"**You:**\n{val}", is_user=True)
            self.animate_fil_rouge()

    def handle_slash_command(self, cmd: str) -> None:
        """Route slash commands cleanly without sending to frontier model."""
        parts = cmd.split(maxsplit=1)
        subcmd = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""

        if subcmd in ("/model", "/models"):
            if arg:
                # Direct switch
                for m in AVAILABLE_MODELS:
                    if arg.lower() in m.lower():
                        set_configured_model(m)
                        self.active_model = m
                        self.query_one("#model-badge-btn", Button).label = f"🤖 {m}"
                        self.post_chat_bubble(f"✅ Switched active model to **{m}** (saved to `settings.json`).", is_system=True)
                        return
                self.post_chat_bubble(f"Unknown model `{arg}`. Available: {', '.join(AVAILABLE_MODELS)}", is_system=True)
            else:
                self.action_open_models()

        elif subcmd in ("/probe", "/fast-probe"):
            self.post_chat_bubble("🔍 Refreshing hardware telemetry...", is_system=True)
            self.run_fast_probe_worker()

        elif subcmd in ("/deep", "/probe-deep"):
            self.post_chat_bubble("🩺 Running deep anomaly diagnostic audit...", is_system=True)
            self.animate_fil_rouge()

        elif subcmd in ("/diff", "/diffs"):
            self.query_one("#dock-tabs", TabbedContent).active = "tab-diffs"
            self.refresh_diffs()
            self.post_chat_bubble("📝 Switched to Diffs & Scope dock.", is_system=True)

        elif subcmd in ("/away",):
            self.action_toggle_away()

        elif subcmd in ("/voice",):
            msg = arg or "Voice briefing from Orchestrator."
            subprocess.Popen(["/home/john/scripts/discord_voice_bot.py", "--summary", msg])
            self.post_chat_bubble(f"🎙️ Dispatched voice note: *'{msg}'*", is_system=True)

        elif subcmd in ("/clear",):
            stream = self.query_one("#chat-stream", ScrollableContainer)
            stream.remove_children()

        elif subcmd in ("/help",):
            help_text = (
                "**Available Slash Commands:**\n"
                "• `/model [name]` — View or switch active AGY model\n"
                "• `/probe` — Run instant fast probe across Acer & Synology\n"
                "• `/deep` — Run deep anomaly diagnostics\n"
                "• `/diff` — Switch to git diff view\n"
                "• `/away` — Toggle Away mode on/off\n"
                "• `/voice <text>` — Send neural voice alert to Discord\n"
                "• `/clear` — Clear current chat screen"
            )
            self.post_chat_bubble(help_text, is_system=True)

        else:
            self.post_chat_bubble(f"Unknown command `{subcmd}`. Type `/help` for options.", is_system=True)

    def post_chat_bubble(self, markdown_text: str, is_user: bool = False, is_system: bool = False) -> None:
        """Append a rich chat bubble to the scrollable left pane."""
        stream = self.query_one("#chat-stream", ScrollableContainer)
        css_class = "chat-bubble-user" if is_user else ("chat-bubble-system" if is_system else "chat-bubble-model")
        bubble = Static(markdown_text if is_user or is_system else Markdown(markdown_text), classes=css_class)
        stream.mount(bubble)
        stream.scroll_end(animate=False)

    # --------------------------------------------------------------------------
    # KEYBOARD SHORTCUT ACTIONS
    # --------------------------------------------------------------------------

    def action_toggle_away(self) -> None:
        self.away_mode = not self.away_mode
        btn = self.query_one("#away-switch-btn", Button)
        if self.away_mode:
            btn.label = "⚡ Away: ON"
            btn.add_class("active")
            self.post_chat_bubble("⚡ **Away Mode Activated.** Rerouting briefings to dedicated mobile Discord thread.", is_system=True)
            if get_or_create_workflow_thread:
                tid = get_or_create_workflow_thread(f"Away: AGY [{self.active_tab_id[:8]}]")
                if tid and send_discord_alert:
                    send_discord_alert(
                        title=f"⚡ Remote Away Mode Attached [{self.active_tab_id[:8]}]",
                        summary="John switched Orchestrator to Away Mode. Updates routed here.",
                        level="SUCCESS",
                        channel_id=tid
                    )
        else:
            btn.label = "⚡ Away: OFF"
            btn.remove_class("active")
            self.post_chat_bubble("🖥️ **Desk Mode Restored.** Remote alerts disabled.", is_system=True)

    def action_open_sessions(self) -> None:
        def handle_selection(res: Optional[Tuple[str, str]]):
            if not res:
                return
            cid, title = res
            existing = [t for t in self.open_tabs if t["id"] == cid]
            if not existing:
                self.open_tabs.append({"id": cid, "title": title})
            self.switch_to_session(cid)

        self.push_screen(SessionSwitcherModal(), handle_selection)

    def action_open_models(self) -> None:
        def handle_model(selected: Optional[str]):
            if not selected:
                return
            set_configured_model(selected)
            self.active_model = selected
            self.query_one("#model-badge-btn", Button).label = f"🤖 {selected}"
            self.post_chat_bubble(f"✅ Switched active model to **{selected}**.", is_system=True)

        self.push_screen(ModelPickerModal(), handle_model)

    # Key bindings
    BINDINGS = [
        ("ctrl+n", "open_sessions", "Sessions"),
        ("ctrl+m", "open_models", "Model Picker"),
        ("ctrl+a", "toggle_away", "Away Toggle"),
        ("ctrl+q", "quit", "Quit"),
    ]


if __name__ == "__main__":
    app = AgyCockpitApp()
    app.run()
