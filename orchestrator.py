"""
Production Sovereign AGY System-1 Orchestrator.
Includes:
- Mode-aware TTY & streaming handling (Fix 4)
- Ephemeral per-turn tool scoping (Fix 6)
- Dynamic meta-tool escalation with max-2 limit & autonomous install (Fix 7)
- Prefix cache stabilization (75-90% prompt cache discounts)
- Adaptive output engine & milestone operational reference blocks (Section 5.B)
- Semantic error-anchored log folding integration (Fix 10)
"""

import os
import sys
import json
import logging
import hashlib
import subprocess
from typing import Dict, Any, List, Optional, Callable, Tuple, Union

from daemon import DaemonClient
from jev_client import JevClient
from retrieval import TargetedVectorRetriever
from meta_tools import LocalToolRegistry, PONYTAIL_SCHEMA
from post_turn_hook import PostTurnHook
from sanitizer import fold_log_output, strip_ansi, find_first_anchor
from exec_tool import ExecTool

logger = logging.getLogger("agy.orchestrator")

JEV_TOOL_MOUNT_THRESHOLD = 0.60
MAX_ESCALATIONS_PER_TURN = 2

# Static invariant system directive for prefix cache locking
STATIC_SYSTEM_DIRECTIVE = (
    "You are AGY, an enterprise-grade agentic assistant. "
    "Follow standard engineering invariants: YAGNI, standard library first, "
    "and minimal atomic diffs. Never hallucinate tools or schemas. "
    "When invoking any tool, emit strictly the tool invocation block without "
    "preambles, introductory narrations, or intent declarations unless user confirmation was explicitly requested. "
    "When a 'Pre-Resolved Execution Scope' is provided, treat those files as the authoritative target set. "
    "Do NOT execute exploratory filesystem searches (find, ls, grep) to discover related files unless an edit produces an unresolved missing-reference error. "
    "[Execution Density & Non-Interactive Protocol] "
    "1. Compound Scripting Over Micro-Probes: Do not execute single-line discovery or diagnostic commands sequentially across multiple turns. "
    "Consolidate pipelines (mkdir, config, permissions, service restart) into single compound blocks or self-contained scripts. "
    "For remote hosts (SSH), execute tasks via consolidated heredocs (ssh host 'bash -s' << 'EOF' ... EOF) rather than sequential single-line SSH calls. "
    "2. Proactive Diagnostics: When executing commands prone to environmental variance, embed diagnostics in the same execution "
    "(cmd || { echo '--- DIAGNOSTICS ---'; cmd --help; exit 1; }). Never consume turns solely to run --help or re-read a file just written. "
    "3. Strictly Non-Interactive & Headless: Never spawn processes that attach to TTY/interactive input loops (nano, vim, less, fzf, rofi, top, bare ssh) without explicit batch/export flags. "
    "4. Bounded Verification & Polling Loops: Never burn API roundtrips repeatedly querying status, databases, sockets, or logs across separate turns to wait for convergence. "
    "Write bounded polling loops directly within the same execution turn (total wait <= 10s), e.g.: "
    "'for i in $(seq 1 10); do if check_cmd; then break; fi; sleep 1; done; check_cmd || journalctl -u unit -n 20 --no-pager'."
)


class AgentResponse:
    """Encapsulates a turn response from the frontier model."""

    def __init__(
        self,
        content: Optional[str] = None,
        tool_calls: Optional[List[Any]] = None,
        message: Optional[Dict[str, Any]] = None,
    ):
        self.content = content or ""
        self.tool_calls = tool_calls or []
        self.message = message or {
            "role": "assistant",
            "content": self.content,
            "tool_calls": [
                {
                    "id": getattr(c, "id", f"call_{idx}"),
                    "type": "function",
                    "function": {
                        "name": getattr(c, "name", ""),
                        "arguments": json.dumps(getattr(c, "arguments", {})),
                    },
                }
                for idx, c in enumerate(self.tool_calls)
            ] if self.tool_calls else None,
        }


class ToolCall:
    """Representation of an invoked tool call."""

    def __init__(self, name: str, arguments: Dict[str, Any], call_id: str = "call_default"):
        self.name = name
        self.arguments = arguments
        self.id = call_id


class AdaptiveOutputEngine:
    """Dynamically adapts model directives based on prompt intent (Section 5.B)."""

    @staticmethod
    def get_mode_directive(prompt: str) -> str:
        p = prompt.lower()
        analytical_keywords = ["explain", "analyze", "evaluate", "compare", "architecture", "mechanics", "why", "how does"]
        code_keywords = ["fix", "implement", "create", "write", "patch", "refactor", "debug", "run", "test"]

        if any(k in p for k in analytical_keywords) and not any(k in p for k in ["fix", "implement", "patch"]):
            return (
                "\n[Adaptive Directive - Analytical Mode]\n"
                "Provide thorough, slide-by-step rigorous explanations. "
                "Focus on architectural mechanics, trade-offs, concrete technical proofs, and systems design. "
                "Do NOT truncate or compress prose."
            )
        elif any(k in p for k in code_keywords):
            return (
                "\n[Adaptive Directive - Execution Mode]\n"
                "Emit direct diffs/commands. Omit conversational filler, polite preamble, "
                "and avoid summarizing unchanged code."
            )
        return ""

    @staticmethod
    def is_milestone_turn(prompt: str, is_summary_flag: bool = False) -> bool:
        if is_summary_flag:
            return True
        p = prompt.lower()
        milestone_keywords = ["milestone", "onboard", "wrap up", "summary of project", "overview of tools", "conclude"]
        return any(k in p for k in milestone_keywords)

    @classmethod
    def generate_operational_reference_block(cls, registry: LocalToolRegistry) -> str:
        """Operational Reference Block documenting project commands and local scripts."""
        anchors = registry.system_anchors
        lines = [
            "\n### Operational Reference & Tooling Block",
            "**Core Workspace Anchors:**",
        ]
        for repo in anchors.get("repositories", [])[:5]:
            lines.append(f"  - Repository: `{repo}`")
        for hub in anchors.get("config_hubs", [])[:5]:
            lines.append(f"  - Config Hub: `{hub}`")

        lines.append("\n**Custom Executable Scripts (~/tools & ~/.local/bin):**")
        scripts = anchors.get("custom_home_scripts", [])
        if scripts:
            for s in scripts[:15]:
                lines.append(f"  - `{s['name']}` -> `{s['path']}`")
        else:
            lines.append("  - (No custom scripts detected)")

        lines.append(
            "\n**Active Project Invariants:**\n"
            "  - Harness: `ponytail` minimal diff policy active\n"
            "  - Dynamic Escalation: `enable_capability` hot-loader active\n"
        )
        return "\n".join(lines)


class PromptPrefixStabilizer:
    """Enforces deterministic prompt assembly for 75-90% cache locking."""

    @staticmethod
    def canonicalize_for_cache(data: Any) -> Any:
        """Recursively sort dict keys and primitive lists, stripping non-deterministic fields."""
        if isinstance(data, dict):
            return {
                k: PromptPrefixStabilizer.canonicalize_for_cache(v)
                for k, v in sorted(data.items(), key=lambda item: item[0])
                if k not in ("mtime", "timestamp", "last_accessed", "pid")
            }
        elif isinstance(data, (list, tuple, set)):
            # Recursively process items
            processed = [PromptPrefixStabilizer.canonicalize_for_cache(x) for x in data]
            # Sort lists of scalar types (e.g. paths, script names)
            if all(isinstance(x, (str, int, float)) for x in processed):
                return sorted(processed)
            return processed
        return data

    @classmethod
    def compute_prefix_cache_hash(
        cls,
        system_directive: str,
        invariant_tools: List[Dict[str, Any]],
        dynamic_active_tools: List[Dict[str, Any]],
        l4a_static_invariants: Union[Dict[str, Any], str],
    ) -> str:
        """
        Computes a canonical SHA-256 digest of the static prefix (L1 through L4a).
        Guarantees prefix cache hit verification across turns.
        """
        canonical_l4a = (
            cls.canonicalize_for_cache(l4a_static_invariants)
            if isinstance(l4a_static_invariants, dict)
            else l4a_static_invariants.strip()
        )

        # Sort dynamic tools deterministically
        sorted_dynamic = sorted(
            [t for t in dynamic_active_tools if t.get("name") not in ("ponytail", "enable_capability")],
            key=lambda x: x.get("name", ""),
        )
        all_tools = list(invariant_tools) + sorted_dynamic

        prefix_repr = {
            "l1_directive": system_directive.strip(),
            "l2_l3_tools": all_tools,
            "l4a_static": canonical_l4a,
        }

        serialized = json.dumps(prefix_repr, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    @staticmethod
    def get_volatile_workspace_state(cwd: Optional[str] = None) -> Dict[str, Any]:
        """
        Captures active git status and dirty workspace state for Layer 5 injection.
        Safe with timeouts and bounds.
        """
        target_dir = os.path.abspath(cwd or os.getcwd())
        state: Dict[str, Any] = {
            "is_git": False,
            "branch": None,
            "dirty_files": [],
            "raw_porcelain": "",
        }

        try:
            # Check if inside git work tree
            is_git_res = subprocess.run(
                ["git", "rev-parse", "--is-inside-work-tree"],
                cwd=target_dir,
                capture_output=True,
                text=True,
                timeout=1.5,
            )
            if is_git_res.returncode == 0 and "true" in is_git_res.stdout.strip():
                state["is_git"] = True

                # Get current branch
                branch_res = subprocess.run(
                    ["git", "branch", "--show-current"],
                    cwd=target_dir,
                    capture_output=True,
                    text=True,
                    timeout=1.5,
                )
                state["branch"] = branch_res.stdout.strip() or "HEAD (detached)"

                # Get porcelain status (capped at 50 entries to prevent context bloat)
                status_res = subprocess.run(
                    ["git", "status", "--porcelain"],
                    cwd=target_dir,
                    capture_output=True,
                    text=True,
                    timeout=2.0,
                )
                raw = status_res.stdout.strip()
                if raw:
                    lines = raw.splitlines()
                    state["dirty_files"] = [line[3:].strip() for line in lines[:50]]
                    state["raw_porcelain"] = "\n".join(lines[:50])
        except Exception as e:
            state["error"] = f"Failed to inspect workspace state: {e}"

        return state

    @classmethod
    def assemble_prompt_layers(
        cls,
        system_directive: str,
        invariant_tools: List[Dict[str, Any]],
        dynamic_active_tools: List[Dict[str, Any]],
        l4a_static_invariants: Union[Dict[str, Any], str],
        l4b_volatile_state: Optional[Union[Dict[str, Any], str]] = None,
        dynamic_turn_payload: str = "",
    ) -> Dict[str, Any]:
        """
        Deterministic ordering:
        [1. Static System Directives]
        [2. Permanent Invariant Tools] (ponytail + enable_capability)
        [3. Sorted Active Tool Schemas] (alphabetically sorted by name)
        [4. Static Core Invariants]
        [5. Dynamic Turn Payload]
        """
        # Calculate prefix hash first
        prefix_hash = cls.compute_prefix_cache_hash(
            system_directive=system_directive,
            invariant_tools=invariant_tools,
            dynamic_active_tools=dynamic_active_tools,
            l4a_static_invariants=l4a_static_invariants,
        )

        # Sort active dynamic tools deterministically by name
        sorted_dynamic = sorted(
            [t for t in dynamic_active_tools if t.get("name") not in ("ponytail", "enable_capability")],
            key=lambda x: x.get("name", ""),
        )

        all_ordered_tools = list(invariant_tools) + sorted_dynamic

        canonical_l4a = (
            cls.canonicalize_for_cache(l4a_static_invariants)
            if isinstance(l4a_static_invariants, dict)
            else l4a_static_invariants.strip()
        )

        l4a_text = json.dumps(canonical_l4a, sort_keys=True, indent=2) if isinstance(canonical_l4a, dict) else canonical_l4a

        system_content = (
            f"{system_directive}\n\n"
            f"[Static Invariants]\n{l4a_text}\n"
        )

        l4b_text = ""
        if isinstance(l4b_volatile_state, dict) and l4b_volatile_state.get("dirty_files"):
            branch = l4b_volatile_state.get("branch", "unknown")
            files_str = "\n".join(f"  - {f}" for f in l4b_volatile_state["dirty_files"])
            l4b_text = (
                f"### Active Workspace Volatile State\n"
                f"- Git Branch: `{branch}`\n"
                f"- Modified/Untracked Files ({len(l4b_volatile_state['dirty_files'])}):\n{files_str}\n"
            )
        elif isinstance(l4b_volatile_state, str) and l4b_volatile_state:
            l4b_text = f"### Active Workspace Volatile State\n{l4b_volatile_state}"

        turn_payload_parts = [p for p in (l4b_text, dynamic_turn_payload) if p]
        final_turn_payload = "\n\n".join(turn_payload_parts)

        return {
            "system_message": {"role": "system", "content": system_content},
            "ordered_tools": all_ordered_tools,
            "turn_payload": final_turn_payload,
            "prefix_cache_hash": prefix_hash,
        }


class AgyOrchestrator:
    """Core Dispatcher and Dynamic Escalation Engine."""

    def __init__(
        self,
        jev_client: Optional[JevClient] = None,
        registry: Optional[LocalToolRegistry] = None,
        retriever: Optional[TargetedVectorRetriever] = None,
        post_turn_hook: Optional[PostTurnHook] = None,
        daemon_client: Optional[DaemonClient] = None,
    ):
        self.registry = registry or LocalToolRegistry()
        self.jev_client = jev_client or JevClient()
        self.retriever = retriever or TargetedVectorRetriever()
        self.post_turn_hook = post_turn_hook or PostTurnHook(
            jev_client=self.jev_client,
            registry=self.registry,
        )
        self.daemon_client = daemon_client or DaemonClient()

    def _default_tool_executor(self, call: ToolCall) -> str:
        """Fallback tool executor with sanitized log folding."""
        if call.name == "ponytail":
            return json.dumps({"status": "ok", "mode": "enforced", "diff_policy": "minimal"})

        # Dispatch shell/terminal commands to ExecTool
        if call.name in ("run_command", "bash", "terminal", "terminal_action") or "command" in call.arguments or "cmd" in call.arguments:
            cmd_str = call.arguments.get("command") or call.arguments.get("cmd")
            if not cmd_str:
                return fold_log_output(json.dumps({"status": "err", "error": "No command provided to shell tool"}), exit_code=1)
            cwd = call.arguments.get("cwd")
            raw_result = ExecTool.execute_shell(cmd_str, cwd=cwd)
            return fold_log_output(raw_result, exit_code=0)

        raw_result = json.dumps({"status": "executed", "tool": call.name, "args": call.arguments})
        return fold_log_output(raw_result, exit_code=0)

    def fold_historical_tool_output(self, tool_name: str, raw_content: str, max_chars: int = 250) -> str:
        """
        Folds historical tool content into a compact summary stub.
        Output format: {"tool": "<name>", "status": "ok"|"err", "summary": "<summary>"}
        """
        clean_text = strip_ansi(raw_content).strip()
        status = "ok"
        summary = ""

        # 1. Try parsing structured JSON tool response
        try:
            data = json.loads(clean_text)
            if isinstance(data, dict):
                if data.get("status") in ("error", "err", "failed") or "error" in data or data.get("exit_code", 0) != 0:
                    status = "err"
                    summary = str(data.get("error") or data.get("message") or clean_text)
                else:
                    status = "ok"
                    summary = str(data.get("summary") or data.get("message") or clean_text)
        except Exception:
            pass

        # 2. Check for text error anchors (Traceback, Error:, FAILED, fatal:)
        if status != "err":
            lines = clean_text.splitlines()
            anchor_idx = find_first_anchor(lines)
            if anchor_idx is not None:
                status = "err"
                # Extract primary failure line
                summary = lines[anchor_idx].strip()
            else:
                # Clean success: take first meaningful line
                non_empty = [l.strip() for l in lines if l.strip()]
                summary = non_empty[0] if non_empty else "Success"

        # Enforce character cap
        if len(summary) > max_chars:
            summary = summary[: max_chars - 3].rstrip() + "..."

        return json.dumps({
            "tool": tool_name,
            "status": status,
            "summary": summary,
        })

    def serialize_messages_for_frontier(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Non-destructively serializes messages for the API payload.
        Compresses older tool responses (T <= -2) into summary stubs, while leaving
        the immediate previous turn (T - 1) full fidelity.
        """
        import copy
        outbound = copy.deepcopy(messages)

        # 1. Scan backwards to find the block of immediate previous tool responses (T-1)
        t_minus_1_indices = []
        in_t_minus_1_block = False

        for i in range(len(outbound) - 1, -1, -1):
            msg = outbound[i]
            role = msg.get("role")
            if role == "tool":
                in_t_minus_1_block = True
                t_minus_1_indices.append(i)
            elif in_t_minus_1_block:
                # We've moved past the contiguous block of T-1 tool responses
                break

        # 2. Compress all tool responses prior to T-1
        for i in range(len(outbound)):
            msg = outbound[i]
            if msg.get("role") == "tool" and i not in t_minus_1_indices:
                tool_name = msg.get("name", "unknown_tool")
                raw_content = msg.get("content", "")
                folded = self.fold_historical_tool_output(tool_name, raw_content)
                msg["content"] = folded

        return outbound

    def run_agent_loop(
        self,
        prompt: str,
        active_tools: List[Dict[str, Any]],
        context: Dict[str, Any],
        call_frontier_model: Optional[Callable[..., AgentResponse]] = None,
        execute_tool: Optional[Callable[[ToolCall], str]] = None,
        system_message: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Phase 3: Frontier Model Execution & Meta-Tool Loop.
        Enforces maximum 2 escalations per turn (Fix 7).
        """
        if call_frontier_model is None:
            def _dummy_frontier_model(messages, tools, context):
                return AgentResponse(content="Completed task under minimal diff policy.")
            call_frontier_model = _dummy_frontier_model

        if execute_tool is None:
            execute_tool = self._default_tool_executor

        messages = []
        if system_message:
            messages.append(system_message)
        messages.append({"role": "user", "content": prompt})
        executed_actions: List[str] = []
        mutated_paths: List[str] = []
        escalation_count = 0

        while True:
            outbound_messages = self.serialize_messages_for_frontier(messages)

            response = call_frontier_model(
                messages=outbound_messages,
                tools=active_tools,
                context=context,
            )

            if not response.tool_calls:
                return {
                    "final_content": response.content,
                    "messages": messages,
                    "active_tools": active_tools,
                    "executed_actions": executed_actions,
                    "mutated_paths": mutated_paths,
                    "escalation_count": escalation_count,
                }

            for call in response.tool_calls:
                if isinstance(call, dict):
                    c_name = call.get("name")
                    c_args = call.get("arguments", {})
                    c_id = call.get("id", "call_id")
                    call_obj = ToolCall(name=c_name, arguments=c_args, call_id=c_id)
                else:
                    call_obj = call

                if call_obj.name == "enable_capability":
                    cap_name = call_obj.arguments.get("capability")

                    # Hard escalation limit: max 2 per turn
                    if escalation_count >= MAX_ESCALATIONS_PER_TURN:
                        logger.warning("Hard escalation limit reached (%d/%d).", escalation_count, MAX_ESCALATIONS_PER_TURN)
                        messages.append(response.message)
                        messages.append({
                            "role": "tool",
                            "tool_call_id": call_obj.id,
                            "name": "enable_capability",
                            "content": json.dumps({"error": "Maximum escalation limit (2) reached for this turn."}),
                        })
                        break

                    escalation_count += 1
                    # Dispatcher with autonomous installation (Fix 7)
                    new_schemas, result_meta = self.registry.resolve_or_install(cap_name)

                    if new_schemas:
                        for s in new_schemas:
                            if s not in active_tools:
                                active_tools.append(s)
                        # Re-sort dynamic tools alphabetically to preserve prefix cache
                        invariant_names = ("ponytail", "enable_capability")
                        invariants = [t for t in active_tools if t.get("name") in invariant_names]
                        dynamic = sorted([t for t in active_tools if t.get("name") not in invariant_names], key=lambda x: x.get("name", ""))
                        active_tools = invariants + dynamic

                        action_msg = f"Mounted capability: {cap_name}" + (" (installed)" if result_meta.get("installed") else "")
                        executed_actions.append(action_msg)
                    else:
                        executed_actions.append(f"Escalation failed: {cap_name} ({result_meta.get('error')})")

                    messages.append(response.message)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": call_obj.id,
                        "name": "enable_capability",
                        "content": json.dumps(result_meta),
                    })
                    # Resume turn with updated tools
                    break
                else:
                    raw_result = execute_tool(call_obj)
                    folded_result = fold_log_output(raw_result, exit_code=0)
                    executed_actions.append(f"Executed tool {call_obj.name}")

                    if "path" in call_obj.arguments and call_obj.name in ("write_file", "edit_file"):
                        mutated_paths.append(call_obj.arguments["path"])

                    messages.append(response.message)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": call_obj.id,
                        "name": call_obj.name,
                        "content": folded_result,
                    })

    def process_turn(
        self,
        user_prompt: str,
        cwd: Optional[str] = None,
        call_frontier_model: Optional[Callable[..., AgentResponse]] = None,
        execute_tool: Optional[Callable[[ToolCall], str]] = None,
        is_summary_flag: bool = False,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Full 4-Phase Pipeline Execution Lifecycle:
        1. Jev/NLI Upstream Gating (Daemon IPC with fallback)
        2. Targeted Vector Retrieval (Strict >= 0.75)
        3. Ephemeral Per-Turn Tool Scoping & Meta-Tool Loop
        4. Jev Downstream Write-Gate & Dual-Mode Change Detection
        """
        cwd = cwd or os.getcwd()
        installed_tool_names = self.registry.list_capabilities()

        # Phase 1: Jev/NLI Upstream Gating via Daemon IPC
        pre_scores = None
        ipc_resp = self.daemon_client.send_request({
            "action": "gate_pre_turn",
            "user_prompt": user_prompt,
            "cwd": cwd,
            "available_tools": installed_tool_names,
        })
        if ipc_resp and "scores" in ipc_resp:
            pre_scores = ipc_resp["scores"]
        else:
            pre_scores = self.jev_client.gate_pre_turn(
                user_prompt=user_prompt,
                cwd=cwd,
                available_tools=installed_tool_names,
            )

        # Phase 2: Targeted Vector Retrieval
        codebase_gate = pre_scores.get("needs_codebase_search", 0.0)
        mem0_gate = pre_scores.get("needs_user_memory", 0.0)

        retrieval_context = self.retriever.retrieve(
            user_prompt=user_prompt,
            codebase_gate=codebase_gate,
            mem0_gate=mem0_gate,
        )

        # Phase 3: Ephemeral Per-Turn Tool Scoping (Fix 6)
        # Turn Reset Rule: reset active tools strictly to baseline invariants
        active_tools = self.registry.get_baseline_schemas()

        # Dynamically mount tools ONLY for this turn
        if pre_scores.get("needs_git", 0.0) > JEV_TOOL_MOUNT_THRESHOLD:
            for s in self.registry.get_schema("git"):
                if s not in active_tools:
                    active_tools.append(s)

        if pre_scores.get("needs_filesystem", 0.0) > JEV_TOOL_MOUNT_THRESHOLD:
            for s in self.registry.get_schema("filesystem"):
                if s not in active_tools:
                    active_tools.append(s)

        if pre_scores.get("needs_terminal", 0.0) > JEV_TOOL_MOUNT_THRESHOLD:
            for s in self.registry.get_schema("terminal"):
                if s not in active_tools:
                    active_tools.append(s)

        # Sort dynamic tools alphabetically for prefix cache stabilization
        invariant_names = ("ponytail", "enable_capability")
        invariants = [t for t in active_tools if t.get("name") in invariant_names]
        dynamic = sorted([t for t in active_tools if t.get("name") not in invariant_names], key=lambda x: x.get("name", ""))
        active_tools = invariants + dynamic

        # Adaptive Output Directive (Section 5.B)
        mode_directive = AdaptiveOutputEngine.get_mode_directive(user_prompt)
        turn_prompt = f"{user_prompt}\n{mode_directive}" if mode_directive else user_prompt

        pre_resolved_scope = retrieval_context.get("pre_resolved_scope", "")
        if pre_resolved_scope:
            turn_prompt = f"{pre_resolved_scope}\n\n{turn_prompt}"

        # 1. Prepare L4a (Static Invariants from Registry)
        l4a_static = {
            "repositories": self.registry.system_anchors.get("repositories", []),
            "workspace_paths": self.registry.system_anchors.get("workspace_paths", []),
            "config_hubs": self.registry.system_anchors.get("config_hubs", []),
            "custom_home_scripts": [
                {"name": s["name"], "path": s["path"]}
                for s in self.registry.system_anchors.get("custom_home_scripts", [])
            ],
            "invariants": [
                "ponytail minimal diff policy active",
                "enable_capability hot-loader active",
                "output ONLY tool invocation block on tool cycles without pre-narration",
            ],
        }

        # 2. Extract L4b (Volatile State)
        l4b_volatile = PromptPrefixStabilizer.get_volatile_workspace_state(cwd=cwd)

        # 3. Phase 3: Assemble All Prompt Layers
        assembled = PromptPrefixStabilizer.assemble_prompt_layers(
            system_directive=STATIC_SYSTEM_DIRECTIVE,
            invariant_tools=self.registry.get_baseline_schemas(),
            dynamic_active_tools=active_tools,
            l4a_static_invariants=l4a_static,
            l4b_volatile_state=l4b_volatile,
            dynamic_turn_payload=turn_prompt,
        )

        logger.info("Prefix cache locked with digest: %s", assembled["prefix_cache_hash"][:16])

        # 4. Execute Agent Loop with Assembled Layers
        loop_result = self.run_agent_loop(
            prompt=assembled["turn_payload"],
            system_message=assembled["system_message"],
            active_tools=assembled["ordered_tools"],
            context=retrieval_context,
            call_frontier_model=call_frontier_model,
            execute_tool=execute_tool,
        )

        final_content = loop_result["final_content"]

        # Milestone Operational Reference Block (Section 5.B)
        if AdaptiveOutputEngine.is_milestone_turn(user_prompt, is_summary_flag=is_summary_flag):
            milestone_block = AdaptiveOutputEngine.generate_operational_reference_block(self.registry)
            final_content = f"{final_content}\n{milestone_block}"

        # Phase 4: Downstream Write-Gate & Dual-Mode Change Detection
        summary = (
            f"Actions: {'; '.join(loop_result['executed_actions'])}. "
            f"Response: {final_content[:200]}"
        )
        post_result = self.post_turn_hook.process_turn_completion(
            user_prompt=user_prompt,
            actions_summary=summary,
            mutated_paths=loop_result.get("mutated_paths", []),
            cwd=cwd,
            prefix_cache_hash=assembled.get("prefix_cache_hash"),
            session_id=session_id,
        )

        return {
            "pre_scores": pre_scores,
            "retrieval_context": retrieval_context,
            "active_tools": [t.get("name") for t in loop_result["active_tools"]],
            "final_content": final_content,
            "messages": loop_result["messages"],
            "post_result": post_result,
            "cache_telemetry": post_result.get("cache_telemetry", {}),
            "escalation_count": loop_result.get("escalation_count", 0),
        }


def main():
    """Mode-Aware TTY & Execution Runner (Fix 4)."""
    args = sys.argv[1:]

    # Use dynamic paths instead of hardcoded home paths
    home_dir = os.path.expanduser("~")
    manager_path = os.path.join(home_dir, ".gemini", "antigravity-cli", "bin", "agy_manager.py")
    orig_bin_fallback = os.path.join(home_dir, ".local", "bin", "agy-bin")
    orig_bin = os.getenv("AGY_ORIGINAL_BIN", "")

    # If AGY_ORIGINAL_BIN is empty, try the fallback path
    if not orig_bin:
        orig_bin = orig_bin_fallback

    is_interactive = sys.stdin.isatty() and sys.stdout.isatty()
    python_bin = sys.executable or "/usr/bin/python3"

    if not args:
        if is_interactive and os.path.exists(manager_path):
            os.execv(python_bin, [python_bin, manager_path, "select"])
        elif orig_bin and os.path.exists(orig_bin):
            os.execv(orig_bin, [orig_bin])
        else:
            # Fallback when no args are provided and the original binary is missing
            print("\n[Jev System-1 Orchestrator]")
            print("Usage: agy <prompt> [options]")
            print("To see full orchestrator output, use: agy --json")
            print("Note: The original agy binary was not found. Please provide a prompt to run the orchestrator directly.")
            print("Example: agy \"What does this codebase do?\"")
            print()
            sys.exit(0)
        return

    subcmd = args[0]
    if subcmd in ("select", "-s", "--select", "list", "ls", "--list", "rename", "title", "auto-titles", "titles") and os.path.exists(manager_path):
        os.execv(python_bin, [python_bin, manager_path] + args[1:])
        return

    if subcmd in ("-n", "--new") and os.path.exists(orig_bin):
        os.execv(orig_bin, [orig_bin] + args[1:])
        return

    json_output = False
    is_summary_flag = False
    prompt_tokens = []
    i = 0
    while i < len(args):
        arg = args[i]
        if arg == "--json":
            json_output = True
        elif arg == "--summary":
            is_summary_flag = True
        elif arg in ("-p", "--prompt", "--print") and i + 1 < len(args):
            i += 1
            prompt_tokens.append(args[i])
        elif not arg.startswith("-"):
            prompt_tokens.append(arg)
        i += 1

    prompt = " ".join(prompt_tokens) if prompt_tokens else " ".join(args)

    orchestrator = AgyOrchestrator()
    result = orchestrator.process_turn(user_prompt=prompt, is_summary_flag=is_summary_flag)

    if json_output:
        print(json.dumps(result, indent=2, default=str))
    else:
        print("\n[Jev System-1 Orchestrator]")
        pre_scores = result.get("pre_scores", {})
        print(f"  • Pre-Gating Decisions: {pre_scores}")
        mounted_tools = result.get("active_tools", [])
        print(f"  • Mounted Active Tools: {mounted_tools}")
        all_caps = orchestrator.registry.list_capabilities()
        unloaded = [c for c in all_caps if c not in ["ponytail", "enable_capability"] and not any(t in mounted_tools for t in [c, f"{c}_status", f"read_{c}", f"run_{c}"])]
        print(f"  • Inactive Tools (Deferred): {unloaded} (hot-loadable via enable_capability)")

        ctx = result.get("retrieval_context", {})
        cb_m = len(ctx.get("codebase_matches", []))
        mem_m = len(ctx.get("mem0_matches", []))
        if cb_m or mem_m:
            print(f"  • Vector Context: {cb_m} codebase, {mem_m} memory matches (similarity >= 0.75)")

        chg = result.get("post_result", {}).get("change_detection", {})
        if chg and chg.get("reindexed"):
            print(f"  • Change Detection: Mode '{chg.get('mode')}' detected changes -> re-indexed {chg.get('path')}")

        print(f"\n[Response]\n{result.get('final_content')}\n")
        post = result.get("post_result", {})
        if post.get("committed"):
            print(f"[Mem0 Write-Gate] Committed durable memory: {post.get('memory_text')}")
        else:
            print(f"[Mem0 Write-Gate] Score {post.get('durable_score', 0):.2f} <= 0.85; memory write discarded.")


if __name__ == "__main__":
    main()
