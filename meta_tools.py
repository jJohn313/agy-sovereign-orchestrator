"""
Meta-Tools & Ephemeral Tool Registry for AGY (Fix 6 & Fix 7).
Maintains baseline invariants (ponytail, enable_capability), local MCP definitions,
autonomous MCP installation integration, and system anchors.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Set, Tuple

from mcp_installer import AutonomousMcpInstaller
from schema_minifier import minify_tool_schema

logger = logging.getLogger("agy.meta_tools")

CONFIG_AGY_PATH = os.path.expanduser("~/.config/agy/mcps.json")
GEMINI_MCP_DIR = os.path.expanduser("~/.gemini/antigravity-cli/mcp")
GEMINI_CONFIG_MCP = os.path.expanduser("~/.gemini/config/mcp_config.json")

# Standard capabilities always recognized
DEFAULT_CAPABILITIES = [
    "git",
    "filesystem",
    "terminal",
    "custom_home_tools",
    "web_search",
    "python_eval",
]

# Baseline invariant tool: ponytail execution harness schema
PONYTAIL_SCHEMA = {
    "name": "ponytail",
    "description": "Core execution harness for lazy, minimal, and direct engineering solutions (YAGNI, standard library first, smallest safe diff).",
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["inspect", "enforce", "status"],
                "description": "Ponytail harness directive to verify minimal footprint or inspect code diff."
            },
            "intensity": {
                "type": "string",
                "enum": ["lite", "full", "ultra"],
                "description": "Ponytail intensity level (default: full)"
            }
        }
    }
}


class LocalToolRegistry:
    """Catalog of local MCP schemas, system anchors, and dynamic meta-tool generator."""

    def __init__(self, config_path: str = CONFIG_AGY_PATH):
        self.config_path = config_path
        self.capabilities: Dict[str, Dict[str, Any]] = {}
        self.system_anchors: Dict[str, Any] = {
            "repositories": [],
            "workspace_paths": [],
            "config_hubs": [],
            "custom_home_scripts": [],
        }
        self.installer = AutonomousMcpInstaller(config_path=config_path)
        self.load_registry()
        self.scan_system_anchors()

    def load_registry(self):
        """Scan configuration files and installed MCP definitions to populate catalog."""
        # 1. Load from ~/.config/agy/mcps.json if present
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    caps = data.get("capabilities", {})
                    for name, meta in caps.items():
                        if "tools" in meta:
                            meta["tools"] = [minify_tool_schema(t) for t in meta["tools"]]
                        self.capabilities[name] = meta
            except Exception as e:
                logger.error("Failed to read %s: %s", self.config_path, e)

        # 2. Load from ~/.gemini/antigravity-cli/mcp/ schema definitions
        if os.path.exists(GEMINI_MCP_DIR):
            try:
                for entry in os.scandir(GEMINI_MCP_DIR):
                    if entry.is_dir():
                        server_name = entry.name
                        if server_name not in self.capabilities:
                            tools = []
                            for sf in os.scandir(entry.path):
                                if sf.is_file() and sf.name.endswith(".json"):
                                    try:
                                        with open(sf.path, "r", encoding="utf-8") as f:
                                            s_data = json.load(f)
                                            if "name" in s_data:
                                                tools.append(minify_tool_schema(s_data))
                                    except Exception:
                                        pass
                            self.capabilities[server_name] = {
                                "name": server_name,
                                "description": f"Installed MCP server: {server_name}",
                                "tools": tools,
                                "standby": True,
                            }
            except Exception as e:
                logger.warning("Error scanning gemini mcp dir: %s", e)

        # Ensure all default capabilities are registered
        for cap in DEFAULT_CAPABILITIES:
            if cap not in self.capabilities:
                self.capabilities[cap] = {
                    "name": cap,
                    "description": f"Capability {cap}",
                    "tools": [self._default_fallback_tool(cap)],
                    "standby": False,
                }

    def _default_fallback_tool(self, cap_name: str) -> Dict[str, Any]:
        return {
            "name": f"{cap_name}_action",
            "description": f"Execute action for {cap_name}",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "description": "Action name"},
                    "args": {"type": "object", "description": "Action arguments"}
                }
            }
        }

    def scan_system_anchors(self):
        """
        Scan and cache known local repositories, workspace paths, configuration hubs,
        and custom executable scripts in $HOME (~/tools, ~/bin, ~/.local/bin).
        """
        home = Path.home()

        # Repositories & workspaces
        candidate_repos = [
            home / "Repositories",
            Path("/mnt/nvme_samsung/Antigravity/03_Repositories"),
            home / "Projects",
        ]
        repos: List[str] = []
        for repo_dir in candidate_repos:
            if repo_dir.exists():
                repos.append(str(repo_dir.resolve()))
                try:
                    for child in repo_dir.iterdir():
                        if child.is_dir() and not child.name.startswith("."):
                            repos.append(str(child.resolve()))
                except Exception:
                    pass
        self.system_anchors["repositories"] = sorted(list(set(repos)))
        self.system_anchors["workspace_paths"] = [str(home)] + self.system_anchors["repositories"]

        # Configuration hubs
        config_hubs = [
            str(home / ".config"),
            str(home / ".config" / "hypr"),
            str(home / ".config" / "agy"),
            str(home / "dotfiles"),
            str(home / ".bashrc"),
            str(home / ".zshrc"),
            str(home / ".profile"),
        ]
        self.system_anchors["config_hubs"] = [p for p in config_hubs if os.path.exists(p)]

        # Custom scripts in ~/tools, ~/bin, ~/.local/bin
        script_dirs = [home / "tools", home / "bin", home / ".local" / "bin"]
        custom_scripts: List[Dict[str, str]] = []
        for s_dir in script_dirs:
            if s_dir.exists() and s_dir.is_dir():
                try:
                    for item in s_dir.iterdir():
                        if item.is_file() and os.access(item, os.X_OK):
                            custom_scripts.append({
                                "name": item.name,
                                "path": str(item.resolve()),
                            })
                except Exception:
                    pass

        self.system_anchors["custom_home_scripts"] = custom_scripts

        # Update custom_home_tools tool description if available
        if "custom_home_tools" in self.capabilities:
            script_names = [s["name"] for s in custom_scripts[:25]]
            summary_desc = f"Execute custom user scripts in $HOME. Available scripts include: {', '.join(script_names)}"
            for tool in self.capabilities["custom_home_tools"].get("tools", []):
                if tool.get("name") == "execute_home_tool":
                    tool["description"] = summary_desc

    def update_registry_delta(self, mutated_paths: List[str]):
        """Update local registry delta on mutated paths."""
        if not mutated_paths:
            return

        updated = False
        for path_str in mutated_paths:
            p = Path(path_str).resolve()
            p_str = str(p)
            is_directory = p.is_dir() or (not p.exists() and not p.suffix)
            if is_directory:
                if p_str not in self.system_anchors["workspace_paths"]:
                    self.system_anchors["workspace_paths"].append(p_str)
                    updated = True
                if ".config" in p_str or "dotfiles" in p_str:
                    if p_str not in self.system_anchors["config_hubs"]:
                        self.system_anchors["config_hubs"].append(p_str)
                        updated = True
            else:
                if (not p.exists() or os.access(p, os.X_OK)) and ("/bin/" in p_str or "/tools/" in p_str):
                    if not any(s["path"] == p_str for s in self.system_anchors["custom_home_scripts"]):
                        self.system_anchors["custom_home_scripts"].append({
                            "name": p.name,
                            "path": p_str,
                        })
                        updated = True
                if any(cf in p_str for cf in [".config", ".bashrc", ".zshrc", "dotfiles"]):
                    if p_str not in self.system_anchors["config_hubs"]:
                        self.system_anchors["config_hubs"].append(p_str)
                        updated = True

        if updated:
            logger.info("Local registry updated with delta: %s", mutated_paths)

    def list_capabilities(self) -> List[str]:
        """List all capability identifiers available for mounting."""
        return sorted(list(self.capabilities.keys()))

    def get_schema(self, capability_name: str) -> List[Dict[str, Any]]:
        """Get schema for an installed capability."""
        cap = self.capabilities.get(capability_name)
        if not cap:
            return []
        return cap.get("tools", [])

    def resolve_or_install(self, capability_name: str) -> Tuple[Optional[List[Dict[str, Any]]], Dict[str, Any]]:
        """
        The enable_capability Dispatcher (Fix 7):
        1. If capability exists in local registry, hot-load its schema.
        2. If capability not in registry, invoke AutonomousMcpInstaller.
        3. If unknown/unresolvable, return (None, {"error": "Capability unresolvable"}) without crashing.
        """
        # 1. Known installed capability
        if capability_name in self.capabilities:
            schemas = self.get_schema(capability_name)
            return (schemas, {"status": "mounted", "capability": capability_name})

        # 2. Autonomous installation attempt
        logger.info("Capability '%s' not found locally. Initiating autonomous install...", capability_name)
        install_res = self.installer.install_and_register(capability_name)
        if "error" in install_res:
            return (None, install_res)

        # Reload registry and return newly mounted schema
        self.load_registry()
        new_schemas = self.get_schema(capability_name)
        return (new_schemas, {"status": "mounted", "capability": capability_name, "installed": True})

    def get_meta_tool_schema(self) -> Dict[str, Any]:
        """Generate lightweight meta-tool schema dynamically populated with capabilities."""
        available_caps = self.list_capabilities()
        enum_list = sorted(list(set(DEFAULT_CAPABILITIES + available_caps)))

        return {
            "name": "enable_capability",
            "description": (
                "Dynamically mounts an installed or verified toolset/MCP into the active turn. "
                "Call this when you need tools for git, terminal execution, filesystem mutation, or verified MCPs."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "capability": {
                        "type": "string",
                        "description": "The identifier of the capability to mount.",
                        "enum": enum_list,
                    },
                    "reason": {
                        "type": "string",
                        "description": "Brief explanation of why this capability is required for the task.",
                    },
                },
                "required": ["capability"],
            },
        }

    def get_baseline_schemas(self) -> List[Dict[str, Any]]:
        """
        Ephemeral Per-Turn Tool Scoping (Fix 6):
        Strictly returns baseline invariants: ponytail and enable_capability.
        """
        return [PONYTAIL_SCHEMA, self.get_meta_tool_schema()]
