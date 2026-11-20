"""
Repository Intelligence & AST Skeleton Client for Sovereign Orchestrator.
Provides dense structural code representations (types, structs, interfaces, methods)
for Layer 5 turn payload injection without altering invariant prefix cache hashes (L1-L4a).
"""

import ast
import json
import logging
import os
import re
import time
import urllib.request
from pathlib import Path
from typing import Dict, Any, List, Optional

logger = logging.getLogger("agy.repo_intel")

CACHE_DIR = Path(os.path.expanduser("~/.cache/agy/repo_intel"))
MOONVEIL_INTEL_URL = "http://192.168.178.67:8083/data/repo_intel"
CACHE_TTL_SEC = 3600  # 1 hour local cache


class RepoIntelClient:
    """Fetches pre-computed Moonveil AST skeletons or generates lightweight local AST outlines."""

    @staticmethod
    def get_repo_name(cwd: str) -> str:
        """Derives repository name from git or folder basename."""
        try:
            p = Path(cwd).resolve()
            # If in git repo, find top-level
            for parent in [p] + list(p.parents):
                if (parent / ".git").exists():
                    return parent.name
            return p.name
        except Exception:
            return "unknown"

    @classmethod
    def fetch_skeleton(cls, repo_name: str) -> Optional[Dict[str, Any]]:
        """Fetches pre-computed skeleton from Moonveil or local cache."""
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        local_cache_file = CACHE_DIR / repo_name / "skeleton.json"

        # Check local cache freshness
        if local_cache_file.exists():
            try:
                mtime = local_cache_file.stat().st_mtime
                if time.time() - mtime < CACHE_TTL_SEC:
                    with open(local_cache_file, "r", encoding="utf-8") as f:
                        return json.load(f)
            except Exception as e:
                logger.debug("Failed to read local skeleton cache: %s", e)

        # Attempt fetch from Moonveil over LAN
        url = f"{MOONVEIL_INTEL_URL}/{repo_name}/skeleton.json"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Agy-Orchestrator-Intel/1.0"})
            with urllib.request.urlopen(req, timeout=0.8) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                # Write through to cache
                (CACHE_DIR / repo_name).mkdir(parents=True, exist_ok=True)
                with open(local_cache_file, "w", encoding="utf-8") as f:
                    json.dump(data, f)
                logger.info("Retrieved fresh skeleton for %s from Moonveil", repo_name)
                return data
        except Exception:
            pass

        return None

    @classmethod
    def extract_file_skeleton_local(cls, file_path: Path) -> List[Dict[str, Any]]:
        """Fallback local AST signature extraction (Python, Rust, TS/JS, Go)."""
        if not file_path.exists() or file_path.is_dir():
            return []

        ext = file_path.suffix.lower()
        symbols = []

        if ext == ".py":
            try:
                content = file_path.read_text(encoding="utf-8", errors="ignore")
                tree = ast.parse(content)
                for node in tree.body:
                    if isinstance(node, ast.ClassDef):
                        methods = []
                        for item in node.body:
                            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                                args = [a.arg for a in item.args.args if a.arg != "self"]
                                prefix = "async def" if isinstance(item, ast.AsyncFunctionDef) else "def"
                                methods.append(f"{prefix} {item.name}({', '.join(args)})")
                        symbols.append({
                            "kind": "class",
                            "name": node.name,
                            "line": node.lineno,
                            "sig": f"class {node.name}",
                            "methods": methods[:6]
                        })
                    elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        args = [a.arg for a in node.args.args]
                        prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
                        symbols.append({
                            "kind": "function",
                            "name": node.name,
                            "line": node.lineno,
                            "sig": f"{prefix} {node.name}({', '.join(args)})"
                        })
            except Exception:
                pass

        elif ext == ".rs":
            try:
                content = file_path.read_text(encoding="utf-8", errors="ignore")
                current_impl = None
                for lineno, line in enumerate(content.splitlines(), start=1):
                    s = line.strip()
                    m_decl = re.match(r'^(?:pub\s+)?(struct|enum|trait|type)\s+(\w+)', s)
                    if m_decl:
                        symbols.append({
                            "kind": m_decl.group(1),
                            "name": m_decl.group(2),
                            "line": lineno,
                            "sig": s.split('{')[0].strip()
                        })
                        continue
                    m_impl = re.match(r'^impl(?:\s*<.*>)?\s+(\w+)', s)
                    if m_impl:
                        current_impl = m_impl.group(1)
                        continue
                    m_fn = re.match(r'^(?:pub\s+)?(?:async\s+)?fn\s+(\w+)\s*(<.*>)?\s*\((.*?)\)(?:\s*->\s*([^{]+))?', s)
                    if m_fn:
                        name = m_fn.group(1)
                        if current_impl:
                            name = f"{current_impl}::{name}"
                        symbols.append({
                            "kind": "function",
                            "name": name,
                            "line": lineno,
                            "sig": s.split('{')[0].strip()
                        })
            except Exception:
                pass

        elif ext in (".ts", ".tsx", ".js", ".jsx", ".go"):
            try:
                content = file_path.read_text(encoding="utf-8", errors="ignore")
                for lineno, line in enumerate(content.splitlines(), start=1):
                    s = line.strip()
                    if re.match(r'^(?:export\s+)?(?:interface|type|class)\s+\w+', s) or re.match(r'^type\s+\w+\s+(?:struct|interface)', s):
                        symbols.append({
                            "kind": "type",
                            "name": s.split()[1] if len(s.split()) > 1 else "type",
                            "line": lineno,
                            "sig": s.split('{')[0].strip()
                        })
                    elif re.match(r'^(?:export\s+)?(?:async\s+)?function\s+\w+', s) or re.match(r'^func\s+(?:\(.*\)\s+)?\w+', s):
                        symbols.append({
                            "kind": "function",
                            "name": s.split()[1] if len(s.split()) > 1 else "func",
                            "line": lineno,
                            "sig": s.split('{')[0].strip()
                        })
            except Exception:
                pass

        return symbols

    @classmethod
    def format_pre_resolved_scope(cls, target_files: List[str], cwd: str) -> str:
        """
        Builds dense AST signatures for the given target files.
        Formulated specifically for Layer 5 turn payload insertion.
        """
        if not target_files:
            return ""

        repo_name = cls.get_repo_name(cwd)
        skeleton_data = cls.fetch_skeleton(repo_name)
        god_nodes = skeleton_data.get("god_nodes", []) if skeleton_data else []
        server_files = skeleton_data.get("files", {}) if skeleton_data else {}

        lines = [
            "### Pre-Resolved Execution Scope (Authoritative)",
            f"Repository: `{repo_name}`" + (f" | Core God-Nodes: {', '.join(f'`{g}`' for g in god_nodes[:3])}" if god_nodes else ""),
            "Target files resolved with structural AST signatures:\n"
        ]

        for target in target_files:
            p = Path(cwd) / target if not Path(target).is_absolute() else Path(target)
            rel_path = target
            try:
                rel_path = str(p.relative_to(Path(cwd)))
            except Exception:
                pass

            symbols = server_files.get(rel_path) or cls.extract_file_skeleton_local(p)
            lines.append(f"• `{rel_path}`")
            if symbols:
                for sym in symbols[:8]:
                    sig = sym.get("sig", "")
                    line_no = sym.get("line", "")
                    lines.append(f"    - L{line_no}: `{sig}`")
                    methods = sym.get("methods", [])
                    for m in methods[:4]:
                        lines.append(f"        • `{m}`")
            else:
                lines.append("    (no structural signatures found)")

        return "\n".join(lines)
