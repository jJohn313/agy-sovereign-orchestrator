"""
Automated Reference Baseline Scout for Sovereign AGY Orchestrator.
Searches reputable GitHub repositories for target technologies (Qt6, Quickshell, Rust, etc.),
performs shallow `--depth 1` caching, extracts architectural idioms, and indexes
the baseline into semantic memory to eliminate hallucinations.
"""

import os
import sys
import json
import shutil
import logging
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional

logger = logging.getLogger("agy.reference_scout")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [SCOUT] %(message)s")

BASELINES_CACHE_DIR = os.path.expanduser("~/.cache/agy/baselines")


class ReferenceScout:
    """Scouts, clones, and indexes reputable GitHub baselines for tricky technologies."""

    def __init__(self, cache_dir: str = BASELINES_CACHE_DIR):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def search_reputable_repo(
        self,
        query: str,
        tech: Optional[str] = None,
        min_stars: int = 5,
        limit: int = 5,
    ) -> Optional[Dict[str, Any]]:
        """
        Query GitHub via authenticated `gh` CLI for reputable repositories.
        Filters out untrusted, low-signal, or empty repos.
        """
        # Deduplicate search tokens
        tokens = []
        for t in f"{query} {tech or ''}".split():
            if t.lower() not in [x.lower() for x in tokens]:
                tokens.append(t)
        search_terms = " ".join(tokens).strip()

        candidates = [search_terms]
        if tech and tech.strip().lower() != search_terms.lower():
            candidates.append(tech.strip())

        for term in candidates:
            cmd = [
                "gh",
                "search",
                "repos",
                term,
                "--sort",
                "stars",
                "--limit",
                str(limit),
                "--json",
                "fullName,description,stargazersCount,url",
            ]

            try:
                res = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
                if res.returncode != 0:
                    logger.warning("gh search failed: %s", res.stderr)
                    continue

                items = json.loads(res.stdout)
                for item in items:
                    stars = item.get("stargazersCount", 0)
                    if stars >= min_stars:
                        logger.info("Found reputable baseline for '%s': %s (%d stars)", term, item["fullName"], stars)
                        return item
            except Exception as e:
                logger.error("Error searching GitHub baselines for '%s': %s", term, e)

        return None

    def clone_or_update_baseline(self, repo_full_name: str) -> Optional[Path]:
        """
        Perform a shallow (--depth 1) clone into the baselines cache.
        If already present, fetches latest changes.
        """
        target_path = self.cache_dir / repo_full_name.replace("/", "_")

        if target_path.exists() and (target_path / ".git").exists():
            try:
                subprocess.run(
                    ["git", "-C", str(target_path), "pull", "--ff-only"],
                    capture_output=True,
                    timeout=15,
                )
                logger.info("Updated existing baseline at %s", target_path)
                return target_path
            except Exception as e:
                logger.warning("Could not pull baseline %s: %s", target_path, e)
                return target_path

        # Shallow clone
        repo_url = f"https://github.com/{repo_full_name}.git"
        logger.info("Shallow cloning baseline %s -> %s", repo_url, target_path)
        try:
            res = subprocess.run(
                ["git", "clone", "--depth", "1", repo_url, str(target_path)],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if res.returncode == 0:
                # Immediately index with sem-index as per workspace directives
                subprocess.run(["sem-index", str(target_path)], capture_output=True, timeout=15)
                return target_path
            else:
                logger.error("Git clone failed: %s", res.stderr)
        except Exception as e:
            logger.error("Clone exception: %s", e)

        return None

    def extract_idioms(self, repo_path: Path, tech: Optional[str] = None) -> Dict[str, Any]:
        """
        Extract key architectural idioms, file structures, and imports
        to ground the model in verified reality.
        """
        idioms: Dict[str, Any] = {
            "key_files": [],
            "imports_or_deps": [],
            "structure_summary": "",
        }

        if not repo_path.exists():
            return idioms

        tech_clean = (tech or "").lower()
        file_patterns = {
            "quickshell": ["*.qml", "*.js", "shell.qml"],
            "qt6": ["*.py", "*.cpp", "*.h", "*.qml", "CMakeLists.txt"],
            "pyside6": ["*.py", "*.qml"],
            "rust": ["Cargo.toml", "src/main.rs", "src/lib.rs"],
            "hyprland": ["*.conf", "*.cpp", "*.hpp"],
        }.get(tech_clean, ["*.py", "*.rs", "*.qml", "*.toml", "*.json"])

        found_files = []
        for pat in file_patterns:
            for f in repo_path.glob(f"**/{pat}"):
                if not any(part.startswith(".") for part in f.parts):
                    found_files.append(f)
                    if len(found_files) >= 15:
                        break
            if len(found_files) >= 15:
                break

        idioms["key_files"] = [str(f.relative_to(repo_path)) for f in found_files]

        # Extract dependency/manifest info
        manifests = ["Cargo.toml", "CMakeLists.txt", "requirements.txt", "package.json"]
        for m in manifests:
            m_path = repo_path / m
            if m_path.exists():
                try:
                    with open(m_path, "r", encoding="utf-8", errors="ignore") as f:
                        lines = [line.strip() for line in f.readlines()[:25] if line.strip() and not line.startswith("#")]
                        idioms["imports_or_deps"].extend(lines[:10])
                except Exception:
                    pass

        return idioms

    def scout_baseline(self, query: str, tech: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Complete workflow: search -> shallow clone -> extract idioms."""
        repo = self.search_reputable_repo(query, tech=tech)
        if not repo:
            return None

        repo_name = repo["fullName"]
        path = self.clone_or_update_baseline(repo_name)
        if not path:
            return None

        idioms = self.extract_idioms(path, tech=tech)
        return {
            "repo": repo_name,
            "stars": repo.get("stargazersCount", 0),
            "description": repo.get("description", ""),
            "local_path": str(path),
            "idioms": idioms,
        }

    def format_directive(self, baseline_info: Dict[str, Any]) -> str:
        """Format baseline info into a clean directive block for the prompt."""
        if not baseline_info:
            return ""

        repo = baseline_info.get("repo", "")
        path = baseline_info.get("local_path", "")
        desc = baseline_info.get("description", "")
        idioms = baseline_info.get("idioms", {})
        key_files = idioms.get("key_files", [])[:8]
        deps = idioms.get("imports_or_deps", [])[:5]

        files_str = "\n".join(f"  - {f}" for f in key_files) if key_files else "  (none)"
        deps_str = "\n".join(f"  - {d}" for d in deps) if deps else "  (standard)"

        return (
            f"[AUTHORITATIVE GROUND-TRUTH BASELINE: {repo}]\n"
            f"Cached at: {path}\n"
            f"Description: {desc}\n"
            f"Key Reference Files:\n{files_str}\n"
            f"Manifest / Core Imports:\n{deps_str}\n"
            f"Directive: Use this verified baseline as the syntax and architectural standard. "
            f"Do not invent non-existent APIs or deprecated patterns."
        )


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Scout and mount reputable GitHub baseline")
    parser.add_argument("--query", required=True, help="Search query (e.g. 'quickshell wayland bar')")
    parser.add_argument("--tech", default="", help="Specific technology/framework (e.g. quickshell, qt6, rust)")
    args = parser.parse_args()

    scout = ReferenceScout()
    result = scout.scout_baseline(args.query, tech=args.tech)
    if result:
        print(scout.format_directive(result))
    else:
        print("[!] No reputable baseline found.")


if __name__ == "__main__":
    main()
