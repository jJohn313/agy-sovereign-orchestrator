import unittest
import tempfile
import os
from pathlib import Path
from repo_intel_client import RepoIntelClient
from orchestrator import PromptPrefixStabilizer, STATIC_SYSTEM_DIRECTIVE


class TestRepoIntel(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dir_path = Path(self.temp_dir.name)

        # Create sample Python file
        self.py_file = self.dir_path / "sample.py"
        self.py_file.write_text("""
class UserService:
    def __init__(self, db_conn):
        self.db = db_conn

    def get_user(self, user_id: int):
        return {"id": user_id}

def calculate_metric(data: list) -> float:
    return sum(data)
""")

        # Create sample Rust file
        self.rs_file = self.dir_path / "notch.rs"
        self.rs_file.write_text("""
pub struct NotchConfig {
    pub width: u32,
    pub height: u32,
}

pub enum NotchMode {
    Compact,
    Expanded,
}

impl NotchConfig {
    pub fn default() -> Self {
        Self { width: 300, height: 40 }
    }
}
""")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_python_skeleton_extraction(self):
        syms = RepoIntelClient.extract_file_skeleton_local(self.py_file)
        names = [s["name"] for s in syms]
        self.assertIn("UserService", names)
        self.assertIn("calculate_metric", names)

        user_svc = next(s for s in syms if s["name"] == "UserService")
        self.assertEqual(user_svc["kind"], "class")
        self.assertTrue(any("get_user" in m for m in user_svc.get("methods", [])))

    def test_rust_skeleton_extraction(self):
        syms = RepoIntelClient.extract_file_skeleton_local(self.rs_file)
        names = [s["name"] for s in syms]
        self.assertIn("NotchConfig", names)
        self.assertIn("NotchMode", names)
        self.assertTrue(any("NotchConfig::default" in s["name"] for s in syms))

    def test_format_pre_resolved_scope(self):
        scope_text = RepoIntelClient.format_pre_resolved_scope(
            ["sample.py", "notch.rs"],
            str(self.dir_path)
        )
        self.assertIn("### Pre-Resolved Execution Scope (Authoritative)", scope_text)
        self.assertIn("sample.py", scope_text)
        self.assertIn("UserService", scope_text)
        self.assertIn("notch.rs", scope_text)
        self.assertIn("NotchConfig", scope_text)

    def test_prefix_cache_invariance_with_ast_signatures(self):
        """
        CRITICAL TEST: Verifies that inserting rich AST signatures into Layer 5 (turn payload)
        does NOT alter the Layer 1-4a prefix cache SHA-256 digest.
        """
        system_directive = STATIC_SYSTEM_DIRECTIVE
        active_tools = [{"name": "ponytail"}, {"name": "enable_capability"}]
        l4a_static = {"repositories": ["/test/repo"]}
        l4b_volatile = {"is_git": True, "branch": "main", "dirty_files": []}

        # Baseline assembled layers (no repo intel)
        assembled_baseline = PromptPrefixStabilizer.assemble_prompt_layers(
            system_directive=system_directive,
            invariant_tools=active_tools,
            dynamic_active_tools=active_tools,
            l4a_static_invariants=l4a_static,
            l4b_volatile_state=l4b_volatile,
            dynamic_turn_payload="How do I modify the user service?"
        )

        # Assembled layers WITH dense AST scope injected into dynamic turn payload (Layer 5)
        rich_scope = RepoIntelClient.format_pre_resolved_scope(
            ["sample.py", "notch.rs"],
            str(self.dir_path)
        )
        assembled_with_intel = PromptPrefixStabilizer.assemble_prompt_layers(
            system_directive=system_directive,
            invariant_tools=active_tools,
            dynamic_active_tools=active_tools,
            l4a_static_invariants=l4a_static,
            l4b_volatile_state=l4b_volatile,
            dynamic_turn_payload=f"{rich_scope}\n\nHow do I modify the user service?"
        )

        # Prefix hashes MUST BE IDENTICAL
        self.assertEqual(
            assembled_baseline["prefix_cache_hash"],
            assembled_with_intel["prefix_cache_hash"],
            "Prefix cache hash MUST remain 100% invariant when rich AST skeletons are injected into Layer 5!"
        )


if __name__ == "__main__":
    unittest.main()
