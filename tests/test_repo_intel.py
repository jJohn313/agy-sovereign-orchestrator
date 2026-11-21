import unittest
import tempfile
import os
import subprocess
from pathlib import Path
from unittest.mock import patch
import repo_intel_client
from repo_intel_client import RepoIntelClient
from orchestrator import PromptPrefixStabilizer, STATIC_SYSTEM_DIRECTIVE


class TestRepoIntel(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dir_path = Path(self.temp_dir.name)
        self.cache_patcher = patch.object(repo_intel_client, "CACHE_DIR", self.dir_path / ".repo_intel_cache")
        self.cache_patcher.start()

        # Create sample Python file with type annotations
        self.py_file = self.dir_path / "sample.py"
        self.py_file.write_text("""
class UserService:
    def __init__(self, db_conn: str):
        self.db = db_conn

    def get_user(self, user_id: int) -> dict:
        return {"id": user_id}

def calculate_metric(data: list) -> float:
    return sum(data)
""")

        # Create sample Rust file with full types and fields
        self.rs_file = self.dir_path / "notch.rs"
        self.rs_file.write_text("""
pub struct NotchConfig {
    pub width: u32,
    pub height: u32,
    pub blur: bool,
}

pub enum NotchMode {
    Compact,
    Expanded(u32),
}

impl NotchConfig {
    pub fn new(width: u32, height: u32) -> Self {
        Self { width, height, blur: false }
    }

    pub fn load_from_disk(path: &Path) -> Result<Self, ConfigError> {
        Ok(Self::new(100, 50))
    }
}
""")

    def tearDown(self):
        self.cache_patcher.stop()
        self.temp_dir.cleanup()

    def test_python_skeleton_with_annotations(self):
        syms = RepoIntelClient.extract_file_skeleton_local(self.py_file)
        names = [s["name"] for s in syms]
        self.assertIn("UserService", names)
        self.assertIn("calculate_metric", names)

        user_svc = next(s for s in syms if s["name"] == "UserService")
        self.assertEqual(user_svc["kind"], "class")
        methods = user_svc.get("methods", [])
        # Verify type annotations were preserved
        self.assertTrue(any("user_id: int" in m and "-> dict" in m for m in methods))

    def test_rust_skeleton_with_fields_and_returns(self):
        syms = RepoIntelClient.extract_file_skeleton_local(self.rs_file)
        struct_sym = next(s for s in syms if s["name"] == "NotchConfig" and s["kind"] == "struct")
        # Verify struct fields are present in signature
        self.assertIn("pub width: u32", struct_sym["sig"])
        self.assertIn("pub blur: bool", struct_sym["sig"])

        # Verify function parameter types and return type are present
        fn_sym = next(s for s in syms if "load_from_disk" in s["name"])
        self.assertIn("path: &Path", fn_sym["sig"])
        self.assertIn("-> Result<Self, ConfigError>", fn_sym["sig"])

    def test_dirty_tree_detection(self):
        # Initialize a temporary git repo to test dirty tree detection
        subprocess.run(["git", "init"], cwd=str(self.dir_path), capture_output=True)
        dirty_files = RepoIntelClient.get_dirty_files(str(self.dir_path))
        # Untracked files should be recognized as dirty
        self.assertTrue("sample.py" in dirty_files or "notch.rs" in dirty_files)

        scope_text = RepoIntelClient.format_pre_resolved_scope(
            ["sample.py", "notch.rs"],
            str(self.dir_path)
        )
        self.assertIn("(locally modified)", scope_text)

    def test_prefix_cache_invariance_with_ast_signatures(self):
        """
        CRITICAL TEST: Verifies that inserting rich AST signatures into Layer 5 (turn payload)
        does NOT alter the Layer 1-4a prefix cache SHA-256 digest.
        """
        system_directive = STATIC_SYSTEM_DIRECTIVE
        active_tools = [{"name": "ponytail"}, {"name": "enable_capability"}]
        l4a_static = {"repositories": ["/test/repo"]}
        l4b_volatile = {"is_git": True, "branch": "main", "dirty_files": []}

        assembled_baseline = PromptPrefixStabilizer.assemble_prompt_layers(
            system_directive=system_directive,
            invariant_tools=active_tools,
            dynamic_active_tools=active_tools,
            l4a_static_invariants=l4a_static,
            l4b_volatile_state=l4b_volatile,
            dynamic_turn_payload="How do I modify the user service?"
        )

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

        self.assertEqual(
            assembled_baseline["prefix_cache_hash"],
            assembled_with_intel["prefix_cache_hash"],
            "Prefix cache hash MUST remain 100% invariant when rich AST skeletons are injected into Layer 5!"
        )

    def test_corrupted_cache_ignored_and_overview_formatted(self):
        cache_repo_dir = repo_intel_client.CACHE_DIR / self.dir_path.name
        cache_repo_dir.mkdir(parents=True, exist_ok=True)
        bad_file = cache_repo_dir / "skeleton.json"
        bad_file.write_text('{"response": "mock"}', encoding="utf-8")

        with patch("repo_intel_client.urllib.request.urlopen", side_effect=OSError("offline")):
            self.assertIsNone(RepoIntelClient.fetch_skeleton(self.dir_path.name))
        self.assertFalse(bad_file.exists())

        good_payload = {
            "repo_name": self.dir_path.name,
            "god_nodes": ["src/main.py", "src/lib.py"],
            "files": {"src/main.py": []},
        }
        bad_file.write_text(__import__("json").dumps(good_payload), encoding="utf-8")
        overview = RepoIntelClient.format_repo_overview(str(self.dir_path))
        self.assertIn(self.dir_path.name, overview)
        self.assertIn("`src/main.py`", overview)

    def test_retrieval_populates_repo_intel_overview_on_vector_hit(self):
        from unittest.mock import MagicMock
        from retrieval import TargetedVectorRetriever, CodebaseRetriever, Mem0Retriever

        cb = MagicMock(spec=CodebaseRetriever)
        cb.search.return_value = [
            {"similarity": 0.90, "path": "src/main.py", "name": "main", "summary": "def main(): pass"}
        ]
        m0 = MagicMock(spec=Mem0Retriever)
        m0.search.return_value = []

        retriever = TargetedVectorRetriever(codebase_retriever=cb, mem0_retriever=m0)
        with patch.object(RepoIntelClient, "format_repo_overview", return_value="Repository: `demo` | Core God-Nodes: `src/main.py`"):
            ctx = retriever.retrieve("how does main work", codebase_gate=1.0, mem0_gate=0.0)
        self.assertEqual(ctx["pre_resolved_scope"], "")
        self.assertIn("Core God-Nodes", ctx["repo_intel_overview"])


if __name__ == "__main__":
    unittest.main()
