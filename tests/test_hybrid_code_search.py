import os
import sys
import tempfile
import sqlite3
import subprocess
import unittest
from pathlib import Path

import sqlite_vec
import numpy as np

from code_indexer import (
    ensure_db_schema,
    get_db_connection,
    ASTChunker,
    IncrementalGitIndexer,
)
from retrieval import (
    CodebaseRetriever,
    compute_cosine_similarity,
    compute_jaccard_similarity,
)


class TestHybridCodeSearch(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test_workspaces_vec.db")
        self.conn = get_db_connection(self.db_path)

    def tearDown(self):
        try:
            self.conn.close()
        except Exception:
            pass
        self.temp_dir.cleanup()

    def test_virtual_table_before_delete_trigger_cleanup(self):
        """
        Mandatory Adjustment 1:
        Verify SQLite foreign keys do not leave ghost vectors in vec0.
        Explicit BEFORE DELETE trigger on code_chunks must delete matching chunk_id rows from vec_code_chunks.
        """
        cur = self.conn.cursor()
        # Insert a repo
        cur.execute("INSERT INTO code_repos (repo_path, repo_name, last_indexed_commit) VALUES (?, ?, ?)",
                    ("/test/repo", "repo", "commit_1"))
        repo_id = cur.lastrowid

        # Insert a chunk
        cur.execute("""
            INSERT INTO code_chunks (
                repo_id, file_path, language, symbol_breadcrumb,
                symbol_type, start_line, end_line, content, token_count, file_hash
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (repo_id, "main.py", "python", "main.py > start", "function", 1, 10, "def start(): pass", 4, "hash1"))
        chunk_id = cur.lastrowid

        # Insert vector embedding into vec_code_chunks
        dummy_vec = [0.1] * 384
        cur.execute(
            "INSERT INTO vec_code_chunks (chunk_id, embedding) VALUES (?, ?)",
            (chunk_id, sqlite_vec.serialize_float32(dummy_vec)),
        )
        self.conn.commit()

        # Verify initial insertion
        v_count = cur.execute("SELECT count(*) FROM vec_code_chunks WHERE chunk_id = ?", (chunk_id,)).fetchone()[0]
        self.assertEqual(v_count, 1)

        # Delete chunk from code_chunks
        cur.execute("DELETE FROM code_chunks WHERE id = ?", (chunk_id,))
        self.conn.commit()

        # Verify vec_code_chunks was purged automatically by trg_code_chunks_bd
        v_after = cur.execute("SELECT count(*) FROM vec_code_chunks WHERE chunk_id = ?", (chunk_id,)).fetchone()[0]
        self.assertEqual(v_after, 0, "Ghost vector remained in vec_code_chunks after code_chunks deletion!")

    def test_ast_chunking_with_symbol_breadcrumbs(self):
        """Verify AST parsing splits classes and methods with canonical breadcrumbs."""
        code = """
import os
import sys

# Preamble comment block
GLOBAL_CONFIG = {"debug": True}

class AuthenticationManager:
    \"\"\"Handles user credentials.\"\"\"
    def __init__(self, secret: str):
        self.secret = secret

    def verify_token(self, token: str) -> bool:
        return token == self.secret

def helper_digest(data: str) -> str:
    return data.strip().lower()
"""
        chunks = ASTChunker.chunk_file("auth/manager.py", code)
        self.assertTrue(len(chunks) >= 3)

        breadcrumbs = [c["symbol_breadcrumb"] for c in chunks]
        self.assertIn("auth/manager.py > AuthenticationManager", breadcrumbs)
        self.assertIn("auth/manager.py > AuthenticationManager > __init__", breadcrumbs)
        self.assertIn("auth/manager.py > AuthenticationManager > verify_token", breadcrumbs)
        self.assertIn("auth/manager.py > helper_digest", breadcrumbs)

        # Verify line bounds
        for c in chunks:
            self.assertGreater(c["end_line"], c["start_line"] - 1)
            self.assertIn("content", c)

    def test_hybrid_search_rrf_combines_dense_and_sparse(self):
        """
        Verify Reciprocal Rank Fusion fuses sqlite-vec dense cosine search
        and FTS5 BM25 sparse exact matches.
        """
        cur = self.conn.cursor()
        cur.execute("INSERT INTO code_repos (repo_path, repo_name) VALUES ('/repo', 'repo')")
        repo_id = cur.lastrowid

        # Insert two chunks
        # Chunk 1: Exact symbol match for "compute_jaccard_similarity"
        cur.execute("""
            INSERT INTO code_chunks (
                repo_id, file_path, language, symbol_breadcrumb,
                symbol_type, start_line, end_line, content, token_count, file_hash
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            repo_id, "retrieval.py", "python", "retrieval.py > compute_jaccard_similarity",
            "function", 25, 34, "def compute_jaccard_similarity(text_a: str, text_b: str) -> float:\n    pass",
            10, "hash1"
        ))
        cid_1 = cur.lastrowid
        cur.execute(
            "INSERT INTO vec_code_chunks (chunk_id, embedding) VALUES (?, ?)",
            (cid_1, sqlite_vec.serialize_float32([0.05] * 384)),
        )

        # Chunk 2: Other unrelated chunk
        cur.execute("""
            INSERT INTO code_chunks (
                repo_id, file_path, language, symbol_breadcrumb,
                symbol_type, start_line, end_line, content, token_count, file_hash
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            repo_id, "daemon.py", "python", "daemon.py > check_idle",
            "function", 40, 55, "def check_idle(self) -> bool:\n    return True",
            8, "hash2"
        ))
        cid_2 = cur.lastrowid
        cur.execute(
            "INSERT INTO vec_code_chunks (chunk_id, embedding) VALUES (?, ?)",
            (cid_2, sqlite_vec.serialize_float32([0.9] * 384)),
        )
        self.conn.commit()

        retriever = CodebaseRetriever(db_path=self.db_path, threshold=0.70)
        # Search for exact symbol "compute_jaccard_similarity"
        results = retriever.search("compute_jaccard_similarity", limit=2)

        self.assertTrue(len(results) >= 1)
        top_hit = results[0]
        self.assertEqual(top_hit["name"], "retrieval.py > compute_jaccard_similarity")
        self.assertGreaterEqual(top_hit["similarity"], 0.75)
        self.assertIsNotNone(top_hit.get("embedding"))

    def test_incremental_git_indexing_on_commit_divergence(self):
        """
        Mandatory Adjustment 2:
        Verify IncrementalGitIndexer tracks commit divergence (HEAD != last_indexed_commit)
        and prunes deleted files without leaving orphaned chunks or ghost vectors.
        """
        git_repo_dir = os.path.join(self.temp_dir.name, "sample_repo")
        os.makedirs(git_repo_dir, exist_ok=True)

        # Initialize git repo
        subprocess.run(["git", "init"], cwd=git_repo_dir, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.name", "TestUser"], cwd=git_repo_dir, check=True)
        subprocess.run(["git", "config", "user.email", "test@test.local"], cwd=git_repo_dir, check=True)

        # Commit 1: Add file_a.py and file_b.py
        file_a = os.path.join(git_repo_dir, "file_a.py")
        file_b = os.path.join(git_repo_dir, "file_b.py")
        with open(file_a, "w") as f:
            f.write("def func_a():\n    return 'alpha'\n")
        with open(file_b, "w") as f:
            f.write("def func_b():\n    return 'beta'\n")

        subprocess.run(["git", "add", "."], cwd=git_repo_dir, check=True)
        subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=git_repo_dir, check=True)

        indexer = IncrementalGitIndexer(db_path=self.db_path)
        # Initial index
        res1 = indexer.index_repository(git_repo_dir)
        self.assertEqual(res1["status"], "indexed_full")

        # Verify no divergence immediately after index
        div_check = indexer.check_commit_divergence(git_repo_dir)
        self.assertFalse(div_check["diverged"])

        # Commit 2: Modify file_a.py, delete file_b.py, add file_c.py
        with open(file_a, "w") as f:
            f.write("def func_a_modified():\n    return 'alpha_v2'\n")
        os.remove(file_b)
        file_c = os.path.join(git_repo_dir, "file_c.py")
        with open(file_c, "w") as f:
            f.write("def func_c():\n    return 'gamma'\n")

        subprocess.run(["git", "add", "-A"], cwd=git_repo_dir, check=True)
        subprocess.run(["git", "commit", "-m", "Commit 2: changes"], cwd=git_repo_dir, check=True)

        # Check commit divergence
        div_check2 = indexer.check_commit_divergence(git_repo_dir)
        self.assertTrue(div_check2["diverged"], "Commit divergence was not detected after git commit!")

        # Re-index incrementally
        res2 = indexer.index_repository(git_repo_dir)
        self.assertEqual(res2["status"], "indexed_incremental")

        # Verify DB state: file_b.py chunks must be purged
        cur = self.conn.cursor()
        b_count = cur.execute("SELECT count(*) FROM code_chunks WHERE file_path = 'file_b.py'").fetchone()[0]
        self.assertEqual(b_count, 0, "file_b.py was not purged after deletion!")

        # Verify file_c.py exists
        c_count = cur.execute("SELECT count(*) FROM code_chunks WHERE file_path = 'file_c.py'").fetchone()[0]
        self.assertGreaterEqual(c_count, 1)

        # Total chunks in code_chunks must match total rows in vec_code_chunks
        total_chunks = cur.execute("SELECT count(*) FROM code_chunks").fetchone()[0]
        total_vecs = cur.execute("SELECT count(*) FROM vec_code_chunks").fetchone()[0]
        self.assertEqual(total_chunks, total_vecs, "Mismatch between code_chunks and vec_code_chunks count!")


if __name__ == "__main__":
    unittest.main()
