import unittest
from unittest.mock import MagicMock
from retrieval import TargetedVectorRetriever, CodebaseRetriever, Mem0Retriever, compute_jaccard_similarity

class TestRetrievalMMR(unittest.TestCase):
    def test_jaccard_similarity(self):
        text_a = "def hello_world(): print('hello')"
        text_b = "def hello_world(): print('hello world')"
        score = compute_jaccard_similarity(text_a, text_b)
        self.assertTrue(score > 0.6)

    def test_top_k_and_deduplication(self):
        codebase_mock = MagicMock(spec=CodebaseRetriever)
        # Create 8 codebase candidates, 4 of which are near identical (high Jaccard overlap)
        codebase_mock.search.return_value = [
            {"similarity": 0.95, "path": "/path/to/file1.py", "name": "File 1", "summary": "def function_one(): return 'unique logic one'"},
            {"similarity": 0.94, "path": "/path/to/file1_copy.py", "name": "File 1 Copy", "summary": "def function_one(): return 'unique logic one'"},
            {"similarity": 0.93, "path": "/path/to/file1_copy2.py", "name": "File 1 Copy 2", "summary": "def function_one(): return 'unique logic one'"},
            {"similarity": 0.92, "path": "/path/to/file1_copy3.py", "name": "File 1 Copy 3", "summary": "def function_one(): return 'unique logic one'"},
            {"similarity": 0.85, "path": "/path/to/file2.py", "name": "File 2", "summary": "def function_two(): print('something completely different')"},
            {"similarity": 0.80, "path": "/path/to/file3.py", "name": "File 3", "summary": "class MyClass:\n    pass"},
            {"similarity": 0.78, "path": "/path/to/file4.py", "name": "File 4", "summary": "import os\nimport sys"},
            {"similarity": 0.76, "path": "/path/to/file5.py", "name": "File 5", "summary": "CONSTANT_VAR = 42"}
        ]

        mem0_mock = MagicMock(spec=Mem0Retriever)
        # 1 additional unique item from mem0
        mem0_mock.search.return_value = [
            {"score": 0.99, "id": "m_1", "memory": "Preference: use tabs over spaces"}
        ]

        retriever = TargetedVectorRetriever(
            codebase_retriever=codebase_mock,
            mem0_retriever=mem0_mock,
            threshold=0.75
        )

        result = retriever.retrieve("query", codebase_gate=1.0, mem0_gate=1.0)

        # Total accepted chunks should be 4
        self.assertEqual(len(result["codebase_matches"]) + len(result["mem0_matches"]), 4)

        lines = result["formatted_context"].split("\n")
        self.assertEqual(len(lines), 4)

        # Verify specific items
        # 1st item should be from mem0 (score 0.99)
        self.assertTrue("[mem0:m_1]" in lines[0])

        # 2nd item should be the first codebase item (score 0.95)
        self.assertTrue("[codebase:/path/to/file1.py]" in lines[1])

        # The next 3 codebase items (copies) should be excluded due to deduplication.
        # Thus 3rd item should be file2.py (score 0.85)
        self.assertTrue("[codebase:/path/to/file2.py]" in lines[2])

        # 4th item should be file3.py (score 0.80)
        self.assertTrue("[codebase:/path/to/file3.py]" in lines[3])

    def test_codebase_lines_tag(self):
        codebase_mock = MagicMock(spec=CodebaseRetriever)
        codebase_mock.search.return_value = [
            {"similarity": 0.95, "path": "/path/to/file_with_lines.py", "name": "File", "summary": "def test(): pass", "start_line": 10, "end_line": 20},
        ]

        mem0_mock = MagicMock(spec=Mem0Retriever)
        mem0_mock.search.return_value = []

        retriever = TargetedVectorRetriever(
            codebase_retriever=codebase_mock,
            mem0_retriever=mem0_mock,
            threshold=0.75
        )

        result = retriever.retrieve("query", codebase_gate=1.0, mem0_gate=1.0)

        self.assertTrue("[codebase:/path/to/file_with_lines.py:10-20]" in result["formatted_context"])

    def test_vector_mmr_vs_jaccard_divergence(self):
        """
        Verify that semantic vector MMR suppresses duplicate concepts with 0% lexical overlap,
        which lexical Jaccard completely fails to catch.
        """
        # Vector A: e.g. representation of "database connection pool"
        emb_a = [1.0, 0.0, 0.0, 0.0]
        # Vector B: semantically identical embedding (cosine similarity = 1.0)
        emb_b = [1.0, 0.0, 0.0, 0.0]
        # Vector C: distinctly different embedding (cosine similarity = 0.0)
        emb_c = [0.0, 1.0, 0.0, 0.0]

        codebase_mock = MagicMock(spec=CodebaseRetriever)
        codebase_mock.search.return_value = [
            {
                "similarity": 0.95,
                "path": "/path/to/db_alpha.py",
                "name": "DB Alpha",
                "summary": "initialize storage repository connections",
                "embedding": emb_a,
            },
            {
                "similarity": 0.94,
                "path": "/path/to/db_beta.py",
                "name": "DB Beta",
                # Completely distinct words: lexical Jaccard is 0.0!
                "summary": "boot up persistency sockets cluster",
                "embedding": emb_b,
            },
            {
                "similarity": 0.88,
                "path": "/path/to/auth.py",
                "name": "Auth Service",
                "summary": "validate session tokens and cryptographic credentials",
                "embedding": emb_c,
            },
        ]

        mem0_mock = MagicMock(spec=Mem0Retriever)
        mem0_mock.search.return_value = []

        retriever = TargetedVectorRetriever(
            codebase_retriever=codebase_mock,
            mem0_retriever=mem0_mock,
            threshold=0.75,
        )

        result = retriever.retrieve("database setup", codebase_gate=1.0, mem0_gate=0.0)

        # Lexical Jaccard between summary 1 and 2 is 0.0:
        jaccard = compute_jaccard_similarity(
            "initialize storage repository connections",
            "boot up persistency sockets cluster"
        )
        self.assertEqual(jaccard, 0.0)

        # But Vector MMR recognizes emb_a and emb_b are semantically identical (cosine >= 0.80),
        # so db_beta.py must be suppressed!
        accepted_paths = [m["path"] for m in result["codebase_matches"]]
        self.assertIn("/path/to/db_alpha.py", accepted_paths)
        self.assertNotIn("/path/to/db_beta.py", accepted_paths)
        self.assertIn("/path/to/auth.py", accepted_paths)
        self.assertEqual(len(accepted_paths), 2)

    def test_no_op_empty_results(self):
        codebase_mock = MagicMock(spec=CodebaseRetriever)
        codebase_mock.search.return_value = []

        mem0_mock = MagicMock(spec=Mem0Retriever)
        mem0_mock.search.return_value = []

        retriever = TargetedVectorRetriever(
            codebase_retriever=codebase_mock,
            mem0_retriever=mem0_mock,
            threshold=0.75
        )

        result = retriever.retrieve("query", codebase_gate=1.0, mem0_gate=1.0)

        self.assertEqual(result["codebase_matches"], [])
        self.assertEqual(result["mem0_matches"], [])
        self.assertEqual(result["formatted_context"], "")

if __name__ == '__main__':
    unittest.main()
