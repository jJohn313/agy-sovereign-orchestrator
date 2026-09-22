"""
Targeted Vector Retrieval & Fault-Tolerant AST Parsing Layer for AGY (Fix 8 & Fix 9).
Wraps Codebase index and Mem0 vector database with strict cosine similarity filtering (>= 0.75),
SQLite WAL concurrency, and fault-tolerant code outline extraction.
"""

import os
import ast
import re
import sqlite3
import logging
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger("agy.retrieval")

DEFAULT_SIMILARITY_THRESHOLD = 0.75
WORKSPACES_DB_PATH = os.path.expanduser("~/.cache/workspaces_vec.db")
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

_embedder = None

# Regex fallback pattern for broken code or non-Python code
STRUCTURAL_REGEX = re.compile(
    r"^(?:[ \t]*)(?:async\s+def\s+[a-zA-Z0-9_]+|def\s+[a-zA-Z0-9_]+|class\s+[a-zA-Z0-9_]+|export\s+(?:default\s+)?(?:function|class|const|let|var)\s+[a-zA-Z0-9_]+|function\s+[a-zA-Z0-9_]+)",
    re.MULTILINE,
)


def get_embedder():
    """Lazily load FastEmbed text embedding model."""
    global _embedder
    if _embedder is None:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        try:
            from fastembed import TextEmbedding
            _embedder = TextEmbedding(model_name=EMBEDDING_MODEL_NAME)
        except Exception as e:
            logger.warning("Could not initialize FastEmbed embedder: %s", e)
            _embedder = None
    return _embedder


def extract_ast_outline(code: str, target_symbol: Optional[str] = None) -> str:
    """
    Fault-tolerant structural outline extraction.
    Tries Python ast.parse(); on SyntaxError or parsing error, falls back immediately
    to a regex-based structural scanner.
    """
    outline_lines: List[str] = []
    symbol_body: Optional[str] = None

    try:
        tree = ast.parse(code)
        lines = code.splitlines()

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                start = getattr(node, "lineno", 1) - 1
                end = getattr(node, "end_lineno", start + 1)
                outline_lines.append("\n".join(lines[start:end]))
            elif isinstance(node, ast.ClassDef):
                doc = ast.get_docstring(node)
                doc_summary = f'    """{doc.splitlines()[0]}"""' if doc else ""
                outline_lines.append(f"\nclass {node.name}:")
                if doc_summary:
                    outline_lines.append(doc_summary)

                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        fdoc = ast.get_docstring(item)
                        fdoc_sum = f'        """{fdoc.splitlines()[0]}"""' if fdoc else ""
                        args_list = [a.arg for a in item.args.args]
                        outline_lines.append(f"    def {item.name}({', '.join(args_list)}): ...")
                        if fdoc_sum:
                            outline_lines.append(fdoc_sum)

                if target_symbol and node.name == target_symbol:
                    start = getattr(node, "lineno", 1) - 1
                    end = getattr(node, "end_lineno", len(lines))
                    symbol_body = "\n".join(lines[start:end])

            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                doc = ast.get_docstring(node)
                doc_summary = f'    """{doc.splitlines()[0]}"""' if doc else ""
                args_list = [a.arg for a in node.args.args]
                prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
                outline_lines.append(f"{prefix} {node.name}({', '.join(args_list)}): ...")
                if doc_summary:
                    outline_lines.append(doc_summary)

                if target_symbol and node.name == target_symbol:
                    start = getattr(node, "lineno", 1) - 1
                    end = getattr(node, "end_lineno", len(lines))
                    symbol_body = "\n".join(lines[start:end])

    except Exception as parse_err:
        logger.info("AST parse failed (%s); falling back to regex structural scan", parse_err)
        lines = code.splitlines()
        outline_lines.append("# [AST Fallback: Regex Structural Extraction]")
        for idx, line in enumerate(lines, 1):
            if STRUCTURAL_REGEX.match(line):
                outline_lines.append(f"{line.rstrip()}  # L{idx}")
                if target_symbol and target_symbol in line:
                    # Capture surrounding block
                    symbol_body = "\n".join(lines[max(0, idx - 1) : min(len(lines), idx + 35)])

    header = "\n".join(outline_lines)
    if symbol_body:
        return f"{header}\n\n# --- Full Target Symbol Body: {target_symbol} ---\n{symbol_body}"
    return header


def read_file_content_smart(file_path: str, target_symbol: Optional[str] = None) -> str:
    """
    Context Extraction Strategy:
    - Files <= 300 lines: Inject full content.
    - Files > 300 lines: Inject structural outline + full body of target symbol.
    """
    if not os.path.exists(file_path):
        return f"# Error: File not found: {file_path}"

    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()

        lines = content.splitlines()
        if len(lines) <= 300:
            return content

        outline = extract_ast_outline(content, target_symbol=target_symbol)
        return (
            f"# File: {file_path} ({len(lines)} lines - Exceeds 300 lines; condensed structural view)\n"
            f"{outline}"
        )
    except Exception as err:
        return f"# Error reading {file_path}: {err}"


class CodebaseRetriever:
    """Vector search over workspaces and codebase index with SQLite WAL protection."""

    def __init__(self, db_path: str = WORKSPACES_DB_PATH, threshold: float = DEFAULT_SIMILARITY_THRESHOLD):
        self.db_path = db_path
        self.threshold = threshold

    def _get_connection(self) -> Optional[sqlite3.Connection]:
        if not os.path.exists(self.db_path):
            return None
        try:
            import sqlite_vec
            conn = sqlite3.connect(self.db_path, timeout=5.0)
            conn.execute("PRAGMA journal_mode = WAL;")
            conn.execute("PRAGMA busy_timeout = 5000;")
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            conn.enable_load_extension(False)
            return conn
        except Exception as e:
            logger.error("Failed connecting to codebase vector DB: %s", e)
            return None

    def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Query semantic codebase index using sqlite-vec.
        Filters strictly by cosine similarity >= threshold.
        """
        conn = self._get_connection()
        if conn is None:
            return []

        embedder = get_embedder()
        if embedder is None:
            conn.close()
            return []

        try:
            import sqlite_vec
            embeddings = list(embedder.embed([query]))
            if not embeddings:
                conn.close()
                return []
            query_embedding = embeddings[0].tolist()
            query_bytes = sqlite_vec.serialize_float32(query_embedding)

            cursor = conn.cursor()
            query_sql = """
                SELECT w.path, w.name, w.summary, v.distance
                FROM vec_workspaces v
                JOIN workspaces w ON w.id = v.workspace_id
                WHERE v.embedding MATCH ? AND k = ?
                ORDER BY v.distance ASC
            """
            rows = cursor.execute(query_sql, (query_bytes, limit)).fetchall()
            conn.close()

            filtered_results: List[Dict[str, Any]] = []
            for path, name, summary, dist in rows:
                similarity = max(0.0, 1.0 - dist)
                if similarity >= self.threshold:
                    filtered_results.append({
                        "path": path,
                        "name": name,
                        "summary": summary,
                        "similarity": round(similarity, 4),
                    })
            return filtered_results
        except Exception as err:
            logger.error("Error querying codebase vector db: %s", err)
            try:
                conn.close()
            except Exception:
                pass
            return []


class Mem0Retriever:
    """Vector search over Mem0 agent memory store."""

    def __init__(self, user_id: str = "john", threshold: float = DEFAULT_SIMILARITY_THRESHOLD):
        self.user_id = user_id
        self.threshold = threshold
        self._memory = None

    def _get_memory(self):
        if self._memory is None:
            os.environ["MEM0_TELEMETRY"] = "false"
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            try:
                from mem0 import Memory
                mem_config = {
                    "vector_store": {
                        "provider": "qdrant",
                        "config": {
                            "path": os.path.expanduser("~/.local/share/mem0/qdrant"),
                            "collection_name": "agent_memories",
                            "embedding_model_dims": 384,
                        },
                    },
                    "history_db_path": os.path.expanduser("~/.local/share/mem0/history.db"),
                    "embedder": {
                        "provider": "fastembed",
                        "config": {"model": "BAAI/bge-small-en-v1.5"},
                    },
                    "llm": {
                        "provider": "openai",
                        "config": {"api_key": "dummy-local-key"},
                    },
                    "version": "v1.1",
                }
                self._memory = Memory.from_config(mem_config)
            except Exception as e:
                logger.warning("Could not initialize Mem0 memory: %s", e)
                self._memory = None
        return self._memory

    def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Query Mem0 for user preferences and architectural rules.
        Filters strictly by cosine similarity >= threshold.
        """
        memory = self._get_memory()
        if not memory:
            return []

        try:
            res = memory.search(query, filters={"user_id": self.user_id}, limit=limit)
            items = res.get("results", []) if isinstance(res, dict) else res
            filtered: List[Dict[str, Any]] = []
            for item in items:
                score = float(item.get("score", 0.0))
                if score >= self.threshold:
                    filtered.append({
                        "id": item.get("id"),
                        "memory": item.get("memory", ""),
                        "score": round(score, 4),
                        "created_at": item.get("created_at"),
                    })
            return filtered
        except Exception as err:
            logger.error("Error querying Mem0 memory: %s", err)
            return []


class TargetedVectorRetriever:
    """Unified retrieval coordinator evaluating Jev/NLI gates and retrieving context."""

    def __init__(
        self,
        codebase_retriever: Optional[CodebaseRetriever] = None,
        mem0_retriever: Optional[Mem0Retriever] = None,
        threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    ):
        self.threshold = threshold
        self.codebase_retriever = codebase_retriever or CodebaseRetriever(threshold=threshold)
        self.mem0_retriever = mem0_retriever or Mem0Retriever(threshold=threshold)

    def retrieve(
        self,
        user_prompt: str,
        codebase_gate: float,
        mem0_gate: float,
    ) -> Dict[str, Any]:
        """
        Phase 2: Targeted Vector Retrieval.
        - If Codebase Gate > 0.5: Query Codebase index, filter >= 0.75.
        - If Mem0 Gate > 0.5: Query Mem0 vector DB, filter >= 0.75.
        """
        context: Dict[str, Any] = {
            "codebase_matches": [],
            "mem0_matches": [],
            "formatted_context": "",
        }

        if codebase_gate > 0.5:
            matches = self.codebase_retriever.search(user_prompt)
            context["codebase_matches"] = matches

        if mem0_gate > 0.5:
            matches = self.mem0_retriever.search(user_prompt)
            context["mem0_matches"] = matches

        lines = []
        if context["mem0_matches"]:
            lines.append("### Relevant Durable Memory (Similarity >= 0.75):")
            for m in context["mem0_matches"]:
                lines.append(f"- [{m['score']:.2f}] {m['memory']}")

        if context["codebase_matches"]:
            lines.append("### Relevant Codebase Workspaces (Similarity >= 0.75):")
            for c in context["codebase_matches"]:
                summary_snippet = (c.get("summary") or "").strip().split("\n")[0]
                lines.append(f"- [{c['similarity']:.2f}] {c['name']} ({c['path']}): {summary_snippet}")

        context["formatted_context"] = "\n".join(lines)
        return context
