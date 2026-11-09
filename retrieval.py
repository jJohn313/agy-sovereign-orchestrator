"""
Targeted Hybrid Retrieval & Fault-Tolerant AST Parsing Layer for AGY.
Wraps Codebase index and Mem0 vector database with:
- Hybrid Code Search: Dense (sqlite-vec) + Sparse (FTS5 BM25)
- Reciprocal Rank Fusion (RRF)
- Semantic Vector MMR Deduplication (cosine similarity >= 0.80) replacing Jaccard
- Strict Top-4 candidate budget and canonical [codebase:path:lines] tagging
- SQLite WAL concurrency protection
"""

import os
import ast
import re
import sqlite3
import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

logger = logging.getLogger("agy.retrieval")

from ephemeral_scout import EphemeralScout

DEFAULT_SIMILARITY_THRESHOLD = 0.75
WORKSPACES_DB_PATH = os.path.expanduser("~/.cache/workspaces_vec.db")
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

_embedder = None


def compute_jaccard_similarity(text_a: str, text_b: str) -> float:
    """Computes word-level Jaccard similarity between two strings."""
    tokens_a = set(re.findall(r"\w+", text_a.lower()))
    tokens_b = set(re.findall(r"\w+", text_b.lower()))
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)
    return intersection / union if union > 0 else 0.0


def compute_cosine_similarity(vec_a: Any, vec_b: Any) -> float:
    """Computes cosine similarity between two 1D embedding vectors."""
    if vec_a is None or vec_b is None:
        return 0.0
    try:
        a = np.asarray(vec_a, dtype=np.float32)
        b = np.asarray(vec_b, dtype=np.float32)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
    except Exception:
        return 0.0


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
    to fast regex extraction of definitions.
    If target_symbol is provided, attempts to capture its full body as well.
    """
    outline_lines: List[str] = []
    symbol_body: Optional[str] = None

    try:
        parsed = ast.parse(code)
        code_lines = code.splitlines()

        for node in parsed.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
                outline_lines.append(f"{prefix} {node.name}(...) (line {node.lineno})")
                if target_symbol and node.name == target_symbol:
                    end_lineno = getattr(node, "end_lineno", node.lineno)
                    symbol_body = "\n".join(code_lines[node.lineno - 1 : end_lineno])

            elif isinstance(node, ast.ClassDef):
                outline_lines.append(f"class {node.name} (line {node.lineno}):")
                if target_symbol and node.name == target_symbol:
                    end_lineno = getattr(node, "end_lineno", node.lineno)
                    symbol_body = "\n".join(code_lines[node.lineno - 1 : end_lineno])

                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        prefix = "async def" if isinstance(item, ast.AsyncFunctionDef) else "def"
                        outline_lines.append(f"    {prefix} {item.name}(...) (line {item.lineno})")
                        if target_symbol and item.name == target_symbol:
                            end_lineno = getattr(item, "end_lineno", item.lineno)
                            symbol_body = "\n".join(code_lines[item.lineno - 1 : end_lineno])

    except Exception:
        # Fault-tolerant regex fallback
        outline_lines = ["[AST Fallback: Regex Structural Extraction]"]
        code_lines = code.splitlines()
        for idx, line in enumerate(code_lines, 1):
            if STRUCTURAL_REGEX.match(line):
                outline_lines.append(f"{line.strip()} (line {idx})")
                if target_symbol and target_symbol in line:
                    symbol_body = "\n".join(code_lines[idx - 1 : min(idx + 30, len(code_lines))])

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


def sanitize_fts5_query(query: str) -> str:
    """Sanitize raw prompt text into safe FTS5 query terms."""
    terms = re.findall(r"[a-zA-Z0-9_\.]+", query)
    if not terms:
        return ""
    safe_terms = [f'"{t}"' for t in terms[:12] if len(t) >= 2]
    return " OR ".join(safe_terms)


class CodebaseRetriever:
    """
    Hybrid Code Search over AST code chunks and workspaces index.
    Combines dense sqlite-vec embeddings with sparse FTS5 BM25 ranks via RRF.
    """

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
        Query hybrid code index using sqlite-vec + FTS5 BM25 with RRF.
        Returns candidates with precomputed embeddings for downstream MMR.
        """
        conn = self._get_connection()
        if conn is None:
            return []

        embedder = get_embedder()
        try:
            # 1. Check if code_chunks table exists
            table_check = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='code_chunks';"
            ).fetchone()
            has_code_chunks = bool(table_check)

            results: List[Dict[str, Any]] = []

            if has_code_chunks:
                results = self._search_hybrid_chunks(conn, query, embedder, limit)

            # If no code chunks found, fall back to workspaces summary index
            if not results:
                results = self._search_workspaces(conn, query, embedder, limit)

            conn.close()
            return results

        except Exception as err:
            logger.error("Error in hybrid codebase search: %s", err)
            try:
                conn.close()
            except Exception:
                pass
            return []

    def _search_hybrid_chunks(
        self,
        conn: sqlite3.Connection,
        query: str,
        embedder: Any,
        limit: int,
    ) -> List[Dict[str, Any]]:
        """Execute dense vec_code_chunks + sparse code_fts and merge via RRF."""
        import sqlite_vec

        dense_candidates: Dict[int, Dict[str, Any]] = {}
        dense_ranks: Dict[int, int] = {}

        # 1. Dense Search
        if embedder is not None:
            embeddings = list(embedder.embed([query]))
            if embeddings:
                query_embedding = embeddings[0].tolist()
                query_bytes = sqlite_vec.serialize_float32(query_embedding)

                query_sql = """
                    SELECT c.id, c.file_path, c.symbol_breadcrumb, c.symbol_type,
                           c.start_line, c.end_line, c.content, v.distance, v.embedding
                    FROM vec_code_chunks v
                    JOIN code_chunks c ON c.id = v.chunk_id
                    WHERE v.embedding MATCH ? AND k = ?
                    ORDER BY v.distance ASC
                """
                try:
                    rows = conn.execute(query_sql, (query_bytes, limit * 4)).fetchall()
                    for rank_idx, (cid, path, breadcrumb, stype, start_l, end_l, content, dist, emb_blob) in enumerate(rows, 1):
                        sim = max(0.0, 1.0 - dist)
                        emb = np.frombuffer(emb_blob, dtype=np.float32).tolist() if emb_blob else None
                        dense_candidates[cid] = {
                            "id": cid,
                            "path": path,
                            "name": breadcrumb,
                            "summary": content,
                            "symbol_type": stype,
                            "start_line": start_l,
                            "end_line": end_l,
                            "dense_sim": sim,
                            "embedding": emb,
                        }
                        dense_ranks[cid] = rank_idx
                except Exception as e:
                    logger.debug("Dense chunk query error: %s", e)

        # 2. Sparse FTS5 Search
        sparse_candidates: Dict[int, Dict[str, Any]] = {}
        sparse_ranks: Dict[int, int] = {}
        fts_query = sanitize_fts5_query(query)

        if fts_query:
            fts_sql = """
                SELECT c.id, c.file_path, c.symbol_breadcrumb, c.symbol_type,
                       c.start_line, c.end_line, c.content, bm25(code_fts) as score
                FROM code_fts
                JOIN code_chunks c ON c.id = code_fts.rowid
                WHERE code_fts MATCH ?
                ORDER BY score ASC
                LIMIT ?
            """
            try:
                rows = conn.execute(fts_sql, (fts_query, limit * 4)).fetchall()
                for rank_idx, (cid, path, breadcrumb, stype, start_l, end_l, content, bm_score) in enumerate(rows, 1):
                    sparse_candidates[cid] = {
                        "id": cid,
                        "path": path,
                        "name": breadcrumb,
                        "summary": content,
                        "symbol_type": stype,
                        "start_line": start_l,
                        "end_line": end_l,
                        "bm25_score": bm_score,
                    }
                    sparse_ranks[cid] = rank_idx
            except Exception as e:
                logger.debug("Sparse chunk query error: %s", e)

        # 3. Reciprocal Rank Fusion (RRF)
        all_ids = set(dense_candidates.keys()) | set(sparse_candidates.keys())
        if not all_ids:
            return []

        fused = []
        k_rrf = 60
        w_dense = 0.6
        w_sparse = 0.4

        for cid in all_ids:
            score_dense = 0.0
            score_sparse = 0.0
            if cid in dense_ranks:
                score_dense = w_dense / (k_rrf + dense_ranks[cid])
            if cid in sparse_ranks:
                score_sparse = w_sparse / (k_rrf + sparse_ranks[cid])
            rrf_score = score_dense + score_sparse

            # Retrieve candidate metadata
            cand = dense_candidates.get(cid) or sparse_candidates.get(cid)
            cand_copy = dict(cand)

            # Compute normalized similarity score for Layer 5 / gating threshold
            # Dense matches use dense_sim boosted by sparse match presence;
            # Sparse exact symbol hits receive authoritative score (>= 0.85)
            if cid in dense_candidates and cid in sparse_ranks:
                dense_sim = cand_copy.get("dense_sim", 0.0)
                final_sim = max(min(1.0, dense_sim + 0.08), 0.85)
            elif cid in dense_candidates:
                final_sim = cand_copy.get("dense_sim", 0.0)
            else:
                final_sim = 0.85  # Pure lexical exact hit

            cand_copy["rrf_score"] = rrf_score
            cand_copy["similarity"] = round(final_sim, 4)

            # Fetch embedding if not already present
            if cand_copy.get("embedding") is None:
                row_emb = conn.execute("SELECT embedding FROM vec_code_chunks WHERE chunk_id = ?", (cid,)).fetchone()
                if row_emb and row_emb[0]:
                    cand_copy["embedding"] = np.frombuffer(row_emb[0], dtype=np.float32).tolist()

            if cand_copy["similarity"] >= self.threshold:
                fused.append(cand_copy)

        # Sort descending by RRF score
        fused.sort(key=lambda x: x["rrf_score"], reverse=True)
        return fused[:limit]

    def _search_workspaces(
        self,
        conn: sqlite3.Connection,
        query: str,
        embedder: Any,
        limit: int,
    ) -> List[Dict[str, Any]]:
        """Legacy fallback to workspaces table."""
        if embedder is None:
            return []
        ws_check = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='vec_workspaces';"
        ).fetchone()
        if not ws_check:
            return []
        import sqlite_vec
        embeddings = list(embedder.embed([query]))
        if not embeddings:
            return []
        query_embedding = embeddings[0].tolist()
        query_bytes = sqlite_vec.serialize_float32(query_embedding)

        query_sql = """
            SELECT w.path, w.name, w.summary, v.distance, v.embedding
            FROM vec_workspaces v
            JOIN workspaces w ON w.id = v.workspace_id
            WHERE v.embedding MATCH ? AND k = ?
            ORDER BY v.distance ASC
        """
        rows = conn.execute(query_sql, (query_bytes, limit)).fetchall()
        filtered_results: List[Dict[str, Any]] = []
        for path, name, summary, dist, emb_blob in rows:
            similarity = max(0.0, 1.0 - dist)
            if similarity >= self.threshold:
                emb = np.frombuffer(emb_blob, dtype=np.float32).tolist() if emb_blob else None
                filtered_results.append({
                    "path": path,
                    "name": name,
                    "summary": summary,
                    "similarity": round(similarity, 4),
                    "embedding": emb,
                })
        return filtered_results


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
                    "embedder": {
                        "provider": "fastembed",
                        "config": {
                            "model": EMBEDDING_MODEL_NAME,
                        },
                    },
                    "version": "v1.1",
                }
                self._memory = Memory.from_config(mem_config)
            except Exception as e:
                logger.error("Failed connecting to Mem0 memory: %s", e)
                self._memory = None
        return self._memory

    def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Query user memory vector store.
        Filters strictly by cosine similarity >= threshold.
        """
        memory = self._get_memory()
        if memory is None:
            return []

        try:
            try:
                results = memory.search(query=query, filters={"user_id": self.user_id}, limit=limit)
            except Exception:
                results = memory.search(query=query, user_id=self.user_id, limit=limit)
            filtered_results: List[Dict[str, Any]] = []
            memories = results.get("results", []) if isinstance(results, dict) else results

            for item in memories:
                score = item.get("score", 0.0)
                if score >= self.threshold:
                    filtered_results.append({
                        "id": item.get("id"),
                        "memory": item.get("memory", ""),
                        "score": round(score, 4),
                    })
            return filtered_results
        except Exception as err:
            logger.error("Error querying Mem0: %s", err)
            return []


class TargetedVectorRetriever:
    """
    Retrieval coordinator: Codebase + Mem0 vector retrieval.
    Enforces:
    - Codebase Gate > 0.5 & Mem0 Gate > 0.5 triggers
    - Hard cap max_k = 4
    - Semantic Vector MMR (cosine similarity >= 0.80) replacing lexical Jaccard
    - Canonical Layer 5 output formatting
    """

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
        - If Codebase Gate > 0.5: Query Codebase index, filter >= threshold.
        - If Mem0 Gate > 0.5: Query Mem0 vector DB, filter >= threshold.
        - Applies dynamic Top-K capping (max_k=4) and Semantic Vector MMR deduplication.
        """
        context: Dict[str, Any] = {
            "codebase_matches": [],
            "mem0_matches": [],
            "formatted_context": "",
            "pre_resolved_scope": "",
        }

        candidates = []

        if codebase_gate > 0.5:
            matches = self.codebase_retriever.search(user_prompt)
            context["codebase_matches"] = matches
            for m in matches:
                candidates.append({
                    "source": "codebase",
                    "score": m["similarity"],
                    "path": m["path"],
                    "name": m["name"],
                    "text": m.get("summary") or m.get("name") or "",
                    "embedding": m.get("embedding"),
                    "raw": m,
                })

        if mem0_gate > 0.5:
            matches = self.mem0_retriever.search(user_prompt)
            context["mem0_matches"] = matches
            for m in matches:
                candidates.append({
                    "source": "mem0",
                    "score": m["score"],
                    "id": m.get("id"),
                    "text": m.get("memory") or "",
                    "embedding": m.get("embedding"),
                    "raw": m,
                })

        if not candidates:
            if codebase_gate > 0.5:
                scout = EphemeralScout()
                if scout.is_available():
                    cwd = os.getcwd()
                    authoritative_targets = scout.resolve_scope(user_prompt, cwd)
                    if authoritative_targets:
                        lines = [
                            "### Pre-Resolved Execution Scope (Authoritative)",
                            "The local engine has resolved the target files for this inquiry:",
                        ]
                        for target in authoritative_targets:
                            lines.append(f"- {target}")
                        context["pre_resolved_scope"] = "\n".join(lines)
            return context

        # Sort combined candidates descending by score
        candidates.sort(key=lambda x: x["score"], reverse=True)

        accepted = []
        max_k = 4

        for cand in candidates:
            if len(accepted) >= max_k:
                break

            cand_emb = cand.get("embedding")
            if cand_emb is None and cand.get("text"):
                # Lazily embed candidate text for vector comparison if not pre-computed
                embedder = get_embedder()
                if embedder is not None:
                    try:
                        embs = list(embedder.embed([cand["text"]]))
                        if embs:
                            cand_emb = embs[0].tolist()
                            cand["embedding"] = cand_emb
                    except Exception:
                        cand_emb = None

            is_redundant = False
            if cand_emb is not None:
                for acc in accepted:
                    acc_emb = acc.get("embedding")
                    if acc_emb is not None:
                        sim = compute_cosine_similarity(cand_emb, acc_emb)
                        if sim >= 0.80:
                            is_redundant = True
                            break
                    else:
                        # Fallback to Jaccard if comparing against unvectorized item
                        if compute_jaccard_similarity(cand["text"], acc["text"]) > 0.60:
                            is_redundant = True
                            break
            else:
                # Fallback to lexical Jaccard if no embedding is available
                for acc in accepted:
                    if compute_jaccard_similarity(cand["text"], acc["text"]) > 0.60:
                        is_redundant = True
                        break

            if not is_redundant:
                accepted.append(cand)

        if not accepted:
            return context

        # Format surviving chunks strictly to preserve Layer 5 prefix cache format
        lines = []
        for item in accepted:
            if item["source"] == "codebase":
                path = item["path"]
                raw = item.get("raw", {})
                start_line = raw.get("start_line")
                end_line = raw.get("end_line")
                if start_line is not None and end_line is not None:
                    tag = f"[codebase:{path}:{start_line}-{end_line}]"
                else:
                    tag = f"[codebase:{path}]"

                summary_snippet = item["text"].strip().split("\n")[0]
                lines.append(f"- {tag} ({item['score']:.2f}) {summary_snippet}")
            elif item["source"] == "mem0":
                mem_id = item.get("id")
                tag_id = mem_id if mem_id is not None else "rule"
                tag = f"[mem0:{tag_id}]"
                lines.append(f"- {tag} ({item['score']:.2f}) {item['text']}")

        context["formatted_context"] = "\n".join(lines)
        context["codebase_matches"] = [item["raw"] for item in accepted if item["source"] == "codebase"]
        context["mem0_matches"] = [item["raw"] for item in accepted if item["source"] == "mem0"]

        return context
