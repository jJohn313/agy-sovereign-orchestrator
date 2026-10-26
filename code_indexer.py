"""
AST-Aware Hybrid Code Indexer for AGY Sovereign Orchestrator.
Manages:
- AST semantic code chunking (Tree-sitter + Python ast fallback)
- Granular symbol breadcrumbs (e.g. 'file.py > Class > method')
- Incremental Git commit diff re-indexing (HEAD != last_indexed_commit)
- SQLite WAL transactions, FTS5 triggers, and vec0 virtual table cleanup
"""

import os
import sys
import ast
import re
import sqlite3
import hashlib
import logging
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger("agy.code_indexer")

WORKSPACES_DB_PATH = os.path.expanduser("~/.cache/workspaces_vec.db")
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# Supported code file extensions
SUPPORTED_EXTENSIONS = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".sh": "bash",
    ".bash": "bash",
    ".c": "c",
    ".cpp": "cpp",
    ".h": "c",
    ".hpp": "cpp",
    ".rs": "rust",
    ".go": "go",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".md": "markdown",
}

_embedder = None


def get_embedder():
    """Lazily load FastEmbed model."""
    global _embedder
    if _embedder is None:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        try:
            from fastembed import TextEmbedding
            _embedder = TextEmbedding(model_name=EMBEDDING_MODEL_NAME)
        except Exception as e:
            logger.error("Failed to initialize FastEmbed in code_indexer: %s", e)
            _embedder = None
    return _embedder


def ensure_db_schema(conn: sqlite3.Connection):
    """
    Ensure all hybrid tables, virtual tables, triggers, and indices exist.
    Applies explicit BEFORE DELETE trigger on code_chunks to delete from vec_code_chunks
    to prevent ghost vectors.
    """
    with conn:
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA busy_timeout = 5000;")
        conn.execute("PRAGMA foreign_keys = ON;")

        # 1. Repositories table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS code_repos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                repo_path TEXT UNIQUE NOT NULL,
                repo_name TEXT NOT NULL,
                last_indexed_commit TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        # 2. Granular AST-Aware Code Chunks table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS code_chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                repo_id INTEGER NOT NULL REFERENCES code_repos(id) ON DELETE CASCADE,
                file_path TEXT NOT NULL,
                language TEXT NOT NULL,
                symbol_breadcrumb TEXT NOT NULL,
                symbol_type TEXT NOT NULL,
                start_line INTEGER NOT NULL,
                end_line INTEGER NOT NULL,
                content TEXT NOT NULL,
                token_count INTEGER,
                file_hash TEXT NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_code_chunks_repo ON code_chunks(repo_id);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_code_chunks_path ON code_chunks(file_path);")

        # 3. Dense Embedding virtual table (vec0)
        conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS vec_code_chunks USING vec0(
                chunk_id INTEGER PRIMARY KEY,
                embedding float[384] distance_metric=cosine
            );
        """)

        # 4. Sparse FTS5 external content table
        conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS code_fts USING fts5(
                content,
                symbol_breadcrumb,
                file_path,
                content='code_chunks',
                content_rowid='id',
                tokenize="unicode61 tokenchars '._'"
            );
        """)

        # 5. BEFORE DELETE trigger to delete matching chunk_id from vec0
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS trg_code_chunks_bd BEFORE DELETE ON code_chunks BEGIN
                DELETE FROM vec_code_chunks WHERE chunk_id = old.id;
            END;
        """)

        # 6. FTS5 Synchronization triggers
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS trg_code_chunks_ai AFTER INSERT ON code_chunks BEGIN
                INSERT INTO code_fts(rowid, content, symbol_breadcrumb, file_path)
                VALUES (new.id, new.content, new.symbol_breadcrumb, new.file_path);
            END;
        """)

        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS trg_code_chunks_ad AFTER DELETE ON code_chunks BEGIN
                INSERT INTO code_fts(code_fts, rowid, content, symbol_breadcrumb, file_path)
                VALUES ('delete', old.id, old.content, old.symbol_breadcrumb, old.file_path);
            END;
        """)

        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS trg_code_chunks_au AFTER UPDATE ON code_chunks BEGIN
                INSERT INTO code_fts(code_fts, rowid, content, symbol_breadcrumb, file_path)
                VALUES ('delete', old.id, old.content, old.symbol_breadcrumb, old.file_path);
                INSERT INTO code_fts(rowid, content, symbol_breadcrumb, file_path)
                VALUES (new.id, new.content, new.symbol_breadcrumb, new.file_path);
            END;
        """)


def get_db_connection(db_path: str = WORKSPACES_DB_PATH) -> sqlite3.Connection:
    """Connect to SQLite database and register sqlite-vec extension."""
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    import sqlite_vec
    conn = sqlite3.connect(db_path, timeout=5.0)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    ensure_db_schema(conn)
    return conn


class ASTChunker:
    """Parses source files into semantic chunks along class/function/method boundaries."""

    @classmethod
    def chunk_file(cls, rel_path: str, content: str) -> List[Dict[str, Any]]:
        """Slice file into semantic chunks with breadcrumbs."""
        if not content.strip():
            return []

        ext = Path(rel_path).suffix.lower()
        lang = SUPPORTED_EXTENSIONS.get(ext, "unknown")
        lines = content.splitlines(keepends=True)
        total_lines = len(lines)

        chunks = []
        if lang == "python":
            chunks = cls._chunk_python(rel_path, content, lines)
        elif lang in ("javascript", "typescript"):
            chunks = cls._chunk_tree_sitter_js(rel_path, content, lines)

        # Fallback to structural regex chunker if AST produced no chunks
        if not chunks:
            chunks = cls._chunk_regex_fallback(rel_path, content, lines, lang)

        # If file had no extractable classes/functions, index as a single file chunk
        if not chunks:
            chunks = [{
                "symbol_breadcrumb": rel_path,
                "symbol_type": "file",
                "start_line": 1,
                "end_line": total_lines,
                "content": content,
                "token_count": len(content.split()),
                "language": lang,
            }]

        return chunks

    @classmethod
    def _chunk_python(cls, rel_path: str, content: str, lines: List[str]) -> List[Dict[str, Any]]:
        """Chunk Python code using Tree-sitter or native ast module."""
        # Try tree-sitter python first
        try:
            import tree_sitter_python
            from tree_sitter import Language, Parser
            py_lang = Language(tree_sitter_python.language())
            parser = Parser(py_lang)
            tree = parser.parse(content.encode("utf-8"))
            chunks = cls._extract_ts_nodes(rel_path, tree.root_node, lines, "python")
            if chunks:
                return chunks
        except Exception:
            pass

        # Fallback to standard library ast module
        try:
            tree = ast.parse(content)
            chunks: List[Dict[str, Any]] = []

            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    start = node.lineno
                    end = getattr(node, "end_lineno", start + 20)
                    chunk_text = "".join(lines[start - 1:end])
                    chunks.append({
                        "symbol_breadcrumb": f"{rel_path} > {node.name}",
                        "symbol_type": "function",
                        "start_line": start,
                        "end_line": end,
                        "content": chunk_text,
                        "token_count": len(chunk_text.split()),
                        "language": "python",
                    })
                elif isinstance(node, ast.ClassDef):
                    class_name = node.name
                    # Class header / docstring
                    class_start = node.lineno
                    class_end = getattr(node, "end_lineno", class_start + 30)

                    # Inspect methods
                    method_found = False
                    for item in node.body:
                        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            method_found = True
                            m_start = item.lineno
                            m_end = getattr(item, "end_lineno", m_start + 15)
                            m_text = "".join(lines[m_start - 1:m_end])
                            chunks.append({
                                "symbol_breadcrumb": f"{rel_path} > {class_name} > {item.name}",
                                "symbol_type": "method",
                                "start_line": m_start,
                                "end_line": m_end,
                                "content": m_text,
                                "token_count": len(m_text.split()),
                                "language": "python",
                            })

                    if not method_found or (class_end - class_start) <= 60:
                        class_text = "".join(lines[class_start - 1:class_end])
                        chunks.append({
                            "symbol_breadcrumb": f"{rel_path} > {class_name}",
                            "symbol_type": "class",
                            "start_line": class_start,
                            "end_line": class_end,
                            "content": class_text,
                            "token_count": len(class_text.split()),
                            "language": "python",
                        })

            return chunks
        except Exception:
            return []

    @classmethod
    def _chunk_tree_sitter_js(cls, rel_path: str, content: str, lines: List[str]) -> List[Dict[str, Any]]:
        """Chunk JavaScript/TypeScript with tree-sitter-javascript if available."""
        try:
            import tree_sitter_javascript
            from tree_sitter import Language, Parser
            js_lang = Language(tree_sitter_javascript.language())
            parser = Parser(js_lang)
            tree = parser.parse(content.encode("utf-8"))
            return cls._extract_ts_nodes(rel_path, tree.root_node, lines, "javascript")
        except Exception:
            return []

    @classmethod
    def _extract_ts_nodes(cls, rel_path: str, root_node, lines: List[str], lang: str) -> List[Dict[str, Any]]:
        """Extract top-level functions, classes, and methods from Tree-sitter AST."""
        chunks: List[Dict[str, Any]] = []

        def get_node_text(node) -> str:
            return "".join(lines[node.start_point[0]:node.end_point[0] + 1])

        def get_identifier(node) -> str:
            for child in node.children:
                if child.type in ("identifier", "name"):
                    return child.text.decode("utf-8", errors="ignore")
            return "anonymous"

        for child in root_node.children:
            if child.type in ("function_definition", "async_function_definition", "function_declaration"):
                name = get_identifier(child)
                start = child.start_point[0] + 1
                end = child.end_point[0] + 1
                text = get_node_text(child)
                chunks.append({
                    "symbol_breadcrumb": f"{rel_path} > {name}",
                    "symbol_type": "function",
                    "start_line": start,
                    "end_line": end,
                    "content": text,
                    "token_count": len(text.split()),
                    "language": lang,
                })
            elif child.type in ("class_definition", "class_declaration"):
                class_name = get_identifier(child)
                start = child.start_point[0] + 1
                end = child.end_point[0] + 1

                # Traverse class body for methods
                body_node = None
                for c in child.children:
                    if c.type in ("block", "class_body", "statement_block") or "body" in c.type:
                        body_node = c
                        break

                methods = []
                if body_node:
                    for m in body_node.children:
                        if m.type in ("function_definition", "method_definition", "async_function_definition"):
                            m_name = get_identifier(m)
                            m_start = m.start_point[0] + 1
                            m_end = m.end_point[0] + 1
                            m_text = get_node_text(m)
                            methods.append({
                                "symbol_breadcrumb": f"{rel_path} > {class_name} > {m_name}",
                                "symbol_type": "method",
                                "start_line": m_start,
                                "end_line": m_end,
                                "content": m_text,
                                "token_count": len(m_text.split()),
                                "language": lang,
                            })

                # Always add class header chunk if substantial
                class_text = get_node_text(child)
                chunks.append({
                    "symbol_breadcrumb": f"{rel_path} > {class_name}",
                    "symbol_type": "class",
                    "start_line": start,
                    "end_line": end,
                    "content": class_text if not methods else "".join(lines[start - 1:min(start + 15, end)]),
                    "token_count": len(class_text.split()),
                    "language": lang,
                })
                if methods:
                    chunks.extend(methods)

        return chunks

    @classmethod
    def _chunk_regex_fallback(cls, rel_path: str, content: str, lines: List[str], lang: str) -> List[Dict[str, Any]]:
        """Slice code files in blocks when AST parsing is unavailable."""
        pattern = re.compile(
            r"^(?:[ \t]*)(?:def|async\s+def|class|function|export\s+(?:default\s+)?function|fn|pub\s+fn)\s+([a-zA-Z0-9_]+)",
            re.MULTILINE,
        )

        matches = list(pattern.finditer(content))
        if not matches:
            # Windowed chunks
            step = 60
            chunks = []
            for i in range(0, len(lines), step):
                chunk_slice = lines[i:i + step]
                start = i + 1
                end = min(i + step, len(lines))
                text = "".join(chunk_slice)
                chunks.append({
                    "symbol_breadcrumb": f"{rel_path}:{start}-{end}",
                    "symbol_type": "block",
                    "start_line": start,
                    "end_line": end,
                    "content": text,
                    "token_count": len(text.split()),
                    "language": lang,
                })
            return chunks

        chunks = []
        for i, m in enumerate(matches):
            symbol_name = m.group(1)
            char_offset = m.start()
            start_line = content[:char_offset].count("\n") + 1
            if i + 1 < len(matches):
                next_offset = matches[i + 1].start()
                end_line = content[:next_offset].count("\n")
            else:
                end_line = len(lines)

            end_line = max(start_line, end_line)
            chunk_text = "".join(lines[start_line - 1:end_line])
            chunks.append({
                "symbol_breadcrumb": f"{rel_path} > {symbol_name}",
                "symbol_type": "symbol",
                "start_line": start_line,
                "end_line": end_line,
                "content": chunk_text,
                "token_count": len(chunk_text.split()),
                "language": lang,
            })
        return chunks


class IncrementalGitIndexer:
    """Incrementally indexes Git repositories tracking commit divergence (HEAD != last_indexed_commit)."""

    def __init__(self, db_path: str = WORKSPACES_DB_PATH):
        self.db_path = db_path

    def get_repo_info(self, target_dir: str) -> Optional[Tuple[str, str, str]]:
        """Resolve repo root, repo name, and current HEAD commit hash."""
        try:
            top_res = subprocess.run(
                ["git", "rev-parse", "--show-toplevel"],
                cwd=target_dir,
                capture_output=True,
                text=True,
                timeout=2.0,
            )
            if top_res.returncode != 0:
                return None
            repo_root = os.path.abspath(top_res.stdout.strip())
            repo_name = os.path.basename(repo_root)

            head_res = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=repo_root,
                capture_output=True,
                text=True,
                timeout=2.0,
            )
            if head_res.returncode != 0:
                return None
            head_commit = head_res.stdout.strip()
            return repo_root, repo_name, head_commit
        except Exception as e:
            logger.warning("Error resolving git repo info in %s: %s", target_dir, e)
            return None

    def check_commit_divergence(self, target_dir: str) -> Dict[str, Any]:
        """
        Check if workspace HEAD has diverged from last_indexed_commit.
        Returns: { 'diverged': bool, 'repo_path': str, 'last_commit': str, 'head_commit': str }
        """
        info = self.get_repo_info(target_dir)
        if not info:
            return {"diverged": False, "is_git": False}

        repo_root, repo_name, head_commit = info
        conn = get_db_connection(self.db_path)
        try:
            row = conn.execute(
                "SELECT last_indexed_commit FROM code_repos WHERE repo_path = ?",
                (repo_root,)
            ).fetchone()
            last_commit = row[0] if row else None
            diverged = (last_commit != head_commit)
            return {
                "diverged": diverged,
                "is_git": True,
                "repo_path": repo_root,
                "repo_name": repo_name,
                "last_commit": last_commit,
                "head_commit": head_commit,
            }
        finally:
            conn.close()

    def index_repository(self, target_dir: str, force: bool = False) -> Dict[str, Any]:
        """
        Index git repository incrementally based on commit delta.
        """
        div_info = self.check_commit_divergence(target_dir)
        if not div_info.get("is_git"):
            return {"status": "error", "message": "Not a git repository"}

        repo_root = div_info["repo_path"]
        repo_name = div_info["repo_name"]
        last_commit = div_info["last_commit"]
        head_commit = div_info["head_commit"]

        if not div_info["diverged"] and not force:
            return {"status": "unchanged", "repo_path": repo_root, "head_commit": head_commit}

        conn = get_db_connection(self.db_path)
        try:
            # Ensure repo entry
            with conn:
                cur = conn.cursor()
                cur.execute(
                    "INSERT INTO code_repos (repo_path, repo_name, last_indexed_commit) VALUES (?, ?, ?) "
                    "ON CONFLICT(repo_path) DO UPDATE SET repo_name = excluded.repo_name",
                    (repo_root, repo_name, None),
                )
                row = cur.execute("SELECT id FROM code_repos WHERE repo_path = ?", (repo_root,)).fetchone()
                repo_id = row[0]

            if last_commit is None or force:
                # Full index
                result = self._full_index(conn, repo_id, repo_root, head_commit)
            else:
                # Incremental index via git diff
                result = self._incremental_diff_index(conn, repo_id, repo_root, last_commit, head_commit)

            with conn:
                conn.execute(
                    "UPDATE code_repos SET last_indexed_commit = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                    (head_commit, repo_id),
                )

            return result
        finally:
            conn.close()

    def _full_index(self, conn: sqlite3.Connection, repo_id: int, repo_root: str, head_commit: str) -> Dict[str, Any]:
        """Index all tracked files in git repository."""
        # 1. Clear old chunks for repo (triggers trg_code_chunks_bd to clean vec_code_chunks)
        with conn:
            conn.execute("DELETE FROM code_chunks WHERE repo_id = ?", (repo_id,))

        # 2. Get git ls-files
        ls_res = subprocess.run(
            ["git", "ls-files"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
        files = [f.strip() for f in ls_res.stdout.splitlines() if f.strip()]

        all_chunks: List[Dict[str, Any]] = []
        for rel_path in files:
            ext = Path(rel_path).suffix.lower()
            if ext not in SUPPORTED_EXTENSIONS:
                continue
            abs_path = os.path.join(repo_root, rel_path)
            try:
                with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()
                file_chunks = ASTChunker.chunk_file(rel_path, content)
                file_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
                for c in file_chunks:
                    c["repo_id"] = repo_id
                    c["file_path"] = rel_path
                    c["file_hash"] = file_hash
                    all_chunks.append(c)
            except Exception as e:
                logger.debug("Failed reading %s: %s", abs_path, e)

        self._batch_insert_chunks(conn, all_chunks)
        return {
            "status": "indexed_full",
            "repo_path": repo_root,
            "head_commit": head_commit,
            "chunks_count": len(all_chunks),
        }

    def _incremental_diff_index(
        self,
        conn: sqlite3.Connection,
        repo_id: int,
        repo_root: str,
        last_commit: str,
        head_commit: str,
    ) -> Dict[str, Any]:
        """Index only added/modified files and delete removed files based on git diff."""
        diff_res = subprocess.run(
            ["git", "diff", "--name-status", last_commit, head_commit],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
        diff_lines = diff_res.stdout.strip().splitlines()

        deleted_files = []
        modified_or_added_files = []

        for line in diff_lines:
            parts = line.split(maxsplit=2)
            if not parts:
                continue
            status = parts[0][0]
            if status == "D":
                deleted_files.append(parts[1])
            elif status in ("M", "A"):
                modified_or_added_files.append(parts[1])
            elif status == "R":  # Rename: R100 old_path new_path
                if len(parts) >= 3:
                    deleted_files.append(parts[1])
                    modified_or_added_files.append(parts[2])

        # Delete chunks for removed or modified files
        files_to_purge = set(deleted_files + modified_or_added_files)
        with conn:
            for fpath in files_to_purge:
                conn.execute(
                    "DELETE FROM code_chunks WHERE repo_id = ? AND file_path = ?",
                    (repo_id, fpath),
                )

        new_chunks = []
        for rel_path in modified_or_added_files:
            ext = Path(rel_path).suffix.lower()
            if ext not in SUPPORTED_EXTENSIONS:
                continue
            abs_path = os.path.join(repo_root, rel_path)
            if not os.path.exists(abs_path):
                continue
            try:
                with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()
                file_chunks = ASTChunker.chunk_file(rel_path, content)
                file_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
                for c in file_chunks:
                    c["repo_id"] = repo_id
                    c["file_path"] = rel_path
                    c["file_hash"] = file_hash
                    new_chunks.append(c)
            except Exception as e:
                logger.debug("Failed reading %s: %s", abs_path, e)

        self._batch_insert_chunks(conn, new_chunks)
        return {
            "status": "indexed_incremental",
            "repo_path": repo_root,
            "last_commit": last_commit,
            "head_commit": head_commit,
            "purged_files": len(files_to_purge),
            "new_chunks_count": len(new_chunks),
        }

    def _batch_insert_chunks(self, conn: sqlite3.Connection, chunks: List[Dict[str, Any]]):
        """Compute embeddings and insert chunks into code_chunks & vec_code_chunks."""
        if not chunks:
            return

        embedder = get_embedder()
        import sqlite_vec

        # Prepare text representation for embedding: breadcrumb + content
        texts_to_embed = [
            f"{c['symbol_breadcrumb']}\n{c['content']}"
            for c in chunks
        ]

        embeddings = []
        batch_size = 64
        for i in range(0, len(texts_to_embed), batch_size):
            batch = texts_to_embed[i:i + batch_size]
            if embedder:
                batch_embs = list(embedder.embed(batch))
                embeddings.extend([b.tolist() for b in batch_embs])
            else:
                embeddings.extend([[0.0] * 384 for _ in batch])

        with conn:
            cur = conn.cursor()
            for chunk, emb in zip(chunks, embeddings):
                cur.execute(
                    """
                    INSERT INTO code_chunks (
                        repo_id, file_path, language, symbol_breadcrumb,
                        symbol_type, start_line, end_line, content, token_count, file_hash
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        chunk["repo_id"],
                        chunk["file_path"],
                        chunk["language"],
                        chunk["symbol_breadcrumb"],
                        chunk["symbol_type"],
                        chunk["start_line"],
                        chunk["end_line"],
                        chunk["content"],
                        chunk.get("token_count", 0),
                        chunk["file_hash"],
                    ),
                )
                chunk_id = cur.lastrowid
                emb_bytes = sqlite_vec.serialize_float32(emb)
                cur.execute(
                    "INSERT INTO vec_code_chunks (chunk_id, embedding) VALUES (?, ?)",
                    (chunk_id, emb_bytes),
                )


def main():
    import argparse
    parser = argparse.ArgumentParser(description="AGY Hybrid Code Indexer")
    parser.add_argument("--repo", default=os.getcwd(), help="Target repository path")
    parser.add_argument("--force", action="store_true", help="Force full reindex")
    args = parser.parse_args()

    indexer = IncrementalGitIndexer()
    res = indexer.index_repository(args.repo, force=args.force)
    print(res)


if __name__ == "__main__":
    main()
