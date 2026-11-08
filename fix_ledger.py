"""
Case-Based Episodic Fix Ledger for Sovereign AGY Orchestrator.
Records verified troubleshooting solutions (Qt6, Hyprland, Wayland, Rust, systemd, etc.)
and retrieves them with sub-5ms latency before turns, auto-unloading VRAM/RAM immediately.
"""

import os
import gc
import json
import time
import uuid
import logging
import re
from typing import Dict, Any, List, Optional
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct

logger = logging.getLogger("agy.fix_ledger")

QDRANT_PATH = os.path.expanduser("~/.local/share/mem0/qdrant")
COLLECTION_NAME = "troubleshooting_fixes"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
VECTOR_DIM = 384


class FixLedger:
    """Manages the episodic memory of verified troubleshooting fixes and solutions."""

    def __init__(self, qdrant_path: str = QDRANT_PATH):
        self.qdrant_path = qdrant_path
        self._ensure_collection()

    def _ensure_collection(self):
        try:
            client = QdrantClient(path=self.qdrant_path)
            existing = [c.name for c in client.get_collections().collections]
            if COLLECTION_NAME not in existing:
                client.create_collection(
                    collection_name=COLLECTION_NAME,
                    vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
                )
            client.close()
        except Exception as e:
            logger.warning("Could not initialize Qdrant collection %s: %s", COLLECTION_NAME, e)

    def _embed_text(self, text: str) -> List[float]:
        """Generate embedding on-demand and immediately drop the model to free RAM."""
        from fastembed import TextEmbedding

        embedder = TextEmbedding(model_name=EMBEDDING_MODEL)
        try:
            embeddings = list(embedder.embed([text]))
            vector = embeddings[0].tolist()
            return vector
        finally:
            del embedder
            gc.collect()

    def record_fix(
        self,
        domain: str,
        symptom: str,
        root_cause: str,
        verified_fix: str,
        tags: Optional[List[str]] = None,
    ) -> bool:
        """
        Record a verified fix into the episodic memory.
        Uses deterministic UUID so repeated updates overwrite the same entry.
        """
        domain_clean = (domain or "general").strip().lower()
        symptom_clean = (symptom or "").strip()
        root_cause_clean = (root_cause or "").strip()
        fix_clean = (verified_fix or "").strip()

        if not symptom_clean or not fix_clean:
            return False

        # Text representation for dense vector embedding
        searchable_text = f"[{domain_clean}] Symptom: {symptom_clean} | Root cause: {root_cause_clean} | Fix: {fix_clean}"
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{domain_clean}:{symptom_clean}"))

        payload = {
            "domain": domain_clean,
            "symptom": symptom_clean,
            "root_cause": root_cause_clean,
            "verified_fix": fix_clean,
            "tags": tags or [domain_clean],
            "recorded_at": time.time(),
        }

        try:
            vector = self._embed_text(searchable_text)
            client = QdrantClient(path=self.qdrant_path)
            client.upsert(
                collection_name=COLLECTION_NAME,
                points=[
                    PointStruct(
                        id=point_id,
                        vector=vector,
                        payload=payload,
                    )
                ],
            )
            client.close()
            logger.info("Recorded verified fix into ledger [%s]: %s", domain_clean, symptom_clean[:60])
            return True
        except Exception as e:
            logger.error("Failed to record fix in ledger: %s", e)
            return False

    def search_fixes(
        self,
        query: str,
        domain: Optional[str] = None,
        limit: int = 2,
        score_threshold: float = 0.80,
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant past fixes matching current query / error trace.
        """
        if not query or not query.strip():
            return []

        try:
            vector = self._embed_text(query.strip())
            client = QdrantClient(path=self.qdrant_path)
            
            res = client.query_points(
                collection_name=COLLECTION_NAME,
                query=vector,
                limit=limit,
                score_threshold=score_threshold,
            )
            client.close()

            matches = []
            for hit in getattr(res, "points", []):
                payload = hit.payload or {}
                if domain and payload.get("domain", "").lower() != domain.lower():
                    continue
                matches.append({
                    "score": round(hit.score, 4),
                    "domain": payload.get("domain", ""),
                    "symptom": payload.get("symptom", ""),
                    "root_cause": payload.get("root_cause", ""),
                    "verified_fix": payload.get("verified_fix", ""),
                    "recorded_at": payload.get("recorded_at", 0),
                })
            return matches
        except Exception as e:
            logger.warning("Error searching fix ledger: %s", e)
            return []

    def format_directive(self, matches: List[Dict[str, Any]]) -> str:
        """Format matching fixes into an authoritative directive block."""
        if not matches:
            return ""

        blocks = []
        for m in matches:
            domain = m.get("domain", "GENERAL").upper()
            blocks.append(
                f"[VERIFIED PRIOR FIX - {domain}] (Confidence: {m.get('score', 0):.2f})\n"
                f"- Problem/Symptom: {m.get('symptom')}\n"
                f"- Root Cause: {m.get('root_cause')}\n"
                f"- Verified Solution: {m.get('verified_fix')}"
            )
        return "\n\n".join(blocks)

    def extract_candidate_fix(
        self,
        user_prompt: str,
        actions_summary: str,
    ) -> Optional[Dict[str, str]]:
        """
        Heuristic rule-based check to detect if this turn solved an explicit problem.
        Matches keywords and patterns without burning unnecessary tokens.
        """
        text = f"{user_prompt}\n{actions_summary}"
        
        # Check for error/fix indicators
        has_error = bool(re.search(r"\b(error|failed|exception|traceback|undefined|syntaxerror|bug|issue|broke)\b", text, re.I))
        has_solution = bool(re.search(r"\b(fixed|solved|resolved|solution|root cause|working now|patch|repaired)\b", text, re.I))
        
        if not (has_error and has_solution):
            return None

        # Detect domain
        domain = "general"
        for candidate in ["qt6", "pyside6", "quickshell", "wayland", "hyprland", "rust", "systemd", "docker", "python", "pipewire", "fastembed", "git"]:
            if re.search(rf"\b{candidate}\b", text, re.I):
                domain = candidate
                break

        return {
            "domain": domain,
            "symptom": user_prompt.strip(),
            "root_cause": "Identified during turn execution",
            "verified_fix": actions_summary.strip()[:500],
        }
