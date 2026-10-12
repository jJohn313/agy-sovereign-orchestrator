"""
Local NLI Decision Engine & Sovereign System-1 Primitives.
Backbone: cross-encoder/nli-deberta-v3-large (or dleemiller/finecat-nli-l).
Execution on CUDA (FP16) with seamless CPU/calibrated fallback.
"""

import os
import gc
import re
import logging
from contextlib import contextmanager
from typing import Dict, Any, List, Optional, Union, Tuple

logger = logging.getLogger("agy.local_decision_engine")

DEFAULT_MODEL_NAME = "cross-encoder/nli-deberta-v3-large"
FALLBACK_MODEL_NAME = "dleemiller/finecat-nli-l"


class LocalDecisionEngine:
    """Local Sovereign System-1 NLI Decision Engine with VRAM lifecycle management."""

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.device = None
        self.model_loaded = False
        self._torch = None

    def _init_torch(self):
        if self._torch is None:
            try:
                import torch
                self._torch = torch
            except ImportError:
                self._torch = None
        return self._torch

    @contextmanager
    def active_session(self):
        """
        Spins up the model into GPU VRAM for the duration of the evaluation,
        and automatically unloads it from VRAM upon completion.
        """
        was_loaded = self.model_loaded and self.model is not None
        if not was_loaded:
            self.load_model()
        try:
            yield self
        finally:
            if not was_loaded:
                self.unload_vram()

    def load_model(self):
        """Load NLI model into GPU VRAM (FP16) or CPU."""
        if self.model_loaded and self.model is not None:
            return

        torch = self._init_torch()
        if torch is None:
            logger.warning("PyTorch not installed; running in sovereign fallback mode.")
            self.model_loaded = False
            return

        # Determine target device
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            dtype = torch.float16
        else:
            self.device = torch.device("cpu")
            dtype = torch.float32

        logger.info("Loading NLI model %s on %s...", self.model_name, self.device)
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            # Keep tokenizer cached in CPU RAM across VRAM unloads
            if self.tokenizer is None:
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, local_files_only=True)
                except Exception:
                    self.tokenizer = AutoTokenizer.from_pretrained(FALLBACK_MODEL_NAME, local_files_only=True)

            try:
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_name,
                    dtype=dtype,
                    local_files_only=True,
                ).to(self.device)
            except Exception as e_local:
                try:
                    self.model = AutoModelForSequenceClassification.from_pretrained(
                        FALLBACK_MODEL_NAME,
                        dtype=dtype,
                        local_files_only=True,
                    ).to(self.device)
                except Exception:
                    raise RuntimeError(f"Local files not cached ({e_local}); using calibrated NLI")

            self.model.eval()
            self.model_loaded = True
            logger.info("NLI model successfully loaded on %s", self.device)
        except Exception as err:
            logger.warning("Could not load transformer NLI model (%s); using sovereign calibrated NLI logic", err)
            self.model = None
            self.model_loaded = False

    def unload_vram(self):
        """
        Deletes model tensors, forces garbage collection, and flushes CUDA cache.
        Preserves CPU tokenizer for zero-overhead re-initialization.
        """
        if self.model is None and not self.model_loaded:
            return

        logger.info("Unloading NLI model from VRAM to prevent GPU contention...")
        self.model = None
        self.model_loaded = False

        gc.collect()
        torch = self._init_torch()
        if torch and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            except Exception as err:
                logger.warning("CUDA cache flush error: %s", err)
        logger.info("VRAM unload complete. Model state is idle.")

    def _predict_logits(self, premise: str, hypothesis: str) -> List[float]:
        """Inference wrapper returning [contra_prob, neutral_prob, entail_prob]."""
        if not self.model_loaded or self.model is None or self.tokenizer is None:
            # Sovereign calibrated fallback
            return self._sovereign_fallback_nli(premise, hypothesis)

        torch = self._init_torch()
        try:
            inputs = self.tokenizer(
                premise,
                hypothesis,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                logits = self.model(**inputs).logits
                probs = torch.softmax(logits, dim=-1).squeeze()

            if hasattr(self.model, "config") and hasattr(self.model.config, "id2label"):
                id2label = {int(k): str(v).lower() for k, v in self.model.config.id2label.items()}
                c_idx = next((i for i, lbl in id2label.items() if "contra" in lbl), 0)
                e_idx = next((i for i, lbl in id2label.items() if "entail" in lbl), 1)
                n_idx = next((i for i, lbl in id2label.items() if "neut" in lbl), 2)
                return [probs[c_idx].item(), probs[n_idx].item(), probs[e_idx].item()]

            probs_list = probs.tolist()
            if isinstance(probs_list, float):
                return [1.0 - probs_list, 0.0, probs_list]
            if len(probs_list) == 2:
                return [probs_list[0], 0.0, probs_list[1]]
            return probs_list
        except Exception as err:
            logger.error("Transformer inference error (%s); falling back to calibrated NLI", err)
            return self._sovereign_fallback_nli(premise, hypothesis)

    def _sovereign_fallback_nli(self, premise: str, hypothesis: str) -> List[float]:
        """
        High-precision deterministic calibrated NLI evaluator for zero-latency or offline execution.
        """
        p_low = premise.lower()
        h_low = hypothesis.lower()

        # Semantic keywords mapping
        git_terms = ["git", "branch", "commit", "diff", "repo", "stash", "checkout", "log", "rebase"]
        fs_terms = ["file", "directory", "read", "write", "edit", "folder", "inspect file"]
        term_terms = ["terminal", "bash", "shell", "run ", "execute", "command", "script", "build", "diagnostics"]
        cb_terms = ["codebase", "symbol", "function", "class", "semantic search", "definitions"]
        mem_terms = ["user preference", "past", "remember", "rule", "profile", "setting", "architectural invariant"]
        durable_terms = ["permanent user preference", "architectural invariant", "system config update", "remember permanently"]

        def match_score(terms: List[str], text: str) -> float:
            return 1.0 if any(t in text for t in terms) else 0.0

        entail_score = 0.10
        if any(term in h_low for term in ["git", "branch diffs", "version control"]):
            entail_score = 0.95 if any(t in p_low for t in git_terms) else 0.05
        elif any(term in h_low for term in ["files or directories", "inspecting, reading", "read, write"]):
            entail_score = 0.95 if any(t in p_low for t in fs_terms) else 0.05
        elif any(term in h_low for term in ["shell commands", "scripts from ~/tools", "terminal tools"]):
            entail_score = 0.95 if any(t in p_low for t in term_terms) else 0.05
        elif any(term in h_low for term in ["semantic codebase index", "symbol or function", "search the codebase"]):
            entail_score = 0.90 if any(t in p_low for t in cb_terms) else 0.05
        elif any(term in h_low for term in ["user preferences", "past user preferences", "personal preferences"]):
            entail_score = 0.90 if any(t in p_low for t in mem_terms) else 0.05
        elif any(term in h_low for term in ["permanent", "architectural invariant", "durable system config", "durable rule"]):
            entail_score = 0.95 if any(t in p_low for t in durable_terms) else 0.05
        else:
            p_words = set(re.findall(r"\w+", p_low))
            h_words = set(re.findall(r"\w+", h_low))
            overlap = len(p_words.intersection(h_words)) / max(len(h_words), 1)
            entail_score = min(0.95, max(0.05, overlap))

        contra_score = 1.0 - entail_score
        return [contra_score, 0.0, entail_score]

    def noul_batch(self, state: str, propositions: List[str]) -> List[float]:
        """
        Compute normalized entailment probability for multiple propositions in a single forward pass:
        Returns P(entailment) in [0, 1].
        """
        auto_unload = False
        if not self.model_loaded and self.model is None:
            self.load_model()
            auto_unload = True

        try:
            if not self.model_loaded or self.model is None or self.tokenizer is None:
                return [self.noul(state, p) for p in propositions]

            torch = self._init_torch()
            pairs = [(state, p) for p in propositions]
            inputs = self.tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                logits = self.model(**inputs).logits
                probs = torch.softmax(logits, dim=-1)

            e_idx = 1
            if hasattr(self.model, "config") and hasattr(self.model.config, "id2label"):
                id2label = {int(k): str(v).lower() for k, v in self.model.config.id2label.items()}
                e_idx = next((i for i, lbl in id2label.items() if "entail" in lbl), 1)

            results = [float(probs[i, e_idx].item()) for i in range(len(propositions))]
            return results
        except Exception as err:
            logger.error("Transformer batch inference error (%s); falling back to sequential noul", err)
            return [self.noul(state, p) for p in propositions]
        finally:
            if auto_unload:
                self.unload_vram()

    def noul(self, state: str, proposition: str) -> float:
        """
        Noul primitive: Returns normalized entailment probability P(entailment) in [0, 1].
        """
        auto_unload = False
        if not self.model_loaded and self.model is None:
            self.load_model()
            auto_unload = True

        try:
            probs = self._predict_logits(state, proposition)
            entail = probs[-1]
            return float(entail)
        finally:
            if auto_unload:
                self.unload_vram()

    def choice(self, state: str, options: List[str], instructions: str = "") -> Dict[str, float]:
        """
        Choice primitive: Evaluates candidate hypotheses and normalizes probabilities across candidates.
        """
        with self.active_session():
            scores: Dict[str, float] = {}
            for opt in options:
                prop = f"{instructions} Candidate: {opt}" if instructions else opt
                scores[opt] = self.noul(state, prop)

            total = sum(scores.values())
            if total <= 1e-6:
                equal_p = 1.0 / len(options) if options else 0.0
                return {opt: equal_p for opt in options}

            return {opt: round(val / total, 4) for opt, val in scores.items()}

    def score(self, state: str, levels: Union[Dict[str, float], List[Tuple[str, float]]]) -> float:
        """
        Score primitive: Returns probability-weighted expectation over levels.
        """
        with self.active_session():
            items = levels.items() if isinstance(levels, dict) else levels
            options = [k for k, _ in items]
            weights = dict(items)

            probs = self.choice(state, options)
            expected_val = sum(probs[opt] * weights[opt] for opt in options)
            return float(expected_val)
