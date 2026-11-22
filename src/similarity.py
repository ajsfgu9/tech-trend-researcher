"""Real similarity engine: Gemini Embedding API (with SQLite cache) + real corpus TF-IDF / BM25.

Never masquerades character n-gram heuristics as dense neural embeddings:
every similarity computation explicitly declares whether it used
`gemini-embedding-001` (or configured model) or `tfidf_bm25`.
"""

from __future__ import annotations

import logging
import math
import os
import re
from collections import Counter

import numpy as np
import requests

from src.http_client import build_http_session
from src.state_store import StateStore

log = logging.getLogger(__name__)

STOPWORDS = frozenset(
    {
        "the", "a", "an", "and", "or", "for", "of", "in", "on", "to", "with", "by",
        "is", "are", "was", "were", "be", "been", "being", "as", "at", "from",
        "that", "this", "these", "those", "it", "its", "into", "over", "after",
        "under", "about", "more", "than", "how", "why", "what", "when", "where",
        "who", "new", "using", "used", "via", "can", "will", "not", "but", "all",
        "out", "has", "have", "had", "their", "they", "our", "your", "you", "we",
        "http", "https", "www", "com", "org", "github", "arxiv", "paper", "show",
    }
)

TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9\-\.\+/]{1,}")


def tokenize_technical(text: str) -> list[str]:
    """Tokenize technical prose while preserving domain identifiers (hbm4, 2nm, cowos-l, fp4, c++)."""
    raw_tokens = TOKEN_RE.findall((text or "").lower())
    cleaned: list[str] = []
    for tok in raw_tokens:
        t = tok.strip(".-+/")
        if len(t) < 2 or t in STOPWORDS or t.isdigit():
            continue
        if len(t) > 4 and t.endswith("s") and not t.endswith(("ss", "us", "is", "os", "ics")):
            t = t[:-1]
        cleaned.append(t)
    return cleaned


def compute_tfidf_matrix(texts: list[str]) -> tuple[np.ndarray, dict[str, float], list[set[str]]]:
    """Build real L2-normalized corpus TF-IDF vectors and IDF dictionary."""
    docs_tokens = [tokenize_technical(t) for t in texts]
    token_sets = [set(toks) for toks in docs_tokens]
    n_docs = len(docs_tokens)
    if n_docs == 0:
        return np.zeros((0, 0), dtype=np.float32), {}, []

    df: Counter[str] = Counter()
    for tset in token_sets:
        for tok in tset:
            df[tok] += 1

    vocab = sorted(df.keys())
    vocab_idx = {tok: idx for idx, tok in enumerate(vocab)}
    idf: dict[str, float] = {
        tok: math.log((n_docs + 1.0) / (count + 1.0)) + 1.0
        for tok, count in df.items()
    }

    mat = np.zeros((n_docs, len(vocab)), dtype=np.float32)
    for row_i, toks in enumerate(docs_tokens):
        if not toks:
            continue
        tf = Counter(toks)
        for tok, cnt in tf.items():
            col_j = vocab_idx[tok]
            mat[row_i, col_j] = (1.0 + math.log(cnt)) * idf[tok]
        norm = float(np.linalg.norm(mat[row_i]))
        if norm > 0.0:
            mat[row_i] /= norm

    return mat, idf, token_sets


def weighted_lexical_overlap(
    tokens_a: set[str],
    tokens_b: set[str],
    idf: dict[str, float],
) -> tuple[float, list[str]]:
    """Compute IDF-weighted Jaccard overlap and return highest-IDF shared terms."""
    if not tokens_a or not tokens_b:
        return 0.0, []
    shared = tokens_a & tokens_b
    if not shared:
        return 0.0, []
    union = tokens_a | tokens_b
    shared_w = sum(idf.get(t, 1.0) for t in shared)
    union_w = sum(idf.get(t, 1.0) for t in union)
    score = shared_w / union_w if union_w > 0 else 0.0
    top_shared = sorted(shared, key=lambda t: idf.get(t, 1.0), reverse=True)[:10]
    return float(score), top_shared


class EmbeddingEngine:
    """Computes real dense embeddings via Gemini API (cached in SQLite) with TF-IDF fallback."""

    def __init__(
        self,
        state_store: StateStore | None = None,
        enable_embeddings: bool | None = None,
        gemini_api_key: str | None = None,
        model_name: str | None = None,
        dimensions: int | None = None,
        session: requests.Session | None = None,
    ) -> None:
        self.state_store = state_store
        if enable_embeddings is None:
            enable_embeddings = os.getenv("ENABLE_SEMANTIC_EMBEDDINGS", "true").strip().lower() in (
                "1", "true", "yes", "on"
            )
        self.gemini_api_key = (
            gemini_api_key if gemini_api_key is not None else os.getenv("GEMINI_API_KEY", "")
        ).strip()
        self.model_name = (
            model_name or os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
        ).strip().removeprefix("models/")
        self.dimensions = int(dimensions or os.getenv("EMBEDDING_DIMENSIONS", "768"))
        self.enabled = bool(enable_embeddings and self.gemini_api_key)
        self.session = session or build_http_session()

    def _fetch_gemini_batch(self, texts: list[str]) -> list[list[float]] | None:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:batchEmbedContents"
        headers = {
            "x-goog-api-key": self.gemini_api_key,
            "Content-Type": "application/json",
        }
        req_items = []
        for text in texts:
            item: dict = {
                "model": f"models/{self.model_name}",
                "content": {"parts": [{"text": text[:4000]}]},
                "outputDimensionality": self.dimensions,
            }
            if self.model_name == "gemini-embedding-001":
                item["taskType"] = "SEMANTIC_SIMILARITY"
            req_items.append(item)

        try:
            resp = self.session.post(
                url,
                headers=headers,
                json={"requests": req_items},
                timeout=45,
            )
            if resp.status_code != 200:
                log.warning(
                    "Gemini batchEmbedContents HTTP %s: %s",
                    resp.status_code,
                    resp.text[:240],
                )
                return None

            embeddings = resp.json().get("embeddings") or []
            if len(embeddings) != len(texts):
                log.warning(
                    "Gemini batchEmbedContents count mismatch: expected %d, got %d",
                    len(texts),
                    len(embeddings),
                )
                return None

            normalized: list[list[float]] = []
            for emb in embeddings:
                vec = np.asarray(emb.get("values") or [], dtype=np.float32)
                norm = float(np.linalg.norm(vec))
                if norm > 0.0:
                    vec = vec / norm
                normalized.append(vec.tolist())
            return normalized
        except Exception as exc:
            log.warning("Gemini batchEmbedContents failed: %s", exc)
            return None

    def embed_texts(self, texts: list[str]) -> tuple[np.ndarray | None, str]:
        """Return (L2-normalized embedding matrix, backend_name) or (None, 'tfidf_bm25')."""
        if not texts or not self.enabled:
            return None, "tfidf_bm25"

        keys = [
            StateStore.make_embedding_cache_key(self.model_name, self.dimensions, t[:4000])
            for t in texts
        ]
        cached = self.state_store.get_cached_embeddings(keys) if self.state_store else {}

        missing_indices = [i for i, k in enumerate(keys) if k not in cached]
        newly_computed: dict[str, list[float]] = {}

        # Gemini batchEmbedContents supports up to 100 requests per call
        batch_size = 50
        for start in range(0, len(missing_indices), batch_size):
            idx_chunk = missing_indices[start : start + batch_size]
            text_chunk = [texts[i] for i in idx_chunk]
            vecs = self._fetch_gemini_batch(text_chunk)
            if vecs is None:
                log.warning("Falling back to corpus TF-IDF similarity for this run.")
                return None, "tfidf_bm25"
            for idx, vec in zip(idx_chunk, vecs):
                key = keys[idx]
                cached[key] = vec
                newly_computed[key] = vec

        if newly_computed and self.state_store:
            self.state_store.save_cached_embeddings(self.model_name, newly_computed)

        matrix = np.asarray([cached[k] for k in keys], dtype=np.float32)
        return matrix, self.model_name
