"""Cross-matcher comparing strictly external live trends against the 3 upstream bot databases.

Zero circular leakage:
- If there are no external trend clusters (Output 1 is empty), Output 2 strictly
  returns `has_viral_db_matches: "NO"` with 0 matches.
- Requires genuine semantic + lexical overlap (no free-pass relevance bonuses).
- Optionally runs `DualLLMClient.verify_and_angle_db_match()` to reject false
  positives and produce domain-accurate X/Twitter angles.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from src.llm_client import DualLLMClient
from src.models import BotDBItem, DBViralMatch, ExternalTrendCluster
from src.similarity import EmbeddingEngine, compute_tfidf_matrix, weighted_lexical_overlap

log = logging.getLogger(__name__)


def match_db_items_to_external_trends(
    external_trends: list[ExternalTrendCluster],
    db_items: list[BotDBItem],
    embedding_engine: EmbeddingEngine | None = None,
    llm_client: DualLLMClient | None = None,
    match_threshold: float = 0.32,
    max_llm_verifications: int = 15,
) -> dict[str, Any]:
    """Cross-compare upstream bot DB items strictly against live external trend clusters."""
    if not external_trends or not db_items:
        reason = (
            "No live external trends were detected in this window."
            if not external_trends
            else "No classified items were loaded from the configured bot databases."
        )
        return {
            "has_viral_db_matches": "NO",
            "matched_count": 0,
            "similarity_backend": "none",
            "summary": f"NO — {reason}",
            "matched_database_items": [],
        }

    trend_texts = [t.semantic_text() for t in external_trends]
    db_texts = [item.semantic_text() for item in db_items]
    all_texts = trend_texts + db_texts

    tfidf_mat, idf, token_sets = compute_tfidf_matrix(all_texts)
    n_trends = len(external_trends)
    trend_tfidf = tfidf_mat[:n_trends]
    db_tfidf = tfidf_mat[n_trends:]
    trend_tokens = token_sets[:n_trends]
    db_tokens = token_sets[n_trends:]

    dense_mat: np.ndarray | None = None
    backend = "tfidf_bm25"
    if embedding_engine is not None:
        dense_mat, backend = embedding_engine.embed_texts(all_texts)

    if dense_mat is not None and dense_mat.shape[0] == len(all_texts):
        trend_dense = dense_mat[:n_trends]
        db_dense = dense_mat[n_trends:]
        semantic_sim_mat = np.clip(db_dense @ trend_dense.T, 0.0, 1.0)
        eff_threshold = max(0.62, match_threshold)
    else:
        semantic_sim_mat = (
            np.clip(db_tfidf @ trend_tfidf.T, 0.0, 1.0)
            if tfidf_mat.size
            else np.zeros((len(db_items), n_trends), dtype=np.float32)
        )
        eff_threshold = match_threshold
        backend = "tfidf_bm25"

    candidates: list[tuple[float, ExternalTrendCluster, BotDBItem, float, float, list[str]]] = []
    seen_urls: set[str] = set()

    for db_idx, item in enumerate(db_items):
        if not item.url or item.url in seen_urls:
            continue

        best_score = 0.0
        best_trend: ExternalTrendCluster | None = None
        best_sem = 0.0
        best_lex = 0.0
        best_shared: list[str] = []

        for t_idx, trend in enumerate(external_trends):
            sem_sim = float(semantic_sim_mat[db_idx, t_idx])
            lex_sim, shared_terms = weighted_lexical_overlap(
                db_tokens[db_idx],
                trend_tokens[t_idx],
                idf,
            )

            # Combine semantic similarity with lexical overlap.
            # When using dense embeddings, require either strong semantic similarity (>=0.72)
            # or moderate semantic + real shared technical terms.
            if backend != "tfidf_bm25":
                if sem_sim < 0.72 and len(shared_terms) < 2:
                    continue
                combined = 0.75 * sem_sim + 0.25 * min(1.0, lex_sim * 2.5)
            else:
                if len(shared_terms) < 2:
                    continue
                combined = 0.70 * sem_sim + 0.30 * min(1.0, lex_sim * 2.0)

            if combined > best_score:
                best_score = combined
                best_trend = trend
                best_sem = sem_sim
                best_lex = lex_sim
                best_shared = shared_terms

        if best_trend is not None and best_score >= eff_threshold:
            seen_urls.add(item.url)
            candidates.append((best_score, best_trend, item, best_sem, best_lex, best_shared))

    candidates.sort(
        key=lambda row: (row[0], row[1].trend_score, row[2].relevance_score),
        reverse=True,
    )

    verified_matches: list[DBViralMatch] = []
    llm_checks = 0
    use_llm = llm_client is not None and bool(llm_client.provider_chain())

    for combined, trend, item, sem_sim, lex_sim, shared_terms in candidates:
        llm_verified: bool | None = None
        rationale = ""
        x_hook = ""
        reply_prompt = ""
        visual_idea = ""

        if use_llm and llm_checks < max_llm_verifications:
            llm_checks += 1
            verdict = llm_client.verify_and_angle_db_match(trend, item)
            llm_verified = verdict.get("is_true_match")
            rationale = str(verdict.get("rationale") or "")
            x_hook = str(verdict.get("x_hook") or "")
            reply_prompt = str(verdict.get("reply_prompt") or "")
            visual_idea = str(verdict.get("visual_idea") or "")
            if llm_verified is False:
                log.info(
                    "LLM rejected false-positive match [%s #%d] vs '%s': %s",
                    item.bot_name,
                    item.item_id,
                    trend.label[:50],
                    rationale,
                )
                continue

        verified_matches.append(
            DBViralMatch(
                bot_name=item.bot_name,
                db_item_id=item.item_id,
                composite_id=item.composite_id,
                title=item.title,
                url=item.url,
                source=item.source,
                domain=item.domain,
                published_at=item.published_at,
                db_relevance_score=item.relevance_score,
                bottleneck_addressed=item.bottleneck_addressed,
                specific_tech=item.specific_tech,
                why_it_matters=item.why_it_matters,
                thesis=item.thesis,
                matched_cluster_id=trend.cluster_id,
                matched_trend_label=trend.label,
                matched_platforms=trend.platforms,
                matched_trend_score=trend.trend_score,
                combined_match_score=round(combined * 100.0, 1),
                semantic_similarity=round(sem_sim, 3),
                lexical_overlap=round(lex_sim, 3),
                similarity_backend=backend,
                shared_terms=shared_terms,
                llm_verified=llm_verified,
                llm_rationale=rationale,
                llm_x_hook=x_hook,
                llm_reply_prompt=reply_prompt,
                llm_visual_idea=visual_idea,
            )
        )

    if verified_matches:
        bots_hit = sorted({m.bot_name for m in verified_matches})
        return {
            "has_viral_db_matches": "YES",
            "matched_count": len(verified_matches),
            "similarity_backend": backend,
            "summary": (
                f"YES — {len(verified_matches)} database item(s) across {', '.join(bots_hit)} "
                f"matched active external trends (backend: {backend})."
            ),
            "matched_database_items": [m.to_dict() for m in verified_matches],
        }

    return {
        "has_viral_db_matches": "NO",
        "matched_count": 0,
        "similarity_backend": backend,
        "summary": (
            f"NO — None of the {len(db_items)} loaded bot DB items met the external trend match "
            f"criteria (backend: {backend})."
        ),
        "matched_database_items": [],
    }
