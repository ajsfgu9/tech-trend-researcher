# Tech Trend Researcher (Longitudinal Trend Engine & Editorial Briefs)

## Strict Architectural Separation (`ai-hardware-bot` vs. `tech-trend-researcher`)

| Repository | Distinct Role | Core Responsibilities |
| --- | --- | --- |
| **[`jijiji88888/ai-hardware-bot`](https://github.com/jijiji88888/ai-hardware-bot)** ([PR #3](https://github.com/jijiji88888/ai-hardware-bot/pull/3)) | **Multi-Source Ingestion & Time-Series Storage** | Polls/scrapes all external RSS/Atom/API feeds, deduplicates URLs/fingerprints, and persists raw items + engagement observations into `agi_hardware_trends.db` (SQLite WAL mode). |
| **[`ajsfgu9/tech-trend-researcher`](https://github.com/ajsfgu9/tech-trend-researcher)** | **Longitudinal Trend Discovery & Editorial Intelligence** | Performs **zero** duplicate web scraping. Reads `agi_hardware_trends.db` (`?mode=ro`) over 7d–30d windows to detect structural trends and generate ready-to-draft **Automated Content Briefs** (`content_briefs.json`). |

## How "Real Trends" Are Studied (`src/trend_analyzer.py`)

Instead of relying on static point-in-time item scores or simple social cross-matching, `tech-trend-researcher` models a trend as a **longitudinal trajectory over time** across four quantitative pillars (with item-level scoring reserved strictly as a **20% sub-component**):

1. **Temporal Burst & Second-Order Acceleration (EWMA + Page-Hinkley / CUSUM)**:
   - Aggregates candidate items into daily/windowed buckets per `(Bottleneck Concept, Entity)` in `concept_timeseries`.
   - Computes Exponentially Weighted Moving Average ($z_{\text{EWMA}}$), second-order velocity acceleration ($\Delta v = v_t - v_{t-1}$), and the cumulative sum ($\text{CUSUM}_t$) regime-shift statistic to classify trajectories into `breakout`, `accelerating`, `steady`, or `decaying`.
2. **Cross-Layer Causal Propagation Graph (Lead-Lag S-Curve Diffusion)**:
   - Tracks the chronological arrival order of a bottleneck across source layers:
     `Primary Engineering (Stage 0: arXiv/GitHub/Patents)` $\rightarrow$ `Debate & Controversy (Stage 1: HN/Reddit)` $\rightarrow$ `Supply Chain & Financial (Stage 2: SemiAnalysis/TrendForce/SEC)` $\rightarrow$ `Mainstream Media (Stage 3)`.
   - Rewards trajectories originating in Stage 0/1 that propagate into Stage 2 *before* Stage 3 mainstream saturation.
3. **Novel Bottleneck Co-Occurrence Shift**:
   - Compares current `(Entity, Bottleneck)` coupling frequency against historical co-occurrence baselines in `entity_graph` to surface newly emerging constraints (e.g., `TSMC` + `3.5x reticle interposer warpage`).
4. **Reserved Item Sub-Score (20% Weight) & Executive Editor Framing (`src/scoring_engine.py`, `src/llm_auditor.py`)**:
   - Retains the item-level viral/information-asymmetry score ($M_{\text{viral}}$) as a 20% tie-breaker alongside the 80% Longitudinal Structural Score, then applies the Executive Editor prompt ($\Phi_{\text{editorial}}$) to generate `content_briefs.json` and `dashboard.md`.
