# Tech Trend Researcher (Longitudinal Social Trend Engine & Database Viral Matcher)

## Distinct Roles Across Both Repositories

| Repository | Distinct Role | Core Responsibilities |
| --- | --- | --- |
| **[`jijiji88888/ai-hardware-bot`](https://github.com/jijiji88888/ai-hardware-bot)** ([PR #4](https://github.com/jijiji88888/ai-hardware-bot/pull/4)) | **Broad Internet Fetcher & LLM Relevance Filter** | Fetches candidate articles, filings, and preprints from the broad internet (`arXiv`, `SEC EDGAR EFTS`, `Semiconductor Engineering`, `ServeTheHome`, `DIGITIMES`, `TrendForce`, `Techmeme`, `Hacker News`) and uses its OpenRouter LLM classifier (`classify_batch`) to filter relevant items and extract `hardware_domain`, `bottleneck_addressed`, `specific_tech`, `relevance_score`, `why_it_matters`, and `ai_revolution_thesis` into `agi_hardware_trends.db`. |
| **[`ajsfgu9/tech-trend-researcher`](https://github.com/ajsfgu9/tech-trend-researcher)** | **Social Trend Researcher & Viral Feature Matcher** | Monitors broad social & developer channels (`Reddit`, `Hacker News`, `GitHub`), detects longitudinal trends via EWMA/CUSUM burst & causal propagation (`src/trend_analyzer.py`), and cross-compares active trends against the previously sorted items in `agi_hardware_trends.db` (`src/viral_matcher.py`). |

## Two-Part Final Output (`content_briefs.json` & `dashboard.md`)

1. **`output_1_trending_now`**:
   - Ranked longitudinal trends discovered across the pipeline, including EWMA velocity $z$-score, second-order acceleration $\Delta v$, CUSUM regime state (`breakout`, `accelerating`, `steady`), cross-layer causal propagation chain, and the Executive Editor's narrative framework (`the_hook`, `the_misconception`, `the_engineering_truth`, `the_conflict`).
2. **`output_2_database_viral_matches` (`src/viral_matcher.py`)**:
   - Cross-compares the viral feature signature of every active trend in Output 1 against the previously sorted items in `agi_hardware_trends.db` (`bottleneck_addressed`, `specific_tech`, `hardware_domain`, entities, and LLM `relevance_score`).
   - Outputs `"has_viral_db_matches": "YES"` with the exact list of matching database items (`matched_database_items`) and their shared viral features if any match is found, or `"has_viral_db_matches": "NO"` if none match.
