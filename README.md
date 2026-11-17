# Tech Trend Researcher (Longitudinal Social Trend Engine & Database-to-X Breakout Catalyst)

## Distinct Roles Across Both Repositories

| Repository | Distinct Role | Core Responsibilities |
| --- | --- | --- |
| **[`jijiji88888/ai-hardware-bot`](https://github.com/jijiji88888/ai-hardware-bot)** ([PR #4](https://github.com/jijiji88888/ai-hardware-bot/pull/4)) | **Broad Internet Fetcher & LLM Relevance Filter** | Fetches candidate articles, filings, and preprints from the broad internet (`arXiv`, `SEC EDGAR EFTS`, `Semiconductor Engineering`, `ServeTheHome`, `DIGITIMES`, `TrendForce`, `Techmeme`, `Hacker News`) and uses its OpenRouter LLM classifier (`classify_batch`) to filter relevant items and extract `hardware_domain`, `bottleneck_addressed`, `specific_tech`, `relevance_score`, `why_it_matters`, and `ai_revolution_thesis` into `agi_hardware_trends.db`. |
| **[`ajsfgu9/tech-trend-researcher`](https://github.com/ajsfgu9/tech-trend-researcher)** | **X/Twitter Trend Researcher & Database-to-X Viral Catalyst** | Monitors broad social, developer, and microblog channels (`Reddit`, `Hacker News`, `GitHub`, `X/Bluesky pulse`), detects longitudinal trends via EWMA/CUSUM burst & causal propagation (`src/trend_analyzer.py`), scores X/Twitter breakout potential via `src/x_breakout_engine.py`, and cross-compares active trends against `agi_hardware_trends.db` (`src/viral_matcher.py`). |

## How Database Discoveries Become Real Trends on X/Twitter (`src/x_breakout_engine.py`)

A deep engineering discovery inside `agi_hardware_trends.db` (such as a finite-element paper on `3.5x reticle CoWoS warpage` or an `SEC 8-K` on substrate capex) will stay buried in the database unless it is bridged into the mechanics of **X's For You recommendation algorithm** (`xai-org/x-algorithm` Phoenix Transformer + `home-mixer/params/param.rs`):

1. **X Pre-Saturation Arbitrage (`FIRST_MOVER_WINDOW`)**:
   - Detects topics that are accelerating in `Primary Engineering` (`arXiv`/`GitHub`/`SEC`) and `Debate` (`Reddit`/`HN`) while X mentions are still low (`x_arbitrage_window == "FIRST_MOVER_WINDOW"`), giving the downstream News Writer first-mover advantage on X before mainstream tech accounts saturate the topic.
2. **Phoenix Action-Weight Optimization (`x_breakout_score`)**:
   - Scores every trend and matching database item against X's highest-weighted positive actions:
     - `share_via_copy_link` (`+20.0`) & `share_via_dm` (`+5.0`): Driven by hard extracted quantitative metrics (`nm`, `pJ/bit`, `reticle`, `$B`, `Phase 2`, `qubits`).
     - `reply` (`+5.0`) & `quote` (`+5.0`): Driven by contrarian framing (debunking a mainstream narrative) and mapping hardware bottlenecks onto high-velocity X cashtags (`$NVDA`, `$TSM`, `$ASML`, `$AMD`, `$MU`, `$AVGO`, `$INTC`).
     - `cont_click_dwell_time` (`+0.40`): Driven by a 2-column spec/unit-economics visual card optimized for Phoenix v5's 1024-d multimodal image embeddings.
     - `link_placement_rule`: Enforces `"first_reply_only"` so external URLs never suppress root-post reach.
3. **Database-to-X Catalyst Playbook (`x_catalyst_playbook`)**:
   - Every matched database item in `output_2_database_viral_matches` includes a ready-to-post `hook_tweet_draft` (bridging the obscure hardware bottleneck to high-velocity X cashtags), `debate_reply_bait`, and `multimodal_visual_card` specification.
