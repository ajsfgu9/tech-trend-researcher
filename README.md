# Tech Trend Researcher

A multi-vertical tech trend research engine built to pair with [`jijiji88888/ai-hardware-bot`](https://github.com/jijiji88888/ai-hardware-bot). It discovers, verifies, and ranks emerging tech trends across **semiconductors**, **financing**, **biomedical**, **robotics**, and **quantum**, then scores each trend against reverse-engineered **X (Twitter) For You** ranking mechanics so a downstream **News Writer** can craft high-signal posts.

## Architecture

1. **Aggregator Bridge (`src/sources.py`)**:
   - Safely reads `ai-hardware-bot`'s SQLite database (`agi_hardware_trends.db`) in read-only URI mode (`?mode=ro`) with WAL support, excluding noisy feeds (`Hacker News` by default).
   - Supplements with live primary & specialist RSS/Atom feeds across all 5 tech verticals (`arXiv`, `bioRxiv`, `SEC EDGAR`, `TrendForce`, `Semiconductor Engineering`, `FierceBiotech`, `IEEE Spectrum Robotics`, `Quantum Computing Report`).
2. **Fairness & Anti-Hype Engine (`src/fairness_engine.py`)**:
   - Clusters related stories across sources, extracts hard quantitative metrics (`nm`, `TFLOPS`, `pJ/bit`, `$B`, `Phase 1/2/3`, `qubits`, `DOF`, etc.), rewards primary-source corroboration, penalizes single-blog rumors with zero metrics, and enforces cross-vertical diversity quotas.
3. **Reverse-Engineered X/Twitter Algorithm Scorer (`src/x_algo_scorer.py`)**:
   - Models the open-sourced [`xai-org/x-algorithm`](https://github.com/xai-org/x-algorithm) Phoenix Transformer & `home-mixer/params/param.rs` action weights:
     - **High-effort explicit signals (~96.9% of positive weight mass)**: `share_via_copy_link` (+20.0), `reply` (+5.0), `quote` (+5.0), `share_via_dm` (+5.0), `follow_author` (+4.0), `share` (+2.0), `retweet` (+1.0).
     - **Attention & passive signals**: `favorite` (+0.66), `click` (+0.30), `open_link` (+0.20), `cont_click_dwell_time` (+0.40), `cont_dwell_time` (+0.006/s).
     - **Negative & suppression signals**: `not_interested` (-52.272), Grox spam classification, and indirect external-link suppression (`url_click_score` omitted from Phoenix's 19-action head; `has_visible_link` feature).
   - Outputs an **X Attractiveness Score (0–100)** plus concrete posting directives (`recommended_format`, `link_placement: first_reply_only`, `visual_suggestion` for Phoenix v5 multimodal embeddings, and 30–60 min author reply-velocity rules).
4. **Handoff to News Writer (`src/researcher.py`)**:
   - Persists ranked trends to `tech_trends.db` (SQLite WAL mode) and exports `trends_output.json` with source URLs, fairness caveats, and X posting hooks.

## Quickstart

```bash
cp .env.example .env
pip install -r requirements.txt
pytest
python -m src.researcher --no-llm
```

### Docker Deployment (alongside `ai-hardware-bot`)

Update the read-only volume path in `docker-compose.yml` to point to your `ai-hardware-bot` data directory, then run:

```bash
docker compose up -d --build
```
