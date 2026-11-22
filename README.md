# Tech Trend Researcher

Multi-database trend researcher and cross-matcher built to pair with your three intelligence bots:

1. **`jijiji88888/ai-hardware-bot`** (`agi_hardware_trends.db` / `agi_hardware_trends2.db`)
2. **`ajsfgu9/frontier-deeptech-bot`** (`frontier_deeptech_trends.db`)
3. **`ajsfgu9/ai-software-stack-quantum-bot`** (`ai_stack_quantum_trends.db`)

## Architecture (Zero Circular Leakage)

1. **Strictly External Live Trend Collectors (`src/external_collectors.py`)**:
   - Collects **only external** live developer/tech momentum — never mixes your internal bot databases into trend discovery:
     - **Hacker News (Algolia `search_by_date`)**: Time-windowed (`created_at_i`) fresh stories with points/hour and comments/hour velocity.
     - **GitHub Velocity (`search/repositories`)**: Recently created high-traction repos (`created:>14d`) with snapshot-to-snapshot star deltas tracked in SQLite (supports optional `GITHUB_TOKEN`).
     - **Hugging Face Daily Papers (`/api/daily_papers`)**: Live AI/ML/systems paper upvotes and discussions.
     - **Lobste.rs (`/hottest.json`)**: High-signal engineering and systems discussions.
     - **Optional Authenticated Social Collectors**: Bluesky (PDS auth, labeled `bluesky`), Reddit (OAuth `client_credentials`), and X API v2 (`X_BEARER_TOKEN`). Disabled cleanly when credentials are unset — zero silent HTTP 403 failures or mislabeled platforms.
2. **Read-Only Multi-DB Bridge (`src/db_bridge.py`)**:
   - Reads all 3 bot SQLite databases across schema variants (`hardware_domain`/`stack_layer`, `ai_revolution_thesis`/`ai_stack_thesis`).
   - Works directly with the 3 bots' existing **Docker named volumes** mounted `:ro` — you do **not** need to convert the 3 bots to bind mounts. When a `:ro` mount or cross-UID permission blocks SQLite WAL `-shm` writes, `db_bridge.py` automatically snapshots `.db` + `.db-wal` into a private `/tmp` directory with retry logic so the bot volumes are never touched.
3. **Honest Similarity & Dynamic Clustering (`src/similarity.py`, `src/trend_detector.py`)**:
   - Uses **Gemini Embedding API** (`gemini-embedding-001`, cached in SQLite) when `ENABLE_SEMANTIC_EMBEDDINGS=true` and `GEMINI_API_KEY` is set, with transparent fallback to real corpus **TF-IDF + BM25** (reported explicitly via `similarity_backend`).
   - Clusters external signals dynamically by cosine similarity (no hardcoded 10-bucket semiconductor taxonomy).
   - Tracks real append-only engagement snapshots in SQLite (`signal_snapshots`) and only computes z-scores once a platform has real baseline history (`>= 10` observations).
4. **Dual-Provider LLM Verification & Hooks (`src/llm_client.py`, `src/matcher.py`)**:
   - Matches the exact `.env` contract of all 3 bots: `LLM_PROVIDER` (`openrouter` | `gemini` | `none`), `LLM_FALLBACK_PROVIDER`, `OPENROUTER_MODEL`, and `GEMINI_MODEL=gemini-3.5-flash-lite` using `x-goog-api-key` header auth (never URL query params).
   - Cross-compares your 3 bots' DB items strictly against external trends (`Output 1` -> `Output 2`) and uses the LLM to verify matches and generate domain-appropriate angles/hooks.
5. **Resilient Daemon & HTML-Escaped Telegram Delivery (`src/autonomous_daemon.py`, `src/telegram_notifier.py`)**:
   - Watches all 3 bot DBs via lightweight `os.stat()` pre-checks before reading SQLite watermarks, isolates exceptions per cycle, handles `SIGTERM`/`SIGINT`, and sends HTML-escaped Telegram digests + JSON/Markdown attachments.

## Docker Deployment (Alongside Your 3 Existing Bots)

You do **not** need to modify your 3 bots to use bind mounts.

1. Find the exact Docker named volume names created by your 3 bots:
   ```bash
   docker volume ls | grep bot-data
   ```
2. Copy `.env.example` to `.env`, set `HARDWARE_BOT_VOLUME`, `DEEPTECH_BOT_VOLUME`, `SOFTWARE_BOT_VOLUME`, your LLM keys (`OPENROUTER_API_KEY` and/or `GEMINI_API_KEY`), and your Telegram credentials:
   ```bash
   cp .env.example .env
   nano .env
   ```
3. Build and start:
   ```bash
   docker compose up -d --build
   docker compose logs -f
   ```

## Local Testing

```bash
pip install -r requirements.txt
pytest -v
python -m src.researcher --once --no-llm
```
