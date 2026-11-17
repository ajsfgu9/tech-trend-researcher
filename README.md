# Tech Trend Researcher (Autonomous Dual-Trigger X/Twitter Trend Engine & Telegram Duplicator)

## Distinct Roles Across Both Repositories

| Repository | Distinct Role | Core Responsibilities |
| --- | --- | --- |
| **[`jijiji88888/ai-hardware-bot`](https://github.com/jijiji88888/ai-hardware-bot)** ([PR #4](https://github.com/jijiji88888/ai-hardware-bot/pull/4)) | **Broad Internet Fetcher & LLM Relevance Filter** | Fetches candidate articles, filings, and preprints from the broad internet (`arXiv`, `SEC EDGAR EFTS`, `Semiconductor Engineering`, `ServeTheHome`, `DIGITIMES`, `TrendForce`, `Techmeme`, `Hacker News`) and uses its OpenRouter LLM classifier (`classify_batch`) to filter relevant items and extract `hardware_domain`, `bottleneck_addressed`, `specific_tech`, `relevance_score`, `why_it_matters`, and `ai_revolution_thesis` into `agi_hardware_trends.db`. |
| **[`ajsfgu9/tech-trend-researcher`](https://github.com/ajsfgu9/tech-trend-researcher)** | **Autonomous X/Twitter Trend Researcher, DB Viral Matcher & Telegram Duplicator** | Monitors broad social, developer, and microblog channels (`Reddit`, `Hacker News`, `GitHub`, `X/Bluesky pulse`), detects longitudinal trends (`src/trend_analyzer.py`), scores X/Twitter breakout potential (`src/x_breakout_engine.py`), cross-compares active trends against `agi_hardware_trends.db` (`src/viral_matcher.py`), and duplicates the full output to a **Telegram Bot** (`src/telegram_notifier.py`). |

## Autonomous Dual-Trigger Execution (`src/autonomous_daemon.py`)

`tech-trend-researcher` runs continuously and autonomously inside Docker without manual intervention via two independent triggers:

1. **Database-Change Trigger (`DBStateWatcher`, default poll every `15s`)**:
   - Inspects `agi_hardware_trends.db` (`MAX(id)`, `COUNT(*)`, `MAX(created_at)` on `items`) in read-only WAL mode (`?mode=ro`).
   - Persists the last processed watermark in `metadata_repo.db` (`pipeline_state` table). Whenever `ai-hardware-bot` inserts or updates sorted items, `tech-trend-researcher` immediately triggers a pipeline run (`trigger_reason="db_change:<watermark>"`).
2. **Time-Based Scheduled Trigger (`SCHEDULE_INTERVAL_MINUTES=30`)**:
   - Runs on a configurable interval even if `agi_hardware_trends.db` has not changed, ensuring emerging social/X velocity spikes are captured on time (`trigger_reason="scheduled_timer"`).
   - Use `--once` (or `DAEMON_MODE=false`) if you ever want a single CLI pass instead of the continuous daemon.

## Telegram Bot Output Duplication (`src/telegram_notifier.py`)

When `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` are set in `.env`, every pipeline run duplicates its output to your Telegram chat:
- **Formatted Two-Part Digest (`sendMessage`)**: Sends Part 1 (`What Is Trending Now & Poised to Break Out on X`) and Part 2 (`Database Viral Feature Matches: YES/NO` with ready-to-post X hooks and cashtags).
- **Full Verbatim Artifacts (`sendDocument`)**: Uploads `dashboard_<timestamp>.md` and `content_briefs_<timestamp>.json` directly to the Telegram chat.
