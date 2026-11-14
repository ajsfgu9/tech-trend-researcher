# Tech Trend Researcher (4-Stage Streaming Architecture)

A four-stage semiconductor and deep-tech trend discovery pipeline paired with [`jijiji88888/ai-hardware-bot`](https://github.com/jijiji88888/ai-hardware-bot), modeling virality strictly as a **velocity trigger** ($V_i$) filtered through a deterministic relevance gate and dynamic LLM scoring engine ($\Phi_{\text{LLM}}$).

## Architecture

1. **Stage 1 — Immutable Raw Layer (`src/raw_layer.py`)**: Ingests from `ai-hardware-bot` (`agi_hardware_trends.db` in read-only WAL mode), HN Algolia, Reddit (`r/semiconductors`, `r/hardware`, `r/LocalLLaMA`, `r/biotech`, `r/robotics`, `r/QuantumComputing`), Technical RSS, and arXiv, persisting untouched JSON payloads.
2. **Stage 2 — Staging & Two-Tier Relevance Gate (`src/relevance_gate.py`)**: Executes Tier 1 lexical whitelist + cosine similarity ($\text{Sim} \ge 0.62$) and Tier 2 NER + consumer-noise filtering.
3. **Stage 3 — Deterministic Transformation (`src/scoring_engine.py`)**: Computes $M_i = W_s \times R_i \times (1 + \ln(1 + V_i)) \times D_i$ using rolling baselines from the SQLite **Metadata Repository (`src/metadata_repo.py`)**.
4. **Stage 4 — Semantic Presentation Layer (`src/llm_auditor.py`, `src/clustering.py`)**: Evaluates Technical Substance ($S$), Industry Impact ($I$), and Lifecycle Phase ($L$) via $\Phi_{\text{LLM}} = (0.4S + 0.6I) \times L$, clusters related events, and outputs the structured JSON briefing (`trends_output.json`) and Markdown dashboard (`dashboard.md`).

## Quickstart

```bash
cp .env.example .env
pip install -r requirements.txt
python -m src.researcher --no-llm
```
