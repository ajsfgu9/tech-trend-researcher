# Tech Trend Researcher (Editorial Viability & Automated Content Briefs)

An editorial-grade tech & semiconductor trend discovery pipeline paired with [`jijiji88888/ai-hardware-bot`](https://github.com/jijiji88888/ai-hardware-bot). Instead of chasing raw social virality, it filters for **editorial viability**, **information asymmetry**, and **narrative tension** to produce ready-to-draft **Automated Content Briefs** (`content_briefs.json`) and an **Editorial Decision Matrix** (`dashboard.md`).

## Editorial Architecture

1. **Editorial Source Re-Weighting (`W_e` in `src/models.py`)**:
   - `primary_engineering` (GitHub, arXiv, bioRxiv, patents): `0.95`
   - `debate_controversy` (Hacker News, Reddit `r/hardware`, `r/semiconductors`): `0.90`
   - `financial_supply_chain` (SEC EDGAR, DigiTimes, TrendForce, SemiAnalysis): `0.85`
   - `viral_amplification` (X/Twitter tech lists, Bluesky, Mastodon): `0.75`
   - `mainstream_tech_media` (The Verge, TechCrunch, Tom's Hardware, Ars Technica): `0.40`
2. **Viral Momentum & Information Asymmetry (`src/relevance_gate.py`, `src/scoring_engine.py`)**:
   - Comments weighted **3x higher than likes** to capture active technical debate.
   - Computes Information Asymmetry (`IA`) from specialized hardware/deep-tech token density and cross-platform controversy signals.
3. **Executive Editor LLM Evaluator (`src/llm_auditor.py`)**:
   - Scores Narrative Tension (`T`), Counter-Intuitive Factor (`C`), and Visual/Exploratory Depth (`E`):
     `Phi_editorial = (0.35 * T + 0.40 * C + 0.25 * E) * Hook_Multiplier`
4. **Automated Content Briefs & Decision Matrix (`src/clustering.py`)**:
   - Outputs `content_briefs.json` (headline recommendation, `narrative_framework`, `story_angles`, `anchor_visual`, `content_structure`, `recommended_formats`) and `dashboard.md` mapped to the 4-tier Editorial Decision Matrix (`85-100` Lead Story, `70-84` Fast-Turnaround Thread, `50-69` Newsletter Segment, `<50` Archive).

## Quickstart

```bash
cp .env.example .env
pip install -r requirements.txt
python -m src.researcher --no-llm
```
