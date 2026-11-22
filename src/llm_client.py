"""Dual-provider LLM client (OpenRouter + Official Gemini API with automatic failover).

Mirrors the exact `.env` contract and hardened calling semantics of:
- jijiji88888/ai-hardware-bot
- ajsfgu9/frontier-deeptech-bot
- ajsfgu9/ai-software-stack-quantum-bot

Security & reliability:
- Sends Gemini credentials via `x-goog-api-key` header (never `?key=` in URLs).
- Filters Gemini `thought: true` parts before JSON parsing.
- Supports configurable `OPENROUTER_MODEL` and `GEMINI_MODEL` (default `gemini-3.5-flash-lite`).
- Retries transient 429/5xx failures with exponential backoff and fails over to
  `LLM_FALLBACK_PROVIDER` if configured.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

import requests

from src.http_client import build_http_session
from src.models import BotDBItem, ExternalTrendCluster

log = logging.getLogger(__name__)

SUPPORTED_LLM_PROVIDERS = {"openrouter", "gemini", "none"}


def normalize_provider(name: str | None) -> str:
    val = (name or "").strip().lower()
    if val in {"google", "gemini-official", "google-gemini"}:
        return "gemini"
    return val


def parse_llm_json(text: str, provider_name: str = "LLM") -> dict[str, Any]:
    if not text or not text.strip():
        raise ValueError(f"{provider_name} returned empty text.")
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        parsed = json.loads(cleaned[start : end + 1])

    if not isinstance(parsed, dict):
        raise ValueError(f"{provider_name} JSON response was not an object.")
    return parsed


class DualLLMClient:
    """OpenRouter + Official Gemini API JSON client with configurable failover."""

    def __init__(
        self,
        provider: str | None = None,
        fallback_provider: str | None = None,
        openrouter_api_key: str | None = None,
        openrouter_model: str | None = None,
        gemini_api_key: str | None = None,
        gemini_model: str | None = None,
        session: requests.Session | None = None,
        max_retries: int = 3,
    ) -> None:
        raw_primary = provider if provider is not None else os.getenv("LLM_PROVIDER", "openrouter")
        raw_fallback = (
            fallback_provider
            if fallback_provider is not None
            else os.getenv("LLM_FALLBACK_PROVIDER", "")
        )
        self.primary = normalize_provider(raw_primary) or "none"
        if self.primary not in SUPPORTED_LLM_PROVIDERS:
            log.warning("Unsupported LLM_PROVIDER=%r; defaulting to 'none'.", raw_primary)
            self.primary = "none"

        self.fallback = normalize_provider(raw_fallback)
        if self.fallback and (self.fallback not in {"openrouter", "gemini"} or self.fallback == self.primary):
            self.fallback = ""

        self.openrouter_api_key = (
            openrouter_api_key if openrouter_api_key is not None else os.getenv("OPENROUTER_API_KEY", "")
        ).strip()
        self.openrouter_model = (
            openrouter_model or os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-v4-flash")
        ).strip()
        self.gemini_api_key = (
            gemini_api_key if gemini_api_key is not None else os.getenv("GEMINI_API_KEY", "")
        ).strip()
        self.gemini_model = (
            gemini_model or os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
        ).strip().removeprefix("models/")

        self.session = session or build_http_session(total_retries=1)
        self.max_retries = max(1, int(max_retries))

    def provider_chain(self) -> list[str]:
        if self.primary == "none":
            return []
        chain = [self.primary]
        if self.fallback:
            chain.append(self.fallback)
        return chain

    def active_label(self) -> str:
        chain = self.provider_chain()
        if not chain:
            return "none (deterministic)"
        labels = [
            f"Gemini ({self.gemini_model})" if p == "gemini" else f"OpenRouter ({self.openrouter_model})"
            for p in chain
        ]
        return " -> fallback ".join(labels)

    def _call_openrouter(self, system_prompt: str, user_prompt: str) -> dict[str, Any] | None:
        if not self.openrouter_api_key:
            log.warning("Missing OPENROUTER_API_KEY; skipping OpenRouter call.")
            return None

        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.openrouter_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.openrouter_model,
            "messages": [
                {"role": "system", "content": system_prompt.strip()},
                {"role": "user", "content": user_prompt.strip()},
            ],
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
        }

        for attempt in range(self.max_retries):
            try:
                resp = self.session.post(url, headers=headers, json=payload, timeout=60)
                if resp.status_code == 200:
                    text = resp.json().get("choices", [{}])[0].get("message", {}).get("content", "")
                    return parse_llm_json(text, provider_name="OpenRouter")
                if resp.status_code in (429, 500, 502, 503, 504) and attempt < self.max_retries - 1:
                    wait_s = 2 ** (attempt + 1)
                    log.warning("OpenRouter HTTP %s; retrying in %ss.", resp.status_code, wait_s)
                    time.sleep(wait_s)
                    continue
                log.error("OpenRouter HTTP %s: %s", resp.status_code, resp.text[:300])
                return None
            except Exception as exc:
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** (attempt + 1))
                    continue
                log.warning("OpenRouter exception: %s", exc)
                return None
        return None

    def _call_gemini(self, system_prompt: str, user_prompt: str) -> dict[str, Any] | None:
        if not self.gemini_api_key:
            log.warning("Missing GEMINI_API_KEY; skipping Gemini call.")
            return None

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.gemini_model}:generateContent"
        headers = {
            "x-goog-api-key": self.gemini_api_key,
            "Content-Type": "application/json",
        }
        payload = {
            "systemInstruction": {"parts": [{"text": system_prompt.strip()}]},
            "contents": [{"role": "user", "parts": [{"text": user_prompt.strip()}]}],
            "generationConfig": {
                "temperature": 0.1,
                "responseMimeType": "application/json",
            },
        }

        for attempt in range(self.max_retries):
            try:
                resp = self.session.post(url, headers=headers, json=payload, timeout=60)
                if resp.status_code == 200:
                    data = resp.json()
                    cands = data.get("candidates") or []
                    parts = (cands[0].get("content") or {}).get("parts") or [] if cands else []
                    non_thought = [
                        str(p.get("text") or "")
                        for p in parts
                        if isinstance(p, dict) and p.get("text") and not p.get("thought")
                    ]
                    text = "".join(non_thought) or "".join(
                        str(p.get("text") or "") for p in parts if isinstance(p, dict)
                    )
                    return parse_llm_json(text, provider_name="Gemini")
                if resp.status_code in (429, 500, 502, 503, 504) and attempt < self.max_retries - 1:
                    wait_s = 2 ** (attempt + 1)
                    log.warning("Gemini HTTP %s; retrying in %ss.", resp.status_code, wait_s)
                    time.sleep(wait_s)
                    continue
                log.error("Gemini HTTP %s: %s", resp.status_code, resp.text[:300])
                return None
            except Exception as exc:
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** (attempt + 1))
                    continue
                log.warning("Gemini exception: %s", exc)
                return None
        return None

    def generate_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any] | None:
        chain = self.provider_chain()
        for idx, prov in enumerate(chain):
            if idx > 0:
                log.warning("Primary LLM failed; failing over to %s.", prov)
            result = (
                self._call_gemini(system_prompt, user_prompt)
                if prov == "gemini"
                else self._call_openrouter(system_prompt, user_prompt)
            )
            if result is not None:
                return result
        return None

    def analyze_external_trend(self, cluster: ExternalTrendCluster) -> dict[str, Any]:
        """Synthesize why an external trend cluster is gaining traction right now."""
        if not self.provider_chain():
            return {
                "mode": "deterministic",
                "summary": (
                    f"Active across {', '.join(cluster.platforms)} ({len(cluster.signals)} signal(s), "
                    f"peak rate {cluster.max_engagement_rate_per_hour:.1f}/hr)."
                ),
                "core_debate": "",
                "editorial_priority": "HIGH" if cluster.trend_score >= 65 else ("MEDIUM" if cluster.trend_score >= 40 else "LOW"),
            }

        items_text = "\n".join(
            f"- [{s.platform} | {s.engagement_rate_per_hour:.1f}/hr] {s.title} ({s.url})"
            for s in cluster.signals[:6]
        )
        system_prompt = (
            "You are a concise technical intelligence editor covering AI hardware, AI software stacks, "
            "and frontier deep-tech. Return strict JSON only."
        )
        user_prompt = (
            f"Analyze this live external trend cluster:\n"
            f"Lead Label: {cluster.label}\n"
            f"Platforms: {cluster.platforms}\n"
            f"Top Terms: {cluster.top_terms}\n"
            f"Signals:\n{items_text}\n\n"
            "Return JSON with exact keys:\n"
            '  "summary": "1-2 factual sentences explaining what is actually happening",\n'
            '  "core_debate": "1 sentence on the main engineering trade-off or controversy",\n'
            '  "editorial_priority": "HIGH" | "MEDIUM" | "LOW"'
        )
        parsed = self.generate_json(system_prompt, user_prompt)
        if not parsed:
            return {
                "mode": "fallback_deterministic",
                "summary": f"Active across {', '.join(cluster.platforms)} ({len(cluster.signals)} signal(s)).",
                "core_debate": "",
                "editorial_priority": "MEDIUM",
            }
        parsed["mode"] = "llm"
        return parsed

    def verify_and_angle_db_match(
        self,
        cluster: ExternalTrendCluster,
        db_item: BotDBItem,
    ) -> dict[str, Any]:
        """Verify whether a DB item truly connects to a live external trend and draft a grounded X hook."""
        if not self.provider_chain():
            return {
                "is_true_match": None,
                "rationale": "",
                "x_hook": "",
                "reply_prompt": "",
                "visual_idea": "",
            }

        system_prompt = (
            "You are a skeptical senior tech editor. Decide whether an archived database item genuinely "
            "connects to a live external trend (same technology, bottleneck, company, or mechanism — NOT "
            "superficial keyword coincidence). If true, write a sharp, domain-accurate X/Twitter opening "
            "hook without generic hype or irrelevant analogies. Return strict JSON only."
        )
        user_prompt = (
            "LIVE EXTERNAL TREND:\n"
            f"- Label: {cluster.label}\n"
            f"- Platforms: {cluster.platforms}\n"
            f"- Top Terms: {cluster.top_terms}\n"
            f"- Sample Titles: {[s.title for s in cluster.signals[:4]]}\n\n"
            "ARCHIVED BOT DATABASE ITEM:\n"
            f"- Bot Source DB: {db_item.bot_name}\n"
            f"- Title: {db_item.title}\n"
            f"- Domain/Layer: {db_item.domain}\n"
            f"- Bottleneck: {db_item.bottleneck_addressed}\n"
            f"- Specific Tech: {db_item.specific_tech}\n"
            f"- Why It Matters: {db_item.why_it_matters}\n"
            f"- Thesis: {db_item.thesis}\n\n"
            "Return JSON with exact keys:\n"
            '  "is_true_match": true or false,\n'
            '  "rationale": "1 sentence explaining the exact causal or technical link (or why it is a false positive)",\n'
            '  "x_hook": "If true, a grounded <=240 char opening post linking the live trend to this DB insight (no external URL in root post); empty string if false",\n'
            '  "reply_prompt": "If true, 1 concrete technical question for replies; empty string if false",\n'
            '  "visual_idea": "If true, 1 specific chart/table/diagram idea; empty string if false"'
        )
        parsed = self.generate_json(system_prompt, user_prompt)
        if not parsed:
            return {
                "is_true_match": None,
                "rationale": "LLM verification unavailable; kept based on similarity threshold.",
                "x_hook": "",
                "reply_prompt": "",
                "visual_idea": "",
            }
        return {
            "is_true_match": bool(parsed.get("is_true_match", False)),
            "rationale": str(parsed.get("rationale") or "").strip(),
            "x_hook": str(parsed.get("x_hook") or "").strip(),
            "reply_prompt": str(parsed.get("reply_prompt") or "").strip(),
            "visual_idea": str(parsed.get("visual_idea") or "").strip(),
        }
