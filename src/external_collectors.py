"""Strictly external live developer, research, and social trend collectors.

Design guarantees:
- Never mixes upstream bot SQLite items into external trend discovery.
- Every collector queries time-bounded, velocity-relevant live endpoints.
- Non-200 HTTP errors are logged explicitly and surfaced in `CollectorStatus`.
- Authenticated social APIs (Bluesky PDS, Reddit OAuth, X API v2) run ONLY when
  valid credentials are configured and are always labeled with their real platform.
"""

from __future__ import annotations

import hashlib
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import requests

from src.http_client import build_http_session
from src.models import CollectorStatus, ExternalSignal

log = logging.getLogger(__name__)

TRACKING_PARAMS = {
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "ref", "source", "fbclid", "gclid", "mc_cid", "mc_eid",
}


def canonicalize_url(url: str) -> str:
    raw = (url or "").strip()
    if not raw:
        return ""
    parsed = urlsplit(raw)
    clean_query = [
        (k, v)
        for k, v in parse_qsl(parsed.query, keep_blank_values=False)
        if k.lower() not in TRACKING_PARAMS
    ]
    return urlunsplit(
        (
            parsed.scheme.lower(),
            parsed.netloc.lower(),
            parsed.path.rstrip("/") or "/",
            urlencode(clean_query),
            "",
        )
    )


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _parse_dt_utc(ts: str | int | float | None, fallback: datetime) -> datetime:
    if ts is None or ts == "":
        return fallback
    if isinstance(ts, (int, float)):
        return datetime.fromtimestamp(float(ts), tz=timezone.utc)
    try:
        dt = datetime.fromisoformat(str(ts).strip().replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except ValueError:
        return fallback


def _make_signal_id(platform: str, canonical_url: str) -> str:
    return hashlib.sha1(f"{platform}:{canonical_url}".encode("utf-8")).hexdigest()[:16]


def collect_hacker_news(
    session: requests.Session,
    lookback_hours: int = 48,
    min_points: int = 20,
    max_items: int = 40,
) -> tuple[list[ExternalSignal], CollectorStatus]:
    """Fetch fresh surging HN stories via Algolia `search_by_date` with created_at_i filter."""
    now = datetime.now(timezone.utc)
    now_iso = now.isoformat()
    cutoff_unix = int((now - timedelta(hours=lookback_hours)).timestamp())

    try:
        resp = session.get(
            "https://hn.algolia.com/api/v1/search_by_date",
            params={
                "tags": "story",
                "numericFilters": f"created_at_i>{cutoff_unix},points>={min_points}",
                "hitsPerPage": min(100, max(10, max_items)),
            },
            timeout=15,
        )
        if resp.status_code != 200:
            msg = f"HTTP {resp.status_code}: {resp.text[:160]}"
            log.warning("HN Algolia collector failed: %s", msg)
            return [], CollectorStatus("hacker_news", True, "error", 0, msg)

        signals: list[ExternalSignal] = []
        for hit in resp.json().get("hits", []):
            title = str(hit.get("title") or "").strip()
            raw_url = str(hit.get("url") or f"https://news.ycombinator.com/item?id={hit.get('objectID')}").strip()
            url = canonicalize_url(raw_url)
            if not title or not url:
                continue
            pub_dt = _parse_dt_utc(hit.get("created_at_i") or hit.get("created_at"), now)
            age_h = max(1.0, (now - pub_dt).total_seconds() / 3600.0)
            points = float(hit.get("points") or 0.0)
            comments = float(hit.get("num_comments") or 0.0)
            signals.append(
                ExternalSignal(
                    signal_id=_make_signal_id("hacker_news", url),
                    platform="hacker_news",
                    source_name="Hacker News",
                    url=url,
                    title=title,
                    summary=str(hit.get("story_text") or "")[:1000],
                    published_at_utc=pub_dt.isoformat(),
                    fetched_at_utc=now_iso,
                    score=points,
                    comments=comments,
                    age_hours=age_h,
                )
            )

        # Rank by real engagement rate per hour
        signals.sort(key=lambda s: (s.score + 2.0 * s.comments) / s.age_hours, reverse=True)
        signals = signals[:max_items]
        return signals, CollectorStatus("hacker_news", True, "ok", len(signals))
    except Exception as exc:
        log.warning("HN Algolia collector exception: %s", exc)
        return [], CollectorStatus("hacker_news", True, "error", 0, repr(exc))


def collect_github_velocity(
    session: requests.Session,
    lookback_days: int = 14,
    min_stars: int = 25,
    max_items: int = 30,
    github_token: str = "",
) -> tuple[list[ExternalSignal], CollectorStatus]:
    """Fetch recently created high-star GitHub repos (avoids old repos bumped by typo commits)."""
    now = datetime.now(timezone.utc)
    now_iso = now.isoformat()
    created_after = (now - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

    headers = {"Accept": "application/vnd.github+json"}
    if github_token:
        headers["Authorization"] = f"Bearer {github_token}"

    queries = (
        f"created:>{created_after} stars:>={min_stars} (llm OR inference OR cuda OR triton OR kernel OR agent)",
        f"created:>{created_after} stars:>={min_stars} (semiconductor OR riscv OR robotics OR quantum OR biotech OR photonics)",
    )

    signals: list[ExternalSignal] = []
    seen_urls: set[str] = set()
    errors: list[str] = []

    for q in queries:
        try:
            resp = session.get(
                "https://api.github.com/search/repositories",
                params={
                    "q": q,
                    "sort": "stars",
                    "order": "desc",
                    "per_page": min(30, max(10, max_items // len(queries) + 5)),
                },
                headers=headers,
                timeout=15,
            )
            if resp.status_code != 200:
                msg = f"HTTP {resp.status_code}: {resp.text[:140]}"
                log.warning("GitHub collector failed for query '%s': %s", q[:40], msg)
                errors.append(msg)
                continue

            for repo in resp.json().get("items", []):
                raw_url = str(repo.get("html_url") or "").strip()
                url = canonicalize_url(raw_url)
                name = str(repo.get("full_name") or "").strip()
                desc = str(repo.get("description") or "").strip()
                if not url or not name or url in seen_urls:
                    continue
                seen_urls.add(url)
                pub_dt = _parse_dt_utc(repo.get("created_at"), now)
                age_h = max(1.0, (now - pub_dt).total_seconds() / 3600.0)
                stars = float(repo.get("stargazers_count") or 0.0)
                forks = float(repo.get("forks_count") or 0.0)
                issues = float(repo.get("open_issues_count") or 0.0)
                topics = [str(t) for t in (repo.get("topics") or []) if t]
                lang = str(repo.get("language") or "").strip()
                if lang:
                    topics.append(lang)
                signals.append(
                    ExternalSignal(
                        signal_id=_make_signal_id("github", url),
                        platform="github",
                        source_name="GitHub Velocity",
                        url=url,
                        title=f"{name}: {desc}"[:220] if desc else name,
                        summary=desc[:1000],
                        published_at_utc=pub_dt.isoformat(),
                        fetched_at_utc=now_iso,
                        score=stars,
                        comments=issues,
                        shares=forks,
                        tags=topics[:8],
                        age_hours=age_h,
                    )
                )
        except Exception as exc:
            log.warning("GitHub collector exception: %s", exc)
            errors.append(repr(exc))

    signals.sort(key=lambda s: (s.score + 2.0 * s.shares) / s.age_hours, reverse=True)
    signals = signals[:max_items]
    status = "ok" if signals or not errors else "error"
    return signals, CollectorStatus("github", True, status, len(signals), "; ".join(errors[:2]))


def collect_hf_daily_papers(
    session: requests.Session,
    max_items: int = 30,
) -> tuple[list[ExternalSignal], CollectorStatus]:
    """Fetch live trending AI/systems papers from Hugging Face Daily Papers public API."""
    now = datetime.now(timezone.utc)
    now_iso = now.isoformat()
    try:
        resp = session.get(
            "https://huggingface.co/api/daily_papers",
            params={"limit": min(60, max(10, max_items))},
            timeout=15,
        )
        if resp.status_code != 200:
            msg = f"HTTP {resp.status_code}: {resp.text[:160]}"
            log.warning("Hugging Face Daily Papers collector failed: %s", msg)
            return [], CollectorStatus("hf_daily_papers", True, "error", 0, msg)

        payload = resp.json()
        entries = payload if isinstance(payload, list) else []
        signals: list[ExternalSignal] = []

        for entry in entries:
            paper = entry.get("paper") if isinstance(entry.get("paper"), dict) else entry
            paper_id = str(paper.get("id") or "").strip()
            title = str(entry.get("title") or paper.get("title") or "").strip()
            if not title or not paper_id:
                continue
            url = canonicalize_url(f"https://huggingface.co/papers/{paper_id}")
            pub_dt = _parse_dt_utc(entry.get("publishedAt") or paper.get("publishedAt"), now)
            age_h = max(1.0, (now - pub_dt).total_seconds() / 3600.0)
            upvotes = float(paper.get("upvotes") or entry.get("upvotes") or 0.0)
            comments = float(entry.get("numComments") or paper.get("numComments") or 0.0)
            summary = str(entry.get("summary") or paper.get("summary") or "").strip()
            ai_keywords = [str(k) for k in (paper.get("ai_keywords") or []) if k][:8]
            signals.append(
                ExternalSignal(
                    signal_id=_make_signal_id("hf_daily_papers", url),
                    platform="hf_daily_papers",
                    source_name="Hugging Face Daily Papers",
                    url=url,
                    title=title,
                    summary=summary[:1000],
                    published_at_utc=pub_dt.isoformat(),
                    fetched_at_utc=now_iso,
                    score=upvotes,
                    comments=comments,
                    tags=ai_keywords,
                    age_hours=age_h,
                )
            )

        signals.sort(key=lambda s: (s.score + 2.0 * s.comments) / s.age_hours, reverse=True)
        signals = signals[:max_items]
        return signals, CollectorStatus("hf_daily_papers", True, "ok", len(signals))
    except Exception as exc:
        log.warning("Hugging Face Daily Papers collector exception: %s", exc)
        return [], CollectorStatus("hf_daily_papers", True, "error", 0, repr(exc))


def collect_lobsters(
    session: requests.Session,
    lookback_hours: int = 72,
    max_items: int = 30,
) -> tuple[list[ExternalSignal], CollectorStatus]:
    """Fetch high-signal engineering discussions from Lobste.rs public JSON endpoint."""
    now = datetime.now(timezone.utc)
    now_iso = now.isoformat()
    cutoff = now - timedelta(hours=lookback_hours)

    try:
        resp = session.get("https://lobste.rs/hottest.json", timeout=15)
        if resp.status_code != 200:
            msg = f"HTTP {resp.status_code}: {resp.text[:160]}"
            log.warning("Lobste.rs collector failed: %s", msg)
            return [], CollectorStatus("lobsters", True, "error", 0, msg)

        stories = resp.json() if isinstance(resp.json(), list) else []
        signals: list[ExternalSignal] = []

        for st in stories:
            title = str(st.get("title") or "").strip()
            raw_url = str(st.get("url") or st.get("comments_url") or "").strip()
            url = canonicalize_url(raw_url)
            if not title or not url:
                continue
            pub_dt = _parse_dt_utc(st.get("created_at"), now)
            if pub_dt < cutoff:
                continue
            age_h = max(1.0, (now - pub_dt).total_seconds() / 3600.0)
            score = float(st.get("score") or 0.0)
            comments = float(st.get("comment_count") or 0.0)
            tags = [str(t) for t in (st.get("tags") or []) if t]
            signals.append(
                ExternalSignal(
                    signal_id=_make_signal_id("lobsters", url),
                    platform="lobsters",
                    source_name="Lobste.rs",
                    url=url,
                    title=title,
                    summary=str(st.get("description_plain") or st.get("description") or "")[:1000],
                    published_at_utc=pub_dt.isoformat(),
                    fetched_at_utc=now_iso,
                    score=score,
                    comments=comments,
                    tags=tags[:8],
                    age_hours=age_h,
                )
            )

        signals.sort(key=lambda s: (s.score + 2.0 * s.comments) / s.age_hours, reverse=True)
        signals = signals[:max_items]
        return signals, CollectorStatus("lobsters", True, "ok", len(signals))
    except Exception as exc:
        log.warning("Lobste.rs collector exception: %s", exc)
        return [], CollectorStatus("lobsters", True, "error", 0, repr(exc))


def collect_bluesky_authenticated(
    session: requests.Session,
    handle: str,
    app_password: str,
    lookback_hours: int = 48,
) -> tuple[list[ExternalSignal], CollectorStatus]:
    """Optional authenticated Bluesky collector (honestly labeled `bluesky`, never X)."""
    if not handle or not app_password:
        return [], CollectorStatus("bluesky", False, "skipped_no_credentials", 0, "BLUESKY_HANDLE/APP_PASSWORD not set")

    now = datetime.now(timezone.utc)
    now_iso = now.isoformat()
    since_iso = (now - timedelta(hours=lookback_hours)).strftime("%Y-%m-%dT%H:%M:%SZ")

    try:
        auth_resp = session.post(
            "https://bsky.social/xrpc/com.atproto.server.createSession",
            json={"identifier": handle, "password": app_password},
            timeout=15,
        )
        if auth_resp.status_code != 200:
            msg = f"Auth HTTP {auth_resp.status_code}: {auth_resp.text[:140]}"
            log.warning("Bluesky auth failed: %s", msg)
            return [], CollectorStatus("bluesky", True, "error", 0, msg)

        access_jwt = str(auth_resp.json().get("accessJwt") or "")
        if not access_jwt:
            return [], CollectorStatus("bluesky", True, "error", 0, "Missing accessJwt in session response")

        queries = ("TSMC CoWoS", "HBM4 NVIDIA", "vLLM SGLang", "silicon photonics", "quantum qubit", "CRISPR")
        signals: list[ExternalSignal] = []
        seen: set[str] = set()
        headers = {"Authorization": f"Bearer {access_jwt}"}

        for q in queries:
            resp = session.get(
                "https://bsky.social/xrpc/app.bsky.feed.searchPosts",
                params={"q": q, "limit": 10, "sort": "top", "since": since_iso},
                headers=headers,
                timeout=12,
            )
            if resp.status_code != 200:
                log.warning("Bluesky search '%s' returned HTTP %s", q, resp.status_code)
                continue
            for post in resp.json().get("posts", []):
                uri = str(post.get("uri") or "").strip()
                record = post.get("record") or {}
                text = str(record.get("text") or "").strip()
                if not uri or not text or uri in seen:
                    continue
                seen.add(uri)
                pub_dt = _parse_dt_utc(record.get("createdAt"), now)
                age_h = max(1.0, (now - pub_dt).total_seconds() / 3600.0)
                signals.append(
                    ExternalSignal(
                        signal_id=_make_signal_id("bluesky", uri),
                        platform="bluesky",
                        source_name="Bluesky",
                        url=uri,
                        title=text[:160],
                        summary=text[:1000],
                        published_at_utc=pub_dt.isoformat(),
                        fetched_at_utc=now_iso,
                        score=float(post.get("likeCount") or 0.0),
                        comments=float(post.get("replyCount") or 0.0),
                        shares=float(post.get("repostCount") or 0.0),
                        age_hours=age_h,
                    )
                )

        return signals, CollectorStatus("bluesky", True, "ok", len(signals))
    except Exception as exc:
        log.warning("Bluesky collector exception: %s", exc)
        return [], CollectorStatus("bluesky", True, "error", 0, repr(exc))


def collect_reddit_oauth(
    session: requests.Session,
    client_id: str,
    client_secret: str,
    subreddits: list[str],
    lookback_hours: int = 48,
) -> tuple[list[ExternalSignal], CollectorStatus]:
    """Optional authenticated Reddit OAuth collector (never uses broken unauthenticated .json)."""
    if not client_id or not client_secret:
        return [], CollectorStatus("reddit", False, "skipped_no_credentials", 0, "REDDIT_CLIENT_ID/SECRET not set")

    now = datetime.now(timezone.utc)
    now_iso = now.isoformat()
    cutoff = now - timedelta(hours=lookback_hours)

    try:
        token_resp = session.post(
            "https://www.reddit.com/api/v1/access_token",
            auth=(client_id, client_secret),
            data={"grant_type": "client_credentials"},
            timeout=15,
        )
        if token_resp.status_code != 200:
            msg = f"OAuth HTTP {token_resp.status_code}: {token_resp.text[:140]}"
            log.warning("Reddit OAuth failed: %s", msg)
            return [], CollectorStatus("reddit", True, "error", 0, msg)

        access_token = str(token_resp.json().get("access_token") or "")
        headers = {"Authorization": f"Bearer {access_token}"}
        signals: list[ExternalSignal] = []

        for sub in subreddits:
            resp = session.get(
                f"https://oauth.reddit.com/r/{sub}/hot",
                params={"limit": 15},
                headers=headers,
                timeout=12,
            )
            if resp.status_code != 200:
                log.warning("Reddit OAuth r/%s returned HTTP %s", sub, resp.status_code)
                continue
            for child in resp.json().get("data", {}).get("children", []):
                post = child.get("data") or {}
                title = str(post.get("title") or "").strip()
                raw_url = str(post.get("url") or f"https://www.reddit.com{post.get('permalink', '')}").strip()
                url = canonicalize_url(raw_url)
                if not title or not url:
                    continue
                pub_dt = _parse_dt_utc(post.get("created_utc"), now)
                if pub_dt < cutoff:
                    continue
                age_h = max(1.0, (now - pub_dt).total_seconds() / 3600.0)
                signals.append(
                    ExternalSignal(
                        signal_id=_make_signal_id("reddit", url),
                        platform="reddit",
                        source_name=f"r/{sub}",
                        url=url,
                        title=title,
                        summary=str(post.get("selftext") or "")[:1000],
                        published_at_utc=pub_dt.isoformat(),
                        fetched_at_utc=now_iso,
                        score=float(post.get("score") or 0.0),
                        comments=float(post.get("num_comments") or 0.0),
                        age_hours=age_h,
                    )
                )
        return signals, CollectorStatus("reddit", True, "ok", len(signals))
    except Exception as exc:
        log.warning("Reddit OAuth collector exception: %s", exc)
        return [], CollectorStatus("reddit", True, "error", 0, repr(exc))


def collect_x_api_v2(
    session: requests.Session,
    bearer_token: str,
    queries: list[str],
    lookback_hours: int = 48,
) -> tuple[list[ExternalSignal], CollectorStatus]:
    """Optional official X API v2 Recent Search collector."""
    if not bearer_token:
        return [], CollectorStatus("x_twitter", False, "skipped_no_credentials", 0, "X_BEARER_TOKEN not set")

    now = datetime.now(timezone.utc)
    now_iso = now.isoformat()
    start_time = (now - timedelta(hours=min(168, lookback_hours))).strftime("%Y-%m-%dT%H:%M:%SZ")
    headers = {"Authorization": f"Bearer {bearer_token}"}
    signals: list[ExternalSignal] = []
    seen: set[str] = set()
    errors: list[str] = []

    for q in queries:
        if not q.strip():
            continue
        try:
            resp = session.get(
                "https://api.x.com/2/tweets/search/recent",
                params={
                    "query": f"({q.strip()}) -is:retweet lang:en",
                    "max_results": 20,
                    "start_time": start_time,
                    "tweet.fields": "created_at,public_metrics",
                },
                headers=headers,
                timeout=15,
            )
            if resp.status_code != 200:
                msg = f"HTTP {resp.status_code}: {resp.text[:140]}"
                log.warning("X API v2 query '%s' failed: %s", q[:40], msg)
                errors.append(msg)
                continue
            for tw in resp.json().get("data", []):
                tid = str(tw.get("id") or "").strip()
                text = str(tw.get("text") or "").strip()
                if not tid or not text or tid in seen:
                    continue
                seen.add(tid)
                url = f"https://x.com/i/web/status/{tid}"
                metrics = tw.get("public_metrics") or {}
                pub_dt = _parse_dt_utc(tw.get("created_at"), now)
                age_h = max(1.0, (now - pub_dt).total_seconds() / 3600.0)
                signals.append(
                    ExternalSignal(
                        signal_id=_make_signal_id("x_twitter", url),
                        platform="x_twitter",
                        source_name="X (API v2)",
                        url=url,
                        title=text[:160],
                        summary=text[:1000],
                        published_at_utc=pub_dt.isoformat(),
                        fetched_at_utc=now_iso,
                        score=float(metrics.get("like_count") or 0.0),
                        comments=float(metrics.get("reply_count") or 0.0),
                        shares=float(metrics.get("retweet_count") or 0.0) + float(metrics.get("quote_count") or 0.0),
                        age_hours=age_h,
                    )
                )
        except Exception as exc:
            log.warning("X API v2 exception: %s", exc)
            errors.append(repr(exc))

    status = "ok" if signals or not errors else "error"
    return signals, CollectorStatus("x_twitter", True, status, len(signals), "; ".join(errors[:2]))


def collect_all_external_signals(
    lookback_hours: int = 48,
    session: requests.Session | None = None,
) -> tuple[list[ExternalSignal], list[CollectorStatus]]:
    """Run all enabled external trend collectors and return deduplicated signals + status report."""
    http = session or build_http_session()
    all_signals: list[ExternalSignal] = []
    statuses: list[CollectorStatus] = []

    if _env_bool("ENABLE_HN_COLLECTOR", True):
        sigs, st = collect_hacker_news(
            session=http,
            lookback_hours=lookback_hours,
            min_points=int(os.getenv("HN_MIN_POINTS", "20")),
            max_items=int(os.getenv("HN_MAX_ITEMS", "40")),
        )
        all_signals.extend(sigs)
        statuses.append(st)
    else:
        statuses.append(CollectorStatus("hacker_news", False, "disabled"))

    if _env_bool("ENABLE_GITHUB_COLLECTOR", True):
        sigs, st = collect_github_velocity(
            session=http,
            lookback_days=int(os.getenv("GITHUB_LOOKBACK_DAYS", "14")),
            min_stars=int(os.getenv("GITHUB_MIN_STARS", "25")),
            max_items=int(os.getenv("GITHUB_MAX_ITEMS", "30")),
            github_token=os.getenv("GITHUB_TOKEN", "").strip(),
        )
        all_signals.extend(sigs)
        statuses.append(st)
    else:
        statuses.append(CollectorStatus("github", False, "disabled"))

    if _env_bool("ENABLE_HF_PAPERS_COLLECTOR", True):
        sigs, st = collect_hf_daily_papers(
            session=http,
            max_items=int(os.getenv("HF_PAPERS_MAX_ITEMS", "30")),
        )
        all_signals.extend(sigs)
        statuses.append(st)
    else:
        statuses.append(CollectorStatus("hf_daily_papers", False, "disabled"))

    if _env_bool("ENABLE_LOBSTERS_COLLECTOR", True):
        sigs, st = collect_lobsters(
            session=http,
            lookback_hours=max(lookback_hours, 72),
            max_items=int(os.getenv("LOBSTERS_MAX_ITEMS", "30")),
        )
        all_signals.extend(sigs)
        statuses.append(st)
    else:
        statuses.append(CollectorStatus("lobsters", False, "disabled"))

    # Optional authenticated social collectors
    bsky_sigs, bsky_st = collect_bluesky_authenticated(
        session=http,
        handle=os.getenv("BLUESKY_HANDLE", "").strip(),
        app_password=os.getenv("BLUESKY_APP_PASSWORD", "").strip(),
        lookback_hours=lookback_hours,
    )
    all_signals.extend(bsky_sigs)
    statuses.append(bsky_st)

    subs = [
        s.strip()
        for s in os.getenv(
            "REDDIT_SUBREDDITS",
            "semiconductors,hardware,LocalLLaMA,MachineLearning,biotech,robotics,QuantumComputing",
        ).split(",")
        if s.strip()
    ]
    reddit_sigs, reddit_st = collect_reddit_oauth(
        session=http,
        client_id=os.getenv("REDDIT_CLIENT_ID", "").strip(),
        client_secret=os.getenv("REDDIT_CLIENT_SECRET", "").strip(),
        subreddits=subs,
        lookback_hours=lookback_hours,
    )
    all_signals.extend(reddit_sigs)
    statuses.append(reddit_st)

    x_queries = [q.strip() for q in os.getenv("X_SEARCH_QUERIES", "").split(",") if q.strip()]
    x_sigs, x_st = collect_x_api_v2(
        session=http,
        bearer_token=os.getenv("X_BEARER_TOKEN", "").strip(),
        queries=x_queries,
        lookback_hours=lookback_hours,
    )
    all_signals.extend(x_sigs)
    statuses.append(x_st)

    # Deduplicate exact (platform, url) pairs
    deduped: list[ExternalSignal] = []
    seen_ids: set[str] = set()
    for sig in all_signals:
        if sig.signal_id not in seen_ids:
            seen_ids.add(sig.signal_id)
            deduped.append(sig)

    return deduped, statuses
