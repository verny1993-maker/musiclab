"""
rate_limits.py — Single gate for ALL external HTTP requests in MusicLab.

Every script (01_parse, 02_enrich, 04_hypothesis, 05_build_cards, etc.)
MUST route all HTTP calls through this module.  Like lib_io for writes —
one door, one set of rules.

Rate limits (sourced from official docs, verified 2025-06):

  MusicBrainz  → 1 req/s per IP (hard on/off block), 50 req/s per User-Agent.
                 503 on limit.  MUST set User-Agent with contact (app + email/URL).
                 Source: https://musicbrainz.org/doc/MusicBrainz_API/Rate_Limiting

  Discogs      → 60 req/min authenticated, 25 req/min unauthenticated.
                 429 on limit.  Header: X-Discogs-Ratelimit-Remaining.
                 Source: https://www.discogs.com/forum/thread/521520689469733cfcfd2089

  Last.fm      → 5 req/s per IP, averaged over 5 minutes.
                 Error code 29 "Rate Limit Exceeded".
                 Source: https://www.last.fm/api/tos §4.4

  Spotify      → Dynamic limit, rolling 30-second window.
                 Development mode: low; Extended quota: high.
                 429 → Retry-After header (seconds).  Use batch endpoints.
                 Source: https://developer.spotify.com/documentation/web-api/concepts/rate-limits

  1001tl       → NO API.  Cloudflare-protected scraping.
                 Adaptive: random delays, exponential backoff, cursor-based
                 progress saving, max_retries with increasing cooldown.

Usage:
    from lib.rate_limits import musicbrainz_get, discogs_get, lastfm_get, spotify_get, scrape_1001tl

    data = musicbrainz_get("/ws/2/artist/...")
    data = discogs_get("/database/search?q=...")
    data = lastfm_get({"method": "artist.getInfo", "artist": "..."})
    data = spotify_get("/v1/tracks/...")
    html = scrape_1001tl("https://1001.tl/...", cursor_path="data/cache/001tl_cursor.json")
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urljoin

import requests

logger = logging.getLogger("rate_limits")

# ═══════════════════════════════════════════════════════════════════════
# Configuration (read from env, fall back to defaults)
# ═══════════════════════════════════════════════════════════════════════

def _env(key: str, default: str = "") -> str:
    val = os.environ.get(key, "")
    if val:
        return val
    # Fallback: read from .env file directly
    env_paths = [
        Path(os.environ.get("HERMES_HOME", "")) / ".env",
        Path.home() / ".hermes" / ".env",
    ]
    for env_path in env_paths:
        if env_path.exists():
            try:
                with open(env_path) as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith(f"{key}="):
                            val = line.split("=", 1)[1].strip()
                            if val and val not in ("***", ""):
                                os.environ[key] = val
                                return val
            except Exception:
                pass
    return default

# MusicBrainz
MB_BASE_URL = "https://musicbrainz.org"
MB_USER_AGENT = _env(
    "MB_USER_AGENT",
    "MusicLab/0.1 ( verny1993@yandex.ru )",
)
MB_RATE = 1.0          # req/s per IP — hard on/off, so stay well under
MB_MIN_INTERVAL = 1.0 / MB_RATE  # 1.0s between requests

# Discogs
DISCOGS_BASE_URL = "https://api.discogs.com"
DISCOGS_TOKEN = _env("DISCOGS_TOKEN", "")
DISCOGS_RATE = 60       # req/min authenticated
DISCOGS_MIN_INTERVAL = 60.0 / DISCOGS_RATE  # 1.0s between requests

# Last.fm
LASTFM_BASE_URL = "https://ws.audioscrobbler.com/2.0/"
LASTFM_API_KEY = _env("LASTFM_API_KEY", "")
LASTFM_RATE = 5          # req/s per IP
LASTFM_MIN_INTERVAL = 1.0 / LASTFM_RATE  # 0.2s

# Spotify
SPOTIFY_BASE_URL = "https://api.spotify.com"
SPOTIFY_ACCESS_TOKEN = _env("SPOTIFY_ACCESS_TOKEN", "")

# 1001tracklists
N001TL_BASE_URL = "https://www.1001tracklists.com"
N001TL_MIN_DELAY = 2.0      # minimum random delay between requests (seconds)
N001TL_MAX_DELAY = 5.0      # maximum random delay
N001TL_BACKOFF_BASE = 10.0  # base backoff after 403/429 (seconds)
N001TL_MAX_RETRIES = 5      # max retries per URL
N001TL_BACKOFF_MULTIPLIER = 2.0  # exponential multiplier


# ═══════════════════════════════════════════════════════════════════════
# Internal: per-source token bucket / interval tracker
# ═══════════════════════════════════════════════════════════════════════

_last_request: dict[str, float] = {}  # source_name → last request timestamp


def _wait_if_needed(source: str, min_interval: float) -> None:
    """Sleep if less than min_interval has passed since the last request to `source`."""
    now = time.monotonic()
    last = _last_request.get(source, 0.0)
    elapsed = now - last
    if elapsed < min_interval:
        sleep_time = min_interval - elapsed
        logger.debug("%s: throttling %.2fs (interval=%.2fs)", source, sleep_time, min_interval)
        time.sleep(sleep_time)
    _last_request[source] = time.monotonic()


def _build_session(headers: dict[str, str], timeout: int = 45) -> requests.Session:
    """Create a session with a connect timeout and standard headers."""
    s = requests.Session()
    adapter = requests.adapters.HTTPAdapter(
        pool_connections=1,
        pool_maxsize=1,
        max_retries=0,  # we handle retries ourselves
    )
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    s.headers.update(headers)
    # Connection-level timeouts are set per-request; session-level defaults
    s.request_timeout = timeout  # type: ignore[attr-defined]
    return s


# ═══════════════════════════════════════════════════════════════════════
# MusicBrainz
# ═══════════════════════════════════════════════════════════════════════

_mb_session: Optional[requests.Session] = None


def musicbrainz_get(path: str, params: dict | None = None) -> dict:
    """
    GET a MusicBrainz API endpoint.

    Args:
        path: URL path relative to https://musicbrainz.org (e.g. '/ws/2/artist/...').
        params: Optional query parameters (fmt=json is added automatically).

    Returns:
        Parsed JSON response.

    Raises:
        requests.HTTPError: on non-2xx response (including 503 rate limit).
    """
    global _mb_session
    if _mb_session is None:
        _mb_session = _build_session({
            "User-Agent": MB_USER_AGENT,
            "Accept": "application/json",
        })
        _mb_session.verify = False  # SSL broken on Windows
    _wait_if_needed("musicbrainz", MB_MIN_INTERVAL)

    if params is None:
        params = {}
    if "fmt" not in params:
        params["fmt"] = "json"

    url = urljoin(MB_BASE_URL, path)
    resp = _mb_session.get(url, params=params, timeout=(10, 45))
    resp.raise_for_status()
    return resp.json()


# ═══════════════════════════════════════════════════════════════════════
# Discogs
# ═══════════════════════════════════════════════════════════════════════

_discogs_session: Optional[requests.Session] = None


def discogs_get(path: str, params: dict | None = None) -> dict:
    """
    GET a Discogs API endpoint.

    Reads X-Discogs-Ratelimit-Remaining from response headers.
    Slows down when approaching zero.

    Args:
        path: URL path relative to https://api.discogs.com.
        params: Optional query parameters.

    Returns:
        Parsed JSON response.

    Raises:
        requests.HTTPError: on 429 rate limit.
    """
    global _discogs_session
    if _discogs_session is None:
        headers = {
            "User-Agent": MB_USER_AGENT,  # reuse MusicBrainz-style UA
            "Accept": "application/json",
        }
        if DISCOGS_TOKEN:
            headers["Authorization"] = f"Discogs token={DISCOGS_TOKEN}"
        _discogs_session = _build_session(headers)

    _wait_if_needed("discogs", DISCOGS_MIN_INTERVAL)

    url = urljoin(DISCOGS_BASE_URL, path)
    resp = _discogs_session.get(url, params=params, timeout=(10, 45))

    # Check rate-limit header
    remaining = resp.headers.get("X-Discogs-Ratelimit-Remaining")
    if remaining is not None:
        remaining = int(remaining)
        logger.debug("discogs: %d requests remaining", remaining)
        if remaining <= 5:
            # Slow down aggressively near the limit
            logger.warning("discogs: only %d requests left, pausing 5s", remaining)
            time.sleep(5.0)

    resp.raise_for_status()
    return resp.json()


# ═══════════════════════════════════════════════════════════════════════
# Last.fm
# ═══════════════════════════════════════════════════════════════════════

_lastfm_session: Optional[requests.Session] = None


def lastfm_get(params: dict) -> dict:
    """
    GET the Last.fm API.

    api_key is injected automatically from env.

    Args:
        params: Query parameters dict. Must include 'method'.
                'format': 'json' is added automatically.

    Returns:
        Parsed JSON response.

    Raises:
        requests.HTTPError: on error 29 "Rate Limit Exceeded".
    """
    global _lastfm_session
    if _lastfm_session is None:
        _lastfm_session = _build_session({
            "User-Agent": MB_USER_AGENT,
        })

    _wait_if_needed("lastfm", LASTFM_MIN_INTERVAL)

    params = dict(params)
    params.setdefault("api_key", LASTFM_API_KEY)
    params.setdefault("format", "json")

    resp = _lastfm_session.get(LASTFM_BASE_URL, params=params, timeout=(10, 45))
    resp.raise_for_status()

    data = resp.json()
    # Check for Last.fm error codes
    if isinstance(data, dict) and data.get("error"):
        code = data["error"]
        msg = data.get("message", "unknown")
        if code == 29:
            raise requests.HTTPError(
                f"Last.fm rate limit exceeded (error 29): {msg}",
                response=resp,
            )
        raise requests.HTTPError(
            f"Last.fm API error {code}: {msg}",
            response=resp,
        )
    return data


# ═══════════════════════════════════════════════════════════════════════
# Spotify
# ═══════════════════════════════════════════════════════════════════════

_spotify_session: Optional[requests.Session] = None


def spotify_get(path: str, params: dict | None = None) -> dict:
    """
    GET a Spotify Web API endpoint.

    OAuth token is injected from env (SPOTIFY_ACCESS_TOKEN).
    On 429, reads Retry-After header and waits before re-raising.

    Args:
        path: URL path relative to https://api.spotify.com/v1/.
        params: Optional query parameters.

    Returns:
        Parsed JSON response.

    Raises:
        requests.HTTPError: on 429 after waiting Retry-After.
    """
    global _spotify_session
    if _spotify_session is None:
        _spotify_session = _build_session({
            "Authorization": f"Bearer {SPOTIFY_ACCESS_TOKEN}",
            "Content-Type": "application/json",
        })

    url = urljoin(SPOTIFY_BASE_URL, f"/v1/{path.lstrip('/')}")
    resp = _spotify_session.get(url, params=params, timeout=(10, 45))

    if resp.status_code == 429:
        retry_after = resp.headers.get("Retry-After", "5")
        try:
            wait_sec = int(retry_after)
        except ValueError:
            wait_sec = 5
        logger.warning("spotify: 429 rate limit, waiting %ds (Retry-After)", wait_sec)
        time.sleep(wait_sec)
        # Re-raise — caller should handle / retry
        resp.raise_for_status()

    resp.raise_for_status()
    return resp.json()


# ═══════════════════════════════════════════════════════════════════════
# 1001tracklists — adaptive scraping (no API, Cloudflare-protected)
# ═══════════════════════════════════════════════════════════════════════

_n001tl_session: Optional[requests.Session] = None


def _n001tl_session_factory() -> requests.Session:
    """Build a session with browser-like headers for 1001tracklists."""
    s = _build_session({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/125.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "DNT": "1",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    })
    return s


def scrape_1001tl(
    url: str,
    cursor_path: Optional[str] = None,
    max_retries: int = N001TL_MAX_RETRIES,
) -> str:
    """
    Scrape a 1001tracklists page with adaptive rate limiting.

    Strategy:
      - Random delay between requests (N001TL_MIN_DELAY – N001TL_MAX_DELAY)
        to avoid uniform patterns that flag bots.
      - Exponential backoff on 403/429 (base: N001TL_BACKOFF_BASE,
        multiplier: N001TL_BACKOFF_MULTIPLIER).
      - Cursor-based progress saving: after each successful fetch, writes
        the URL to `cursor_path`.  On restart, skips already-fetched URLs.
      - max_retries with increasing cooldown per attempt.

    Args:
        url: Full 1001tracklists URL to scrape.
        cursor_path: Path to a JSON file tracking completed URLs.
                     If provided, skips URLs already present in the file.
        max_retries: Max retry attempts for this URL.

    Returns:
        HTML content as string.

    Raises:
        RuntimeError: after exhausting max_retries.
    """
    global _n001tl_session
    if _n001tl_session is None:
        _n001tl_session = _n001tl_session_factory()

    # ── Cursor check ──
    completed: set[str] = set()
    if cursor_path:
        cursor_file = Path(cursor_path)
        if cursor_file.exists():
            try:
                with open(cursor_file, encoding="utf-8") as f:
                    completed = set(json.load(f))
            except (json.JSONDecodeError, OSError):
                completed = set()
        if url in completed:
            logger.info("1001tl: skipping already-scraped URL: %s", url)
            return ""  # caller should handle empty

    # ── Adaptive delay ──
    delay = random.uniform(N001TL_MIN_DELAY, N001TL_MAX_DELAY)
    logger.debug("1001tl: random delay %.2fs before %s", delay, url)
    time.sleep(delay)

    # ── Request with exponential backoff ──
    last_exception: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = _n001tl_session.get(url, timeout=(10, 60))
            if resp.status_code in (403, 429):
                backoff = N001TL_BACKOFF_BASE * (N001TL_BACKOFF_MULTIPLIER ** (attempt - 1))
                logger.warning(
                    "1001tl: %d on %s (attempt %d/%d), backoff %.1fs",
                    resp.status_code, url, attempt, max_retries, backoff,
                )
                time.sleep(backoff)
                last_exception = requests.HTTPError(
                    f"1001tl returned {resp.status_code}", response=resp
                )
                continue
            resp.raise_for_status()

            # ── Save cursor ──
            if cursor_path:
                completed.add(url)
                cursor_file = Path(cursor_path)
                cursor_file.parent.mkdir(parents=True, exist_ok=True)
                # Atomic write
                tmp = cursor_file.with_suffix(".tmp")
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(sorted(completed), f)
                os.replace(tmp, cursor_file)

            return resp.text

        except requests.RequestException as e:
            last_exception = e
            if attempt < max_retries:
                backoff = N001TL_BACKOFF_BASE * (N001TL_BACKOFF_MULTIPLIER ** (attempt - 1))
                logger.warning(
                    "1001tl: request failed (attempt %d/%d): %s, backoff %.1fs",
                    attempt, max_retries, e, backoff,
                )
                time.sleep(backoff)

    raise RuntimeError(
        f"1001tl: exhausted {max_retries} retries for {url}"
    ) from last_exception


# ═══════════════════════════════════════════════════════════════════════
# set79 — no documented rate limit; standard 1 req/s throttle
# ═══════════════════════════════════════════════════════════════════════

_SET79_MIN_INTERVAL = 1.0  # 1 req/s — safe default
_set79_session: Optional[requests.Session] = None


def set79_get(url: str) -> str:
    """
    GET a set79.com page.

    Args:
        url: Full set79 URL.

    Returns:
        HTML content as string.

    Raises:
        requests.HTTPError: on non-2xx.
    """
    global _set79_session
    if _set79_session is None:
        _set79_session = _build_session({
            "User-Agent": MB_USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        })

    _wait_if_needed("set79", _SET79_MIN_INTERVAL)

    resp = _set79_session.get(url, timeout=(10, 45))
    resp.raise_for_status()
    return resp.text
