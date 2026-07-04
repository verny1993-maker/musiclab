"""
01_parse.py — Parse DJ set tracklists from source platforms into set.schema.json.

Backend architecture:
  - 'set79': primary parser — fetches set79.com tracklist pages.
  - '1001tl': placeholder (Cloudflare-protected, needs scrape_1001tl from rate_limits).
  - 'fallback': placeholder for future YouTube description parser.

Usage:
    python 01_parse.py --url "https://set79.com/tracklist/..."   [--backend set79]
    python 01_parse.py --url "https://1001.tl/..."               [--backend 1001tl]

Output:
    Writes data/sets/<set_id>.json via lib_io (validated against set.schema.json).
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

from bs4 import BeautifulSoup

# Project imports
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.lib_io import write_json
from lib.rate_limits import set79_get  # ← single gate for all HTTP

logger = logging.getLogger("01_parse")

Backend = Literal["set79", "1001tl", "fallback"]

# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════


def _parse_timecode(tc: str) -> float:
    """Convert 'HH:MM:SS' or 'MM:SS' to seconds."""
    parts = tc.strip().split(":")
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    elif len(parts) == 2:
        return int(parts[0]) * 60 + int(parts[1])
    return 0.0


def _split_artist_title(raw: str, *, title_first: bool = False) -> tuple[str, str]:
    """Split a name string into (artist, title).

    By default assumes 'Artist - Title' (standard format).
    Set title_first=True for 'Title - Artist' (set79 format).
    """
    raw = raw.strip()
    for sep in (" – ", " - ", "–", "-"):
        if sep in raw:
            left, right = raw.split(sep, 1)
            a, t = (left.strip(), right.strip())
            if title_first:
                a, t = t, a
            return a, t
    return ("", raw)


def _slugify(text: str) -> str:
    """Make a filename-safe slug."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[-\s]+", "-", text)
    return text[:60]


def _extract_soundcloud_url(html: str, set79_url: str) -> str | None:
    """
    Extract SoundCloud URL from set79 page.
    Strategy:
      1. Look for <a href="https://soundcloud.com/..."> in the HTML.
      2. Fall back to reconstructing from set79 URL path
         (format: /tracklist/soundcloud.com/<user>/<slug>).
    """
    soup = BeautifulSoup(html, "lxml")
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "soundcloud.com" in href and "/tracklist/" not in href:
            return href
    # Reconstruct from set79 path
    parsed = urlparse(set79_url)
    path = parsed.path
    if path.startswith("/tracklist/"):
        sc_path = path[len("/tracklist/") :]
        return f"https://{sc_path}"
    return None


def _extract_youtube_url(html: str) -> str | None:
    """Extract YouTube URL from set79 page HTML."""
    soup = BeautifulSoup(html, "lxml")
    # Look for YouTube iframe or link
    for iframe in soup.find_all("iframe", src=True):
        src = iframe["src"]
        if "youtube.com" in src or "youtu.be" in src:
            return src
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "youtube.com/watch" in href or "youtu.be/" in href:
            return href
    return None


# ═══════════════════════════════════════════════════════════════════════
# set79 parser
# ═══════════════════════════════════════════════════════════════════════


def parse_set79(url: str) -> dict:
    """
    Parse a set79.com tracklist page into set.schema.json format.

    Args:
        url: Full set79 tracklist URL, e.g.
             https://set79.com/tracklist/soundcloud.com/user/slug

    Returns:
        Dict conforming to set.schema.json.
    """
    logger.info("Fetching set79: %s", url)
    html = set79_get(url)

    soup = BeautifulSoup(html, "lxml")

    # ── Title ──
    h1 = soup.find("h1")
    title = h1.get_text(strip=True) if h1 else "Unknown Set"

    # ── Artist ──
    # set79 doesn't always have a dedicated artist field — use title or page context.
    # The master page lists sets by artist; we pass artist separately via CLI.
    # For now, try to extract from breadcrumbs or metadata.
    artist = ""
    # Check for "DJs:" section
    for link in soup.find_all("a", href=True):
        href = link["href"]
        if href.startswith("/dj/"):
            artist = link.get_text(strip=True)
            break

    # ── Venue / description ──
    venue = ""
    desc_el = soup.find("p")  # First paragraph is usually the set description
    if desc_el:
        desc_text = desc_el.get_text()
        # Look for venue patterns in description
        for kw in [
            "Boiler Room",
            "HÖR",
            "Keep Hush",
            "DEF",
            "NTS",
            "Rinse FM",
            "Coachella",
            "Ultra",
            "EDC",
            "Awakenings",
            "Tomorrowland",
        ]:
            if kw in desc_text:
                venue = kw
                break

    # ── Date ──
    date_str = ""
    # Try to find date in title or description
    date_match = re.search(r"(\d{4}-\d{2}-\d{2})", title + " " + html[:5000])
    if date_match:
        date_str = date_match.group(1)
    # Also try NTS format: "260419" → 2019-04-26
    nts_match = re.search(r"(\d{2})(\d{2})(\d{2})", title)
    if nts_match and not date_str:
        d, m, y = nts_match.group(1), nts_match.group(2), nts_match.group(3)
        date_str = f"20{y}-{m}-{d}"

    # ── Audio URLs ──
    audio_url = _extract_soundcloud_url(html, url)
    youtube_url = _extract_youtube_url(html)

    # ── Tracklist table ──
    tracks = []
    # set79 uses a standard <table> with <tr> rows
    # Data rows: 6 cells — [checkbox, #, ♥, Track Name, Time, Links]
    table = soup.find("table")
    if table:
        for row in table.find_all("tr"):
            cells = row.find_all("td")
            if len(cells) < 5:
                continue  # skip header row (uses <th>)
            try:
                pos = int(cells[1].get_text(strip=True))
            except (ValueError, IndexError):
                continue

            # Track name — cell[3]. set79 format: "Title - Artist"
            raw_name = cells[3].get_text(" ", strip=True)
            if raw_name.lower().startswith("unknown"):
                raw_name = raw_name[len("Unknown ") :].strip()
            track_artist, track_title = _split_artist_title(raw_name, title_first=True)

            # Time — cell[4]
            time_str = cells[4].get_text(strip=True).lstrip("▶").strip()
            start_sec = _parse_timecode(time_str)

            tracks.append(
                {
                    "position": pos,
                    "title": track_title or raw_name,
                    "artist": track_artist,
                    "timecode": time_str,
                    "start_sec": start_sec,
                    "audio": None,
                    "meta": None,
                }
            )

    # ── Build set dict ──
    # Derive set_id
    parsed = urlparse(url)
    slug = parsed.path.rstrip("/").split("/")[-1] if parsed.path else "unknown"
    set_id = f"set_{_slugify(artist or 'unknown')}_{slug[:30]}"

    set_data: dict[str, Any] = {
        "id": set_id,
        "title": title,
        "artist": artist,
        "venue": venue or None,
        "audio_url": audio_url,
        "youtube_url": youtube_url if youtube_url != audio_url else None,
        "source_url": url,
        "source_platform": "set79",
        "parsed_at": datetime.utcnow().isoformat() + "Z",
        "enriched_at": None,
        "tracks": tracks,
        "energy_curve_archetype": {
            "type": None,
            "segments": [],
        },
        "errors": [],
    }

    return set_data


# ═══════════════════════════════════════════════════════════════════════
# 1001tl parser (placeholder)
# ═══════════════════════════════════════════════════════════════════════


def parse_1001tl(url: str) -> dict:
    """
    Parse a 1001tracklists.com tracklist page.

    NOT IMPLEMENTED — placeholder.
    Will use scrape_1001tl() from rate_limits when activated.
    """
    raise NotImplementedError(
        "1001tl parser not implemented. Cloudflare-protected — "
        "needs scrape_1001tl() from rate_limits."
    )


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(description="Parse a DJ set tracklist from URL")
    parser.add_argument("--url", required=True, help="Tracklist source URL")
    parser.add_argument(
        "--backend",
        choices=["set79", "1001tl", "fallback"],
        default="set79",
        help="Parser backend (default: set79)",
    )
    parser.add_argument(
        "--artist",
        help="Override artist name (set79 doesn't always auto-detect)",
    )
    parser.add_argument(
        "--venue",
        help="Override venue name",
    )
    parser.add_argument(
        "--date",
        help="Override date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--output",
        help="Output JSON path (default: data/sets/<set_id>.json)",
    )
    args = parser.parse_args()

    if args.backend == "fallback":
        print("[01_parse] fallback backend is a placeholder — not implemented")
        sys.exit(1)

    if args.backend == "1001tl":
        print("[01_parse] 1001tl backend is a placeholder — not implemented")
        sys.exit(1)

    # Parse
    set_data = parse_set79(args.url)

    # Overrides
    if args.artist:
        set_data["artist"] = args.artist
    if args.venue:
        set_data["venue"] = args.venue
    # Note: --date is informational only (not in schema) — stored in title context

    # Output path
    output_path = args.output or f"data/sets/{set_data['id']}.json"
    result = write_json(set_data, "set", output_path)
    print(f"[01_parse] Written: {result}")
    print(f"  Tracks: {len(set_data['tracks'])}")
    print(f"  Artist: {set_data['artist']}")
    print(f"  Venue:  {set_data['venue']}")
    print(f"  Audio:  {set_data.get('audio_url', 'none')}")


if __name__ == "__main__":
    main()
