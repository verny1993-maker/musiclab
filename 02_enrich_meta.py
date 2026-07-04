"""
02_enrich_meta.py — Enrich track metadata from Discogs, MusicBrainz, Last.fm.

For each track with meta=null:
  1. Discogs (primary) — label, release, year, discogs_url
  2. MusicBrainz / Last.fm — genres
  3. All through rate_limits
  4. Idempotent: skips already-enriched tracks

Usage:
    python 02_enrich_meta.py --set data/sets/<set_id>.json
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.lib_io import load_json, write_json
from lib.rate_limits import discogs_get, lastfm_get, musicbrainz_get

logger = logging.getLogger("02_enrich_meta")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")


def _search_discogs(artist: str, title: str) -> dict | None:
    """Search Discogs for a release matching artist + title."""
    query = f"{artist} {title}"
    try:
        data = discogs_get(
            "/database/search",
            params={
                "q": query,
                "type": "release",
                "per_page": 3,
            },
        )
    except Exception as e:
        logger.warning("Discogs search failed for %s — %s: %s", artist, title, e)
        return None

    results = data.get("results", [])
    if not results:
        return None

    best = results[0]
    # Extract fields
    label = ""
    if isinstance(best.get("label"), list) and best["label"]:
        label = best["label"][0]
    release_title = best.get("title", "")
    year = best.get("year")
    genres = []
    for field in ("genre", "style"):
        val = best.get(field, [])
        if isinstance(val, list):
            genres.extend(val)
        elif isinstance(val, str):
            genres.append(val)

    resource_url = best.get("resource_url", "")
    discogs_url = resource_url.replace("api.discogs.com", "www.discogs.com").replace(
        "https://www.discogs.com/releases/", "https://www.discogs.com/release/"
    )

    return {
        "label": label or None,
        "release": release_title or None,
        "year": int(year) if year else None,
        "genres": genres if genres else None,
        "discogs_url": discogs_url or None,
        "source": "discogs",
    }


def _search_musicbrainz(artist: str, title: str) -> dict | None:
    """Search MusicBrainz for genres."""
    query = f'artist:"{artist}" AND recording:"{title}"'
    try:
        data = musicbrainz_get(
            "/ws/2/recording/",
            params={
                "query": query,
                "limit": 3,
            },
        )
    except Exception as e:
        logger.warning("MusicBrainz search failed for %s — %s: %s", artist, title, e)
        return None

    recordings = data.get("recordings", [])
    genres: list[str] = []
    for rec in recordings[:2]:
        for tag in rec.get("tags", []):
            name = tag.get("name", "")
            if name and name not in genres:
                genres.append(name)

    if genres:
        return {"genres": genres, "source": "musicbrainz"}
    return None


def _search_lastfm(artist: str, title: str) -> dict | None:
    """Search Last.fm for genres."""
    try:
        data = lastfm_get(
            {
                "method": "track.getInfo",
                "artist": artist,
                "track": title,
            }
        )
    except Exception as e:
        logger.warning("Last.fm search failed for %s — %s: %s", artist, title, e)
        return None

    track_info = data.get("track", {})
    toptags = track_info.get("toptags", {}).get("tag", [])
    if isinstance(toptags, dict):
        toptags = [toptags]

    genres = [t["name"] for t in toptags[:5] if isinstance(t, dict) and t.get("name")]
    if genres:
        return {"genres": genres, "source": "lastfm"}
    return None


def enrich_meta(set_path: str) -> Path:
    """Enrich metadata for all tracks in the set. Idempotent."""
    set_data = load_json(set_path, "set")
    errors: list[dict] = list(set_data.get("errors", []))
    enriched = 0
    skipped = 0

    for track in set_data["tracks"]:
        # Idempotent: skip already-enriched
        if track.get("meta") is not None:
            skipped += 1
            continue

        artist = track.get("artist", "")
        title = track.get("title", "")
        if not artist or not title:
            continue

        pos = track["position"]
        logger.info("[%d/%d] %s — %s", pos, len(set_data["tracks"]), artist, title)

        # 1. Discogs (primary)
        meta = _search_discogs(artist, title)

        # 2. MusicBrainz / Last.fm for genres if Discogs didn't find them
        if meta and not meta.get("genres"):
            mb = _search_musicbrainz(artist, title)
            if mb:
                meta["genres"] = mb.get("genres")
                if meta.get("source") != "discogs":
                    meta["source"] = mb["source"]

        if meta and not meta.get("genres"):
            lf = _search_lastfm(artist, title)
            if lf:
                meta["genres"] = lf.get("genres")

        if meta:
            # Discogs year might be string
            if meta.get("year") is not None:
                try:
                    meta["year"] = int(meta["year"])
                except (ValueError, TypeError):
                    meta["year"] = None
            track["meta"] = meta
            enriched += 1
            logger.info(
                "  -> %s | year=%s | %d genres",
                meta.get("label", "?"),
                meta.get("year"),
                len(meta.get("genres") or []),
            )
        else:
            # Partial: try Last.fm directly if Discogs failed
            lf = _search_lastfm(artist, title)
            if lf:
                track["meta"] = {"genres": lf.get("genres"), "source": "lastfm"}
                enriched += 1
                logger.info("  -> lastfm only | %d genres", len(lf.get("genres") or []))
            else:
                errors.append(
                    {
                        "position": pos,
                        "stage": "02_enrich_meta",
                        "reason": f"No metadata found for {artist} — {title}",
                    }
                )
                logger.warning("  -> NOT FOUND")

        # Small delay between tracks (already handled by rate_limits per-request)

    set_data["errors"] = errors
    result = write_json(set_data, "set", set_path)
    logger.info(
        "Done: %d enriched, %d skipped, %d errors",
        enriched,
        skipped,
        len(errors) - len(set_data.get("errors", [])),
    )
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Enrich track metadata from Discogs/MusicBrainz/Last.fm"
    )
    parser.add_argument("--set", required=True, help="Path to set JSON file")
    args = parser.parse_args()
    enrich_meta(args.set)


if __name__ == "__main__":
    main()
