"""
enrich_library.py — Enrich Spotify liked tracks with Discogs/Last.fm/MusicBrainz metadata.
Saves to data/library/tracks_enriched.json
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from collections import Counter

# Add AudioLab to path for lib imports
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.rate_limits import discogs_get, musicbrainz_get, lastfm_get

logger = logging.getLogger("enrich_library")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")


def _search_discogs(artist: str, title: str) -> dict | None:
    primary_artist = artist.split(",")[0].strip()
    query = f"{primary_artist} {title}"
    try:
        data = discogs_get("/database/search", params={
            "q": query, "type": "release", "per_page": 3,
        })
    except Exception as e:
        logger.warning("Discogs failed for %s — %s: %s", primary_artist, title, e)
        return None

    results = data.get("results", [])
    if not results:
        return None

    best = results[0]
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
    primary_artist = artist.split(",")[0].strip()
    query = f'artist:"{primary_artist}" AND recording:"{title}"'
    try:
        data = musicbrainz_get("/ws/2/recording/", params={
            "query": query, "fmt": "json", "limit": 3,
        })
    except Exception as e:
        logger.warning("MusicBrainz failed for %s — %s: %s", primary_artist, title, e)
        return None

    recordings = data.get("recordings", [])
    if not recordings:
        return None

    best = recordings[0]
    tags = []
    if isinstance(best.get("tags"), list):
        tags = [t.get("name", "") for t in best["tags"] if t.get("name")]

    return {
        "genres": tags if tags else None,
        "source": "musicbrainz",
    }


def _search_lastfm(artist: str, title: str) -> dict | None:
    primary_artist = artist.split(",")[0].strip()
    try:
        data = lastfm_get({
            "method": "track.getInfo",
            "artist": primary_artist,
            "track": title,
            "autocorrect": "1",
        })
    except Exception as e:
        logger.warning("Last.fm failed for %s — %s: %s", primary_artist, title, e)
        return None

    track = data.get("track")
    if not track:
        return None

    toptags = track.get("toptags", {})
    tags = []
    if isinstance(toptags.get("tag"), list):
        tags = [t.get("name", "") for t in toptags["tag"] if t.get("name")]
    elif isinstance(toptags.get("tag"), dict):
        t = toptags["tag"]
        if t.get("name"):
            tags = [t["name"]]

    return {
        "genres": tags if tags else None,
        "source": "lastfm",
    }


def enrich_track(track: dict) -> dict:
    artist = track["artist"]
    title = track["name"]
    
    meta = {"sources": {}}
    
    logger.info("Discogs: %s — %s", artist, title)
    discogs = _search_discogs(artist, title)
    if discogs:
        meta["label"] = discogs.get("label")
        meta["release"] = discogs.get("release")
        meta["year"] = discogs.get("year") or track.get("year")
        meta["discogs_url"] = discogs.get("discogs_url")
        if discogs.get("genres"):
            meta["sources"]["discogs_genres"] = discogs["genres"]
    else:
        meta["year"] = track.get("year")
    
    logger.info("Last.fm: %s — %s", artist, title)
    lastfm = _search_lastfm(artist, title)
    if lastfm and lastfm.get("genres"):
        meta["sources"]["lastfm_genres"] = lastfm["genres"]
    
    logger.info("MusicBrainz: %s — %s", artist, title)
    mb = _search_musicbrainz(artist, title)
    if mb and mb.get("genres"):
        meta["sources"]["musicbrainz_genres"] = mb["genres"]
    
    all_genres = []
    for src_key in ["discogs_genres", "lastfm_genres", "musicbrainz_genres"]:
        if src_key in meta["sources"]:
            all_genres.extend(meta["sources"][src_key])
    
    meta["genres"] = sorted(set(g.lower() for g in all_genres)) if all_genres else None
    
    has_any = bool(meta.get("label") or meta.get("genres") or meta.get("discogs_url"))
    meta["meta_status"] = "enriched" if has_any else "unmatched"
    
    return meta


def main():
    import argparse
    project_root = Path(__file__).resolve().parent
    default_input = project_root / "data" / "library" / "tracks_all.json"
    default_output = project_root / "data" / "library" / "tracks_enriched_all.json"
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(default_input))
    parser.add_argument("--output", default=str(default_output))
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    with open(input_path, encoding="utf-8") as f:
        tracks = json.load(f)
    
    logger.info("Loaded %d tracks from %s", len(tracks), input_path)
    
    for i, track in enumerate(tracks):
        artist = track["artist"]
        title = track["name"]
        logger.info("[%d/%d] %s — %s", i+1, len(tracks), artist, title)
        
        try:
            meta = enrich_track(track)
            track["meta"] = meta
        except Exception as e:
            logger.error("Failed track %d: %s", i, e)
            track["meta"] = {"meta_status": "error", "error": str(e)}
        
        if (i + 1) % 10 == 0:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(tracks, f, ensure_ascii=False, indent=2)
            logger.info("Progress saved: %d/%d tracks", i+1, len(tracks))
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(tracks, f, ensure_ascii=False, indent=2)
    
    enriched = sum(1 for t in tracks if t.get("meta", {}).get("meta_status") == "enriched")
    unmatched = sum(1 for t in tracks if t.get("meta", {}).get("meta_status") == "unmatched")
    has_discogs = sum(1 for t in tracks if t.get("meta", {}).get("label") or t.get("meta", {}).get("discogs_url"))
    has_lastfm = sum(1 for t in tracks if "lastfm_genres" in t.get("meta", {}).get("sources", {}))
    has_mb = sum(1 for t in tracks if "musicbrainz_genres" in t.get("meta", {}).get("sources", {}))
    
    logger.info("=" * 50)
    logger.info("ENRICHMENT COMPLETE")
    logger.info("Total: %d", len(tracks))
    logger.info("Enriched: %d (%.1f%%)", enriched, enriched/len(tracks)*100)
    logger.info("Unmatched: %d (%.1f%%)", unmatched, unmatched/len(tracks)*100)
    logger.info("Discogs hits: %d", has_discogs)
    logger.info("Last.fm hits: %d", has_lastfm)
    logger.info("MusicBrainz hits: %d", has_mb)
    
    all_genres = []
    for t in tracks:
        genres = t.get("meta", {}).get("genres") or []
        all_genres.extend(genres)
    
    genre_counts = Counter(all_genres)
    logger.info("Top 20 genres:")
    for genre, count in genre_counts.most_common(20):
        logger.info("  %s: %d", genre, count)
    
    # Also show label stats
    labels = [t.get("meta", {}).get("label") for t in tracks if t.get("meta", {}).get("label")]
    label_counts = Counter(labels)
    logger.info("Top 10 labels:")
    for label, count in label_counts.most_common(10):
        logger.info("  %s: %d", label, count)


if __name__ == "__main__":
    main()
