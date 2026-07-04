"""
05_build_cards.py — Build artist and venue cards from enriched sets.

Usage:
    python 05_build_cards.py --artist <slug>
    python 05_build_cards.py --all
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.lib_io import load_json, write_json

logger = logging.getLogger("05_build_cards")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")

DATA_SETS = Path("data/sets")
DATA_ARTISTS = Path("data/artists")
DATA_VENUES = Path("data/venues")


def collect_tracks(sets_dir: Path = DATA_SETS) -> list[dict]:
    """Load all enriched sets and return a flat list of tracks with set context."""
    all_tracks = []
    if sets_dir.exists():
        for f in sorted(sets_dir.glob("*.json")):
            try:
                s = load_json(str(f), "set")
            except Exception:
                continue
            for t in s["tracks"]:
                af = t.get("audio") or {}
                meta = t.get("meta") or {}
                all_tracks.append({
                    "artist": s["artist"],
                    "venue": s.get("venue"),
                    "set_id": s["id"],
                    "set_title": s["title"],
                    "track_artist": t.get("artist", ""),
                    "track_title": t.get("title", ""),
                    "position": t["position"],
                    "bpm": af.get("bpm"),
                    "camelot": af.get("camelot"),
                    "energy": af.get("energy"),
                    "danceability": af.get("danceability"),
                    "loudness": af.get("loudness"),
                    "key_strength": af.get("key_strength"),
                    "label": meta.get("label"),
                    "year": meta.get("year"),
                    "genres": meta.get("genres"),
                    "bpm_ambiguous": af.get("bpm_ambiguous", False),
                    "id_status": t.get("id_status", "identified"),
                })
    return all_tracks


def build_artist_card(artist_slug: str, all_tracks: list[dict]) -> dict:
    """Build a computed artist card from track data."""
    tracks = [t for t in all_tracks if t["artist"].lower().replace(" ", "-") == artist_slug.lower().replace(" ", "-")
              or artist_slug.lower() in t["artist"].lower()]

    if not tracks:
        # Try fuzzy match
        artist_key = artist_slug.lower().replace(" ", "")
        tracks = [t for t in all_tracks if artist_key in t["artist"].lower().replace(" ", "")]
        if not tracks:
            raise ValueError(f"No tracks found for artist '{artist_slug}'")

    bpms = [t["bpm"] for t in tracks if t["bpm"] is not None
            and t.get("id_status") != "unidentified"
            and not t.get("bpm_ambiguous")]
    energies = [t["energy"] for t in tracks if t["energy"] is not None]
    camelots = [t["camelot"] for t in tracks if t["camelot"]]
    labels = [t["label"] for t in tracks if t["label"]]

    set_ids = sorted(set(t["set_id"] for t in tracks))
    n_sets = len(set_ids)

    # Top camelot
    camelot_counter = Counter(c for c in camelots if c)
    top_camelot = [{"code": c, "count": n} for c, n in camelot_counter.most_common(5)]

    # Top labels
    label_counter = Counter(l for l in labels if l)
    top_labels = [{"label": l, "count": n} for l, n in label_counter.most_common(5)]

    card: dict[str, Any] = {
        "id": artist_slug,
        "name": tracks[0]["artist"],
        "manual": {},
        "computed": {
            "n_sets_analyzed": n_sets,
            "n_tracks_analyzed": len(tracks),
            "mean_bpm": round(statistics.mean(bpms), 1) if bpms else None,
            "bpm_range": [round(min(bpms), 1), round(max(bpms), 1)] if bpms else None,
            "bpm_std": round(statistics.stdev(bpms), 1) if len(bpms) > 1 else 0.0,
            "mean_energy": round(statistics.mean(energies), 4) if energies else None,
            "energy_std": round(statistics.stdev(energies), 4) if len(energies) > 1 else 0.0,
            "top_camelot": top_camelot,
            "top_labels": top_labels,
            "set_ids": set_ids,
            "computed_at": __import__("datetime").datetime.utcnow().isoformat() + "Z",
        },
    }
    return card


def build_venue_card(venue_name: str, all_tracks: list[dict]) -> dict:
    """Build a venue card from track data."""
    venue_tracks = [t for t in all_tracks if t.get("venue") and t["venue"].lower() == venue_name.lower()]

    if not venue_tracks:
        raise ValueError(f"No tracks found for venue '{venue_name}'")

    artists = sorted(set(t["artist"] for t in venue_tracks))
    bpms = [t["bpm"] for t in venue_tracks if t["bpm"] is not None]
    energies = [t["energy"] for t in venue_tracks if t["energy"] is not None]
    camelots = [t["camelot"] for t in venue_tracks if t["camelot"]]
    set_ids = sorted(set(t["set_id"] for t in venue_tracks))

    camelot_counter = Counter(c for c in camelots if c)
    top_camelot = [{"code": c, "count": n} for c, n in camelot_counter.most_common(5)]

    card = {
        "id": venue_name.lower().replace(" ", "-"),
        "name": venue_name,
        "n_sets": len(set_ids),
        "n_tracks": len(venue_tracks),
        "artists": artists,
        "mean_bpm": round(statistics.mean(bpms), 1) if bpms else None,
        "bpm_range": [round(min(bpms), 1), round(max(bpms), 1)] if bpms else None,
        "mean_energy": round(statistics.mean(energies), 4) if energies else None,
        "top_camelot": top_camelot,
        "set_ids": set_ids,
        "computed_at": __import__("datetime").datetime.utcnow().isoformat() + "Z",
    }
    return card


def build_all() -> dict[str, int]:
    """Build all artist and venue cards from data/sets/."""
    all_tracks = collect_tracks()
    counts = {"artists": 0, "venues": 0}

    # Group by artist
    artists: dict[str, list[dict]] = {}
    venues: dict[str, list[dict]] = {}
    for t in all_tracks:
        a = t["artist"]
        artists.setdefault(a, []).append(t)
        v = t.get("venue")
        if v:
            venues.setdefault(v, []).append(t)

    DATA_ARTISTS.mkdir(parents=True, exist_ok=True)
    DATA_VENUES.mkdir(parents=True, exist_ok=True)

    for artist_name, _ in artists.items():
        slug = artist_name.lower().replace(" ", "-")
        card = build_artist_card(slug, all_tracks)
        path = DATA_ARTISTS / f"{slug}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(card, f, indent=2, ensure_ascii=False)
        counts["artists"] += 1
        logger.info("Artist: %s (%d tracks, %d sets)", artist_name, card["computed"]["n_tracks_analyzed"], card["computed"]["n_sets_analyzed"])

    for venue_name, _ in venues.items():
        slug = venue_name.lower().replace(" ", "-")
        card = build_venue_card(venue_name, all_tracks)
        path = DATA_VENUES / f"{slug}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(card, f, indent=2, ensure_ascii=False)
        counts["venues"] += 1
        logger.info("Venue: %s (%d tracks)", venue_name, card["n_tracks"])

    return counts


def main():
    parser = argparse.ArgumentParser(description="Build artist/set cards")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--artist", type=str, help="Artist slug")
    group.add_argument("--all", action="store_true", help="Build all cards")
    args = parser.parse_args()

    if args.all:
        counts = build_all()
        print(f"[05_build_cards] Done: {counts['artists']} artists, {counts['venues']} venues")
    elif args.artist:
        all_tracks = collect_tracks()
        card = build_artist_card(args.artist, all_tracks)
        slug = args.artist.lower().replace(" ", "-")
        DATA_ARTISTS.mkdir(parents=True, exist_ok=True)
        with open(DATA_ARTISTS / f"{slug}.json", "w", encoding="utf-8") as f:
            json.dump(card, f, indent=2, ensure_ascii=False)
        print(json.dumps(card, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
