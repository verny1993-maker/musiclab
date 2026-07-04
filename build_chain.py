#!/usr/bin/env python3
"""
build_chain.py — Build a DJ set chain using the transition engine.

Loads analyzed tracks from audio_analysis.db + tracks_enriched_all.json,
scores transitions using Camelot wheel + BPM + energy + vibe, and builds
an optimal chain using greedy best-next selection.

Usage:
    python build_chain.py --start-track-id <spotify_id> --length 10
    python build_chain.py --random --length 8
"""

from __future__ import annotations

import argparse
import json
import random
import sqlite3
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_PROJECT_ROOT))

from musiclab.transitions import analyze_chain, build_chain


def load_tracks(db_path: str, tracks_path: str, limit: int = 0) -> list[dict]:
    """Load tracks with audio features + metadata from SQLite + JSON."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    rows = conn.execute(
        """SELECT track_id, artist, title, bpm, camelot, key_text,
                  energy, danceability, loudness, vector_8d
           FROM audio_analysis
           WHERE vector_8d IS NOT NULL AND bpm IS NOT NULL
           ORDER BY RANDOM()"""
    ).fetchall()

    if limit > 0:
        rows = rows[:limit]

    # Load genres from JSON
    genres_map = {}
    if Path(tracks_path).exists():
        with open(tracks_path, encoding="utf-8") as f:
            enriched = json.load(f)
        for t in enriched:
            meta = t.get("meta") or {}
            genres_map[t["id"]] = (meta.get("genres") or [])[:3]

    tracks = []
    for row in rows:
        vector = json.loads(row["vector_8d"]) if row["vector_8d"] else [0.5] * 8
        tracks.append(
            {
                "track_id": row["track_id"],
                "artist": row["artist"],
                "title": row["title"],
                "bpm": row["bpm"],
                "camelot": row["camelot"],
                "key": row["key_text"],
                "energy": row["energy"] or 0.5,
                "danceability": row["danceability"] or 0.5,
                "vector": vector,
                "genres": genres_map.get(row["track_id"], []),
            }
        )

    conn.close()
    return tracks


def main():
    parser = argparse.ArgumentParser(description="Build a DJ set chain")
    parser.add_argument("--start-track-id", help="Spotify track ID to start from")
    parser.add_argument(
        "--start-artist", help="Artist name to search for starting track"
    )
    parser.add_argument(
        "--random", action="store_true", help="Pick random starting track"
    )
    parser.add_argument("--length", type=int, default=10, help="Chain length")
    parser.add_argument(
        "--direction", default="build", choices=["build", "cool", "peak", "neutral"]
    )
    parser.add_argument(
        "--diversity",
        type=float,
        default=0.3,
        help="Artist diversity penalty (0=none, 1=max)",
    )
    parser.add_argument(
        "--pool-size", type=int, default=500, help="Number of candidate tracks to load"
    )
    parser.add_argument(
        "--top-k", type=int, default=50, help="Top-K candidates per step"
    )
    parser.add_argument(
        "--db",
        default=str(_PROJECT_ROOT / "data" / "library" / "audio_analysis.db"),
    )
    parser.add_argument(
        "--tracks",
        default=str(_PROJECT_ROOT / "data" / "library" / "tracks_enriched_all.json"),
    )
    args = parser.parse_args()

    # Load tracks
    print(f"Loading {args.pool_size} tracks from library...")
    tracks = load_tracks(args.db, args.tracks, limit=args.pool_size)
    print(f"Loaded {len(tracks)} tracks with audio features")

    # Select starting track
    if args.random:
        start = random.choice(tracks)
    elif args.start_artist:
        matches = [
            t
            for t in tracks
            if args.start_artist.lower() in (t["artist"] or "").lower()
        ]
        if not matches:
            print(f"No tracks found for artist '{args.start_artist}'")
            sys.exit(1)
        start = random.choice(matches)
    elif args.start_track_id:
        matches = [t for t in tracks if t["track_id"] == args.start_track_id]
        if not matches:
            print(f"Track ID '{args.start_track_id}' not found in pool")
            sys.exit(1)
        start = matches[0]
    else:
        start = tracks[0]

    # Remove start track from candidates
    candidates = [t for t in tracks if t["track_id"] != start["track_id"]]

    print(f"\nStarting track: {start['artist']} — {start['title']}")
    print(f"  BPM: {start['bpm']}  Key: {start['camelot']} ({start['key']})")
    print(f"  Energy: {start['energy']:.3f}  Dance: {start['danceability']:.3f}")
    if start["genres"]:
        print(f"  Genres: {', '.join(start['genres'])}")

    # Build chain
    print(f"\nBuilding {args.length}-track chain (direction={args.direction})...")
    chain = build_chain(
        start,
        candidates,
        chain_length=args.length,
        direction=args.direction,
        diversity_penalty=args.diversity,
        top_k=args.top_k,
    )

    # Print chain
    print(f"\n{'=' * 70}")
    print(
        f"{'#':>3} {'Score':>6} {'C':>5} {'BPM':>6} {'Energy':>8} {'Artist':<25} Title"
    )
    print(f"{'=' * 70}")

    print(
        f"{'1':>3} {'—':>6} {'—':>5} {start['bpm']:>6.1f} {start['energy']:>8.3f} "
        f"{start['artist'][:24]:<25} {start['title'][:35]}"
    )

    for i, transition in enumerate(chain):
        score = transition["total"]
        cs = transition["camelot"]
        bpms = transition["bpm"]
        energy = transition["energy"]
        artist = (transition.get("track_b_artist") or transition["track_b"])[:24]
        title = transition["track_b"][:35]

        # Score color indicators
        "█" * int(score * 10)
        print(
            f"{i + 2:>3} {score:>5.3f} {cs:>5.2f} {bpms:>5.2f} {energy:>8.3f} "
            f"{artist:<25} {title}"
        )

    # Analyze chain
    print(f"\n{'=' * 70}")
    stats = analyze_chain(chain)
    print("Chain stats:")
    print(f"  Mean score: {stats['mean_score']:.3f}")
    print(f"  Score range: {stats['min_score']:.3f} — {stats['max_score']:.3f}")
    print(f"  Mean BPM ratio: {stats['mean_bpm_ratio']:.3f}")
    print(
        f"  Energy: {stats['energy_start']:.3f} → {stats['energy_end']:.3f} ({stats['energy_trend']})"
    )


if __name__ == "__main__":
    main()
