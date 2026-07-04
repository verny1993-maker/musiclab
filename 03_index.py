"""
03_index.py — Index enriched DJ sets into Qdrant for vector search.

Vector (8 dims): [bpm_norm, key_cos, key_sin, energy, danceability, mfcc_1, mfcc_2, mfcc_3]
All dimensions globally normalized to [0, 1] using fixed anchors (stable across new sets).
MFCC: only first 3 coefficients (spectral centroid, brightness, shape) — rest are mastering noise.

Key encoding: camelot field (e.g. '8A', '11B') → circle-of-fifths angle.
  B-ring (major): angle = (N-1)*2π/12
  A-ring (minor): angle = (N-1)*2π/12 + π/12

Normalization anchors (fixed — new sets clamped, not re-fitted):
  bpm:       (value - 60) / 140  →  [0, 1]
  key_cos/sin: already in [-1, 1]
  energy, danceability: already in [0, 1]
  mfcc_1:    (value - MFCC1_MIN) / (MFCC1_MAX - MFCC1_MIN)  →  [0, 1]
  mfcc_2:    (value - MFCC2_MIN) / (MFCC2_MAX - MFCC2_MIN)  →  [0, 1]
  mfcc_3:    (value - MFCC3_MIN) / (MFCC3_MAX - MFCC3_MIN)  →  [0, 1]

Strategy: upsert by set_id+position (idempotent — repeat runs don't duplicate).

Usage:
    python 03_index.py --index --set data/sets/<set_id>.json
    python 03_index.py --rebuild
    python 03_index.py --search N --set data/sets/<set_id>.json
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import math
import sys
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.lib_io import load_json

logger = logging.getLogger("03_index")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "tracks"
VECTOR_SIZE = 8

# ── Fixed normalization anchors (from 58-track global sample) ──
# Padded by ±10% to accommodate future sets. Values outside → clamped.
BPM_MIN, BPM_MAX = 60.0, 200.0  # bpm_norm = (bpm - 60) / 140
MFCC1_MIN, MFCC1_MAX = -358.0, -24.0  # mfcc_1 norm
MFCC2_MIN, MFCC2_MAX = 55.0, 194.0  # mfcc_2 norm
MFCC3_MIN, MFCC3_MAX = -73.0, 63.0  # mfcc_3 norm


def _norm(value: float, vmin: float, vmax: float) -> float:
    """Normalize value to [0, 1] with clamping."""
    if vmax == vmin:
        return 0.5
    return max(0.0, min(1.0, (value - vmin) / (vmax - vmin)))


def camelot_to_angle(camelot: str) -> tuple[float, float]:
    """
    Convert Camelot code to (cos, sin) on the circle of fifths.
    """
    if not camelot or len(camelot) < 2:
        return 0.0, 0.0
    try:
        n = int(camelot[:-1])
        ring = camelot[-1].upper()
    except (ValueError, IndexError):
        return 0.0, 0.0
    if n < 1 or n > 12:
        return 0.0, 0.0
    angle = (n - 1) * 2 * math.pi / 12
    if ring == "A":
        angle += math.pi / 12
    return math.cos(angle), math.sin(angle)


def track_to_vector(track: dict) -> list[float]:
    """Build 8-dim vector from a track's audio_features (all axes [0, 1])."""
    af = track.get("audio", {}) or {}
    bpm = af.get("bpm", 120.0) or 120.0
    camelot = af.get("camelot", "") or ""
    energy = af.get("energy", 0.5) or 0.5
    danceability = af.get("danceability", 0.5) or 0.5
    mfcc = af.get("mfcc_mean", [0.0, 0.0, 0.0]) or [0.0, 0.0, 0.0]

    bpm_norm = max(0.0, min(1.0, (bpm - BPM_MIN) / (BPM_MAX - BPM_MIN)))
    kc, ks = camelot_to_angle(camelot)
    m1 = _norm(mfcc[0], MFCC1_MIN, MFCC1_MAX)
    m2 = _norm(mfcc[1], MFCC2_MIN, MFCC2_MAX)
    m3 = _norm(mfcc[2], MFCC3_MIN, MFCC3_MAX)

    return [bpm_norm, kc, ks, energy, danceability, m1, m2, m3]


def ensure_collection(client: QdrantClient) -> None:
    """Create collection if it doesn't exist."""
    collections = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in collections:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
        logger.info(
            "Created collection '%s' (size=%d, distance=cosine)",
            COLLECTION_NAME,
            VECTOR_SIZE,
        )


def index_set(set_path: str) -> int:
    """Index all tracks from a set JSON. Idempotent via upsert."""
    set_data = load_json(set_path, "set")
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    ensure_collection(client)

    points = []
    for track in set_data["tracks"]:
        af = track.get("audio") or {}
        meta = track.get("meta") or {}

        raw_id = f"{set_data['id']}_{track['position']}"
        point_id = int(hashlib.md5(raw_id.encode()).hexdigest()[:16], 16) % (2**63)

        vector = track_to_vector(track)
        payload = {
            "artist": track.get("artist", ""),
            "title": track.get("title", ""),
            "bpm": af.get("bpm"),
            "camelot": af.get("camelot"),
            "energy": af.get("energy"),
            "danceability": af.get("danceability"),
            "loudness": af.get("loudness"),
            "label": meta.get("label"),
            "set_id": set_data["id"],
            "venue": set_data.get("venue"),
            "position": track["position"],
            "bpm_ambiguous": af.get("bpm_ambiguous", False),
            "id_status": track.get("id_status", "identified"),
            "tracklist_complete": set_data.get("tracklist_complete", True),
        }
        points.append(PointStruct(id=point_id, vector=vector, payload=payload))

    client.upsert(collection_name=COLLECTION_NAME, points=points)
    logger.info("Indexed %d tracks from %s", len(points), set_data["title"])
    return len(points)


def rebuild_index(sets_dir: str = "data/sets") -> int:
    """Delete collection and re-index every enriched set."""
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    client.delete_collection(COLLECTION_NAME)
    ensure_collection(client)
    logger.info("Cleared collection '%s'", COLLECTION_NAME)

    total = 0
    sets_path = Path(sets_dir)
    if sets_path.exists():
        for f in sorted(sets_path.glob("*.json")):
            try:
                n = index_set(str(f))
                total += n
            except Exception as e:
                logger.error("Failed to index %s: %s", f.name, e)
    return total


def search_similar(track_index: int, set_path: str, top_k: int = 5) -> list[dict]:
    """Search for tracks similar to track at `track_index`."""
    set_data = load_json(set_path, "set")
    tracks = set_data["tracks"]
    if track_index < 1 or track_index > len(tracks):
        raise ValueError(f"Track index {track_index} out of range (1–{len(tracks)})")

    query_track = tracks[track_index - 1]
    vector = track_to_vector(query_track)

    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=vector,
        limit=top_k + 1,
    ).points

    similar = []
    for r in results:
        payload = r.payload or {}
        if (
            payload.get("set_id") == set_data["id"]
            and payload.get("position") == track_index
        ):
            continue
        similar.append(
            {
                "score": r.score,
                "artist": payload.get("artist", "?"),
                "title": payload.get("title", "?"),
                "bpm": payload.get("bpm"),
                "camelot": payload.get("camelot"),
                "energy": payload.get("energy"),
                "label": payload.get("label"),
                "set_id": payload.get("set_id", "?"),
                "venue": payload.get("venue"),
            }
        )
        if len(similar) >= top_k:
            break
    return similar


def main():
    parser = argparse.ArgumentParser(
        description="Index DJ sets into Qdrant (8d vectors)"
    )
    parser.add_argument("--set", type=str, help="Path to set JSON")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--index", action="store_true", help="Index the set specified by --set"
    )
    group.add_argument(
        "--rebuild", action="store_true", help="Full index rebuild from data/sets/"
    )
    group.add_argument(
        "--search", type=int, help="Search similar to track position N (requires --set)"
    )
    args = parser.parse_args()

    if args.rebuild:
        total = rebuild_index()
        print(f"[03_index] Rebuild done: {total} points indexed")
    elif args.search:
        if not args.set:
            print("--search requires --set")
            sys.exit(1)
        similar = search_similar(args.search, args.set, top_k=5)
        print(f"Similar to track #{args.search}:")
        for i, s in enumerate(similar):
            sid = s["set_id"]
            if "boiler-room" in sid:
                sn = "BR Tokyo"
            elif "keep-hush" in sid:
                sn = "Keep Hush"
            elif "midnight" in sid:
                sn = "HÖR Berlin"
            else:
                sn = sid[:20]
            print(
                f"  {i + 1}. [{s['score']:.4f}] [{sn}] {s['artist'][:25]} — {s['title'][:25]} BPM={s['bpm']} {s['camelot']}"
            )
    else:
        if not args.set:
            print("--index requires --set")
            sys.exit(1)
        n = index_set(args.set)
        print(f"[03_index] Indexed {n} tracks")


if __name__ == "__main__":
    main()
