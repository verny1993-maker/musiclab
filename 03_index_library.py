"""
03_index_library.py — Index Spotify liked tracks into Qdrant collection 'spotify_liked'.
8D vectors with MusicLab fixed anchors. Idempotent (upsert by MD5 ID).
"""

import hashlib
import json
import logging
import sqlite3
import sys
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

sys.path.insert(0, str(Path(__file__).resolve().parent))

logger = logging.getLogger("index_library")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")

QDRANT_HOST, QDRANT_PORT = "localhost", 6333
COLLECTION = "spotify_liked"
VECTOR_SIZE = 8


def main():
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    # Create collection
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION not in existing:
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
        logger.info("Created collection '%s'", COLLECTION)
    else:
        logger.info("Collection '%s' exists", COLLECTION)

    # Load audio vectors
    db = sqlite3.connect("data/library/audio_analysis.db")
    rows = db.execute("""
        SELECT track_id, artist, title, bpm, camelot, key_text, energy, danceability,
               mfcc_mean, vector_8d, source_url, source_type, loudness
        FROM audio_analysis WHERE vector_8d IS NOT NULL
    """).fetchall()
    db.close()

    # Load metadata
    with open("data/library/tracks_all_enriched.json", encoding="utf-8") as f:
        meta_tracks = json.load(f)
    meta_map = {t["id"]: t.get("meta", {}) for t in meta_tracks}

    points = []
    for row in rows:
        (
            tid,
            artist,
            title,
            bpm,
            camelot,
            key_text,
            energy,
            dance,
            mfcc_json,
            vec_json,
            url,
            src,
            loudness,
        ) = row
        vector = json.loads(vec_json)
        json.loads(mfcc_json) if mfcc_json else []
        meta = meta_map.get(tid, {})

        point_id = hashlib.md5(tid.encode()).hexdigest()

        payload = {
            "spotify_id": tid,
            "artist": artist,
            "title": title,
            "bpm": bpm,
            "camelot": camelot,
            "key": key_text,
            "energy": energy,
            "danceability": dance,
            "loudness": loudness,
            "source": src,
            "source_url": url,
            "label": meta.get("label"),
            "genres": meta.get("genres", []),
            "year": meta.get("year"),
            "meta_sources": list(meta.get("sources", {}).keys()),
        }

        points.append(PointStruct(id=point_id, vector=vector, payload=payload))

    # Upsert in batches
    batch = 100
    for i in range(0, len(points), batch):
        chunk = points[i : i + batch]
        client.upsert(collection_name=COLLECTION, points=chunk, wait=True)
        logger.info("Upserted %d/%d points", i + len(chunk), len(points))

    # Verify
    info = client.get_collection(COLLECTION)
    logger.info("Collection '%s': %d vectors", COLLECTION, info.points_count)

    # Sample search
    if points:
        sample = points[0]
        results = client.search(
            collection_name=COLLECTION,
            query_vector=sample.vector,
            limit=3,
        )
        logger.info("Sample search for '%s':", sample.payload["title"])
        for r in results:
            logger.info(
                "  %s — %s (score=%.4f)",
                r.payload["artist"],
                r.payload["title"],
                r.score,
            )

    logger.info("Done! %d tracks indexed", len(points))


if __name__ == "__main__":
    main()
