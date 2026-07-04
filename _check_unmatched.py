"""Quick check: how many unmatched tracks and sample them."""
import json
from pathlib import Path

tracks = json.loads(Path("data/library/tracks_enriched_all.json").read_text(encoding="utf-8"))
unmatched = [t for t in tracks if t.get("meta", {}).get("meta_status") == "unmatched"]
print(f"Total tracks: {len(tracks)}")
print(f"Unmatched: {len(unmatched)} ({len(unmatched)/len(tracks)*100:.1f}%)")
print()
print("First 15 unmatched:")
for i, t in enumerate(unmatched[:15]):
    print(f"  {i+1}. {t['artist']} — {t['name']}")
