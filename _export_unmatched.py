"""Extract unmatched track names for MusicBrainz batch enrichment."""

import json
from pathlib import Path

tracks = json.loads(
    Path("data/library/tracks_enriched_all.json").read_text(encoding="utf-8")
)
unmatched = [t for t in tracks if t.get("meta", {}).get("meta_status") == "unmatched"]

# Output as JSON lines for easy parsing
for i, t in enumerate(unmatched):
    print(
        json.dumps(
            {"idx": i, "artist": t["artist"], "name": t["name"]}, ensure_ascii=False
        )
    )
