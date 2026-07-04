# MusicLab — DJ Set Analysis Pipeline

[![CI](https://github.com/verny1993-maker/musiclab/actions/workflows/ci.yml/badge.svg)](https://github.com/verny1993-maker/musiclab/actions)

**Parse, analyze, enrich, and build artist intelligence from DJ sets and music libraries.**

MusicLab is a multi-stage pipeline that takes DJ set tracklists, downloads the audio, extracts musical features (BPM, key, Camelot, energy, danceability, MFCC, beat grid), enriches with metadata from 5 external APIs, builds 8D embedding vectors for similarity search, and generates artist/venue intelligence cards.

---

## Architecture

Open [`architecture.html`](architecture.html) in any browser for an interactive dark-themed diagram.

[→ Download SVG diagram](architecture.html)

```
set79.com ──→ 01_parse ──→ 02_enrich_audio ──→ 02_enrich_meta ──→ 03_enrich_library ──→ 05_build_cards
                  │              │                    │                    │
                  ▼              ▼                    ▼                    ▼
             JSON sets    Docker API :8777     Discogs/Last.fm/MB     SQLite + Qdrant
```

---

## Installation

```bash
# From GitHub (recommended)
pip install git+https://github.com/verny1993-maker/musiclab.git

# Or clone + editable install
git clone https://github.com/verny1993-maker/musiclab.git
cd musiclab
pip install -e ".[dev]"
```

After installation, use the CLI commands:

```bash
musiclab-parse --url "https://set79.com/tracklist/..." --artist "DJ Name"
musiclab-enrich-audio --set data/sets/set_<id>.json
musiclab-enrich-meta --set data/sets/set_<id>.json
musiclab-enrich-library --workers 1 --resume
musiclab-build-cards --all
```

Or import programmatically:

```python
from musiclab.utils import camelot_to_angle, pitch_to_camelot, build_8d_vector

cos_a, sin_a = camelot_to_angle("8A")
camelot = pitch_to_camelot("A", "minor")  # → "8A"
vector = build_8d_vector(128, "8A", 0.7, 0.65, [-140.2, 85.3, -12.1])
```

---

## Pipeline Stages

| # | Script | Input | Output | Description |
|---|--------|-------|--------|-------------|
| 1 | `01_parse.py` | set79.com URL | `data/sets/*.json` | Parse tracklist HTML: artist, title, timecode, SC/YT URLs |
| 2 | `02_enrich_audio.py` | set JSON | set JSON (with audio features) | Download audio (SC→YT fallback), slice by timecodes, analyze via Docker API |
| 3 | `enrich_library.py` | tracks JSON | `tracks_enriched.json` | Multi-source metadata: Discogs (label/year/genres), Last.fm (tags), MusicBrainz (tags) |
| 4 | `03_enrich_audio_library.py` | tracks JSON | `audio_analysis.db` | Batch download + analyze 2388 tracks, build 8D vectors, SQLite with resume |
| 5 | `05_build_cards.py` | enriched sets | `data/artists/*.json`, `data/venues/*.json` | Compute artist/venue stats: mean BPM, energy range, top Camelot, top labels |

---

## Quick Start

```bash
# 1. Parse a DJ set from set79
python 01_parse.py --url "https://set79.com/tracklist/soundcloud.com/user/slug" --artist "DJ Name"

# 2. Enrich audio (download + analyze each track)
python 02_enrich_audio.py --set data/sets/set_<id>.json

# 3. Enrich metadata (label, genres, year)
python enrich_library.py --input data/library/tracks_all.json

# 4. Batch analyze your Spotify library
python 03_enrich_audio_library.py --workers 1 --resume

# 5. Build artist/venue intelligence cards
python 05_build_cards.py --all
```

**Prerequisites:**
- Docker running with `music-analysis` container on port 8777
- Python 3.11+ with dependencies: `librosa`, `soundfile`, `requests`, `beautifulsoup4`, `numpy`
- `yt-dlp` and `ffmpeg` installed
- Environment variables in `.env`: `DISCOGS_TOKEN`, `LASTFM_API_KEY`, `SPOTIFY_ACCESS_TOKEN`

---

## Music Analysis API

The core audio analysis runs as a Docker service (`music-analysis/server.py`):

```
POST /analyze  { "filepath": "/tmp/analysis/track.wav" }
```

**Returns:**
```json
{
  "bpm": 128.5,
  "key": "A",
  "camelot": "8A",
  "key_strength": 0.87,
  "energy": 0.72,
  "danceability": 0.65,
  "loudness": -8.3,
  "duration": 234.5,
  "mfcc_mean": [-140.2, 85.3, -12.1],
  "spectral_centroid": 2100.5,
  "zero_crossing_rate": 0.08,
  "beat_positions": [0.0, 0.47, 0.94, ...]
}
```

**Features:**
- **BPM**: librosa beat tracker
- **Key + Camelot**: essentia KeyExtractor with Camelot wheel mapping (1A–12B)
- **Danceability**: essentia Danceability algorithm
- **Energy**: RMS energy normalized to 0–1
- **Beat Grid**: essentia BeatTrackerDegara for precise beat positions

---

## 8D Vector Embedding

Each track is encoded as an 8-dimensional vector for similarity search:

| Dim | Feature | Range |
|-----|---------|-------|
| 0 | BPM (normalized) | 60–200 → 0–1 |
| 1 | Key cos(angle) | Camelot → unit circle |
| 2 | Key sin(angle) | Camelot → unit circle |
| 3 | Energy | 0–1 |
| 4 | Danceability | 0–1 |
| 5 | MFCC mean 1 (normalized) | −358…−24 → 0–1 |
| 6 | MFCC mean 2 (normalized) | 55–194 → 0–1 |
| 7 | MFCC mean 3 (normalized) | −73–63 → 0–1 |

Vectors are stored in SQLite and indexed in Qdrant for nearest-neighbor search.

---

## DJ Transition Engine

`musiclab/transitions.py` scores the quality of a DJ transition between two tracks:

| Factor | Weight | Algorithm |
|--------|--------|-----------|
| **Camelot (harmonic)** | 40% | Distance on Camelot wheel (0=perfect, 1=good, 4+=avoid) |
| **BPM matching** | 35% | Ratio scoring: <3%=perfect, 3-6%=good, 6-8%=acceptable |
| **Energy direction** | 15% | Bonus for building/cooling/peak energy curves |
| **Vibe similarity** | 10% | Cosine similarity on 8D vectors |

Build a DJ set chain from any starting track:

```bash
python build_chain.py --random --length 10 --direction build --diversity 0.3
```

```python
from musiclab.transitions import build_chain, score_transition

chain = build_chain(start_track, candidates, chain_length=10, direction="build")
for t in chain:
    print(f"{t['total']:.3f} | Camelot={t['camelot']} | BPM={t['bpm']:.2f}")
```

---

## Genre Classifier (PyTorch)

`musiclab/classifier.py` — 3-layer MLP that classifies tracks into genres from 8D vectors:

```bash
pip install musiclab[ml]
python train_genre_classifier.py --epochs 50 --genres 8
```

```python
from musiclab.classifier import GenreClassifier, predict

model = GenreClassifier(num_genres=8)
# Load checkpoint: torch.load("models/genre_classifier.pt")
genres = predict(model, vector, genre_names)
```

1209 tracks trained on 8 genres (electronic, deep house, ambient, electro, acid, drum n bass, disco, breakbeat). 2920 parameters, trains in <1 min on CPU.

---

## Deployment

See [`deploy/README.md`](deploy/README.md) for full instructions.

| Option | Cost | Complexity |
|--------|------|------------|
| **AWS Lightsail** (Terraform) | ~$7/mo | Medium — `deploy/terraform/main.tf` |
| **GCP Cloud Run** | Free tier | Low — `gcloud run deploy` |
| **Local Docker** | Free | Minimal — `docker run -p 8777:8777` |

---

## Tech Stack

| Category | Technology |
|----------|-----------|
| Language | Python 3.11+ |
| Audio DSP | librosa, essentia |
| Machine Learning | PyTorch |
| API Framework | FastAPI |
| Containerization | Docker |
| Database | SQLite (WAL mode) |
| Vector Search | Qdrant |
| Infrastructure | Terraform (AWS) |
| Download | yt-dlp, FFmpeg |
| Parsing | BeautifulSoup4 |
| Metadata APIs | Discogs, Last.fm, MusicBrainz, Spotify |
| Rate Limiting | Token bucket per API, exponential backoff |

---

## Data

| Entity | Count | Storage |
|--------|-------|---------|
| Spotify liked tracks | 2,388 | JSON |
| Tracks analyzed (audio) | ~1,700 | SQLite |
| Tracks enriched (metadata) | ~2,200 | JSON |
| DJ sets parsed | 5 | JSON |
| Artist cards | 4 | JSON |
| Venue cards | 4 | JSON |

---

## Project Structure

```
AudioLab/
├── 01_parse.py                # Parse set79 tracklists
├── 02_enrich_audio.py         # Download + analyze set audio
├── 02_enrich_meta.py          # Multi-source metadata enrichment
├── 03_enrich_audio_library.py # Batch library analysis + 8D vectors
├── 03_index.py                # Qdrant vector indexing
├── 05_build_cards.py          # Artist & venue intelligence cards
├── enrich_library.py          # Discogs + Last.fm + MusicBrainz metadata
├── build_chain.py             # DJ set chain builder
├── train_genre_classifier.py  # Genre classifier training script
├── musiclab/
│   ├── __init__.py
│   ├── utils.py               # Pure functions: Camelot, vectors, parsing
│   ├── transitions.py         # DJ transition scoring + chain builder
│   ├── classifier.py          # PyTorch genre classifier (3-layer MLP)
│   └── cli.py                 # CLI entry points
├── tests/
│   ├── test_vectors.py        # 35 tests
│   ├── test_parsing.py        # 16 tests
│   ├── test_camelot.py        # 12 tests
│   ├── test_classifier.py     # 9 tests (PyTorch)
│   ├── test_transitions.py    # 43 tests (DJ transitions)
│   └── test_analysis.py       # 7 integration tests
├── lib/
│   ├── rate_limits.py         # Rate limiting for 5 APIs
│   └── lib_io.py              # JSON Schema validation
├── deploy/
│   ├── Dockerfile             # Production Docker image
│   ├── server.py              # FastAPI audio analysis service
│   ├── terraform/main.tf      # AWS Lightsail deployment
│   └── README.md              # Deployment guide
├── models/
│   └── genre_classifier.pt    # Trained PyTorch checkpoint
├── scripts/                   # Utility scripts
├── data/                      # Sample data + schemas
├── .github/workflows/ci.yml   # CI: test + lint + format + build
├── pyproject.toml             # Package config
├── .env.example               # Environment variables template
├── architecture.html          # Architecture diagram
└── README.md
```

---

## Design Decisions

- **Idempotent by default**: every stage checks if data exists before reprocessing
- **Resilience**: 3-attempt retry on API calls, Docker restart on memory leak, SC→YT fallback
- **Rate limiting**: single gate (`lib/rate_limits.py`) for all 5 APIs with token bucket
- **Data contracts**: all JSON validated against JSON Schema in `lib/lib_io.py`
- **Windows-aware**: file-lock retry loops for yt-dlp/FFmpeg cleanup
- **Source attribution**: every metadata field tracks its origin (discogs/lastfm/musicbrainz)

---

## License

MIT
