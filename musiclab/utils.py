"""
musiclab.utils — Pure functions with zero external dependencies.

Extracted from the pipeline scripts so they can be:
  - Imported without triggering heavy deps (qdrant, librosa, bs4, etc.)
  - Unit-tested in isolation
  - Reused across scripts without copy-paste

All functions are mathematical / string-processing — no I/O, no API calls.
"""

import math
import re
from typing import Optional

# ═══════════════════════════════════════════════════════════════════════
# Normalization
# ═══════════════════════════════════════════════════════════════════════


def norm(value: float, vmin: float, vmax: float) -> float:
    """Normalize value to [0, 1] with clamping. Returns 0.5 when vmin == vmax."""
    if vmax == vmin:
        return 0.5
    return max(0.0, min(1.0, (value - vmin) / (vmax - vmin)))


# ═══════════════════════════════════════════════════════════════════════
# Camelot wheel encoding
# ═══════════════════════════════════════════════════════════════════════


def camelot_to_angle(camelot: Optional[str]) -> tuple[float, float]:
    """
    Convert a Camelot code (e.g. '8A', '11B') to (cos, sin) on the circle of fifths.

    Encoding:
      B-ring (major): angle = (N-1) * 2π/12
      A-ring (minor): angle = (N-1) * 2π/12 + π/12

    Returns (0.0, 0.0) for invalid/empty codes.
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


# Camelot wheel constants — see music-analysis/server.py for the full mapping
KEYS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

_CAMELOT_MAJOR = {
    11: 1,
    6: 2,
    1: 3,
    8: 4,
    3: 5,
    10: 6,
    5: 7,
    0: 8,
    7: 9,
    2: 10,
    9: 11,
    4: 12,
}

_CAMELOT_MINOR = {
    8: 1,
    3: 2,
    10: 3,
    5: 4,
    0: 5,
    7: 6,
    2: 7,
    9: 8,
    4: 9,
    11: 10,
    6: 11,
    1: 12,
}

_PITCH_TO_INDEX = {name: i for i, name in enumerate(KEYS)}
_FLAT_TO_SHARP = {"Ab": "G#", "Bb": "A#", "Db": "C#", "Eb": "D#", "Gb": "F#"}


def pitch_to_camelot(key_name: str, scale: str) -> str:
    """
    Convert a pitch class + scale to Camelot notation.

    Examples:
      pitch_to_camelot("A", "minor") → "8A"
      pitch_to_camelot("C", "major") → "8B"

    Returns "?" for unknown keys.
    """
    key_name = _FLAT_TO_SHARP.get(key_name, key_name)
    idx = _PITCH_TO_INDEX.get(key_name)
    if idx is None:
        return "?"
    if scale == "major":
        num = _CAMELOT_MAJOR.get(idx, 0)
        return f"{num}B" if num else "?"
    else:
        num = _CAMELOT_MINOR.get(idx, 0)
        return f"{num}A" if num else "?"


# ═══════════════════════════════════════════════════════════════════════
# 8D vector construction
# ═══════════════════════════════════════════════════════════════════════

BPM_MIN, BPM_MAX = 60.0, 200.0
MFCC1_MIN, MFCC1_MAX = -358.0, -24.0
MFCC2_MIN, MFCC2_MAX = 55.0, 194.0
MFCC3_MIN, MFCC3_MAX = -73.0, 63.0


def build_8d_vector(
    bpm: Optional[float],
    camelot: Optional[str],
    energy: Optional[float],
    danceability: Optional[float],
    mfcc_mean: Optional[list[float]],
) -> list[float]:
    """
    Build an 8-dimensional embedding vector from audio features.

    Dimensions:
      0: BPM    (normalized 60-200 → 0-1)
      1: Key cos(angle)
      2: Key sin(angle)
      3: Energy (0-1)
      4: Danceability (0-1)
      5: MFCC mean 1 (normalized)
      6: MFCC mean 2 (normalized)
      7: MFCC mean 3 (normalized)
    """
    key_cos, key_sin = camelot_to_angle(camelot)
    mfcc = (mfcc_mean or [0, 0, 0])[:3]
    return [
        round(norm(bpm or 0, BPM_MIN, BPM_MAX), 6),
        round(key_cos, 6),
        round(key_sin, 6),
        round(energy or 0, 6),
        round(danceability or 0, 6),
        round(norm(mfcc[0], MFCC1_MIN, MFCC1_MAX), 6),
        round(norm(mfcc[1], MFCC2_MIN, MFCC2_MAX), 6),
        round(norm(mfcc[2], MFCC3_MIN, MFCC3_MAX), 6),
    ]


def track_to_vector(track: dict) -> list[float]:
    """Build 8D vector from a track dict (with 'audio' sub-dict)."""
    af = track.get("audio", {}) or {}
    bpm = af.get("bpm", 120.0) or 120.0
    camelot = af.get("camelot", "") or ""
    energy = af.get("energy", 0.5) or 0.5
    danceability = af.get("danceability", 0.5) or 0.5
    mfcc = af.get("mfcc_mean", [0.0, 0.0, 0.0]) or [0.0, 0.0, 0.0]

    bpm_norm = max(0.0, min(1.0, (bpm - BPM_MIN) / (BPM_MAX - BPM_MIN)))
    kc, ks = camelot_to_angle(camelot)
    m1 = norm(mfcc[0], MFCC1_MIN, MFCC1_MAX)
    m2 = norm(mfcc[1], MFCC2_MIN, MFCC2_MAX)
    m3 = norm(mfcc[2], MFCC3_MIN, MFCC3_MAX)

    return [bpm_norm, kc, ks, energy, danceability, m1, m2, m3]


# ═══════════════════════════════════════════════════════════════════════
# Parsing utilities
# ═══════════════════════════════════════════════════════════════════════

_SANITIZE_RE = re.compile(r'[&/\\:*?"<>|()\[\]{}—–' "'" "]+")
_SANITIZE_DASH_RE = re.compile(r"-{2,}")


def sanitize_filename(name: str) -> str:
    """Remove characters unsafe for filenames, collapse dashes, truncate to 120."""
    name = _SANITIZE_RE.sub("", name)
    name = _SANITIZE_DASH_RE.sub("-", name)
    return name.strip(" -_.")[:120]


def parse_timecode(tc: str) -> float:
    """Convert 'HH:MM:SS' or 'MM:SS' to seconds."""
    parts = tc.strip().split(":")
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    elif len(parts) == 2:
        return int(parts[0]) * 60 + int(parts[1])
    return 0.0


def split_artist_title(raw: str, *, title_first: bool = False) -> tuple[str, str]:
    """Split 'Artist - Title' (or 'Title - Artist' if title_first) into (artist, title)."""
    raw = raw.strip()
    for sep in (" – ", " - ", "–", "-"):
        if sep in raw:
            left, right = raw.split(sep, 1)
            a, t = (left.strip(), right.strip())
            if title_first:
                a, t = t, a
            return a, t
    return ("", raw)


def slugify(text: str) -> str:
    """Make a filename-safe slug, truncated to 60 chars."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[-\s]+", "-", text)
    return text[:60]
