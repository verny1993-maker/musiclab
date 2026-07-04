"""
musiclab.transitions — DJ transition scoring and chain builder.

Scores the quality of a transition between two tracks using:
  - Camelot wheel compatibility (harmonic mixing rules)
  - BPM ratio (dancefloor-safe ranges)
  - Energy curve direction (building vs cooling)
  - Optional: 8D vector cosine similarity for vibe matching

Then builds optimal DJ set chains using greedy or beam search.
"""

from __future__ import annotations

import math

# ═══════════════════════════════════════════════════════════
# Camelot wheel distance
# ═══════════════════════════════════════════════════════════

def camelot_distance(code_a: str, code_b: str) -> int:
    """
    Compute the harmonic distance between two Camelot codes.

    Distance on the Camelot wheel (circular, 1-12):
      0 = same key (perfect match)
      1 = adjacent (good — one step on wheel)
      2 = two steps (acceptable)
      6 = opposite (tritone — worst)

    Cross-ring transitions (A↔B at same number) are distance 0
    (relative major/minor — they share the same notes).
    """
    try:
        n_a, r_a = int(code_a[:-1]), code_a[-1].upper()
        n_b, r_b = int(code_b[:-1]), code_b[-1].upper()
    except (ValueError, IndexError):
        return 6  # invalid → worst distance

    if n_a < 1 or n_a > 12 or n_b < 1 or n_b > 12:
        return 6

    # Same number, different ring = relative major/minor = distance 0
    if n_a == n_b and r_a != r_b:
        return 0

    # Same number + same ring = identical
    if n_a == n_b:
        return 0

    # Circular distance on 12-position wheel
    diff = abs(n_a - n_b)
    return min(diff, 12 - diff)


def camelot_score(code_a: str, code_b: str) -> float:
    """
    Score a Camelot transition from 0.0 (worst) to 1.0 (perfect).

    Scoring:
      distance 0 (same/relative) → 1.0
      distance 1 (adjacent)      → 0.8
      distance 2                 → 0.4
      distance 3                 → 0.1
      distance 4+                → 0.0
    """
    dist = camelot_distance(code_a, code_b)
    if dist == 0:
        return 1.0
    if dist == 1:
        return 0.8
    if dist == 2:
        return 0.4
    if dist == 3:
        return 0.1
    return 0.0


# ═══════════════════════════════════════════════════════════
# BPM matching score
# ═══════════════════════════════════════════════════════════

def bpm_score(bpm_a: float, bpm_b: float) -> float:
    """
    Score a BPM transition from 0.0 to 1.0.

    DJ mixing rules:
      0–3% difference → 1.0 (barely noticeable, no pitch adjust needed)
      3–6%            → 0.7 (noticeable, still mixable with pitch fader)
      6–8%            → 0.3 (needs key lock + aggressive pitch)
      8%+             → 0.0 (tempo clash, avoid unless intentional)
    """
    if bpm_a <= 0 or bpm_b <= 0:
        return 0.0
    ratio = abs(bpm_a - bpm_b) / max(bpm_a, bpm_b)
    if ratio <= 0.03:
        return 1.0
    if ratio <= 0.06:
        return 0.7
    if ratio <= 0.08:
        return 0.3
    return 0.0


# ═══════════════════════════════════════════════════════════
# Energy direction bonus
# ═══════════════════════════════════════════════════════════

def energy_bonus(energy_a: float, energy_b: float, direction: str = "build") -> float:
    """
    Bonus for energy direction in a DJ set.

    direction="build": reward rising energy (+0.1 per 0.2 step)
    direction="cool": reward falling energy
    direction="peak": highest energy tracks preferred
    direction="neutral": no bonus
    """
    delta = energy_b - energy_a
    if direction == "build":
        return max(0.0, min(0.3, delta * 0.5))
    if direction == "cool":
        return max(-0.3, min(0.0, delta * 0.5))
    if direction == "peak":
        return energy_b * 0.2
    return 0.0


# ═══════════════════════════════════════════════════════════
# Vector similarity (vibe matching)
# ═══════════════════════════════════════════════════════════

def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Cosine similarity between two vectors (0.0 to 1.0)."""
    if len(vec_a) != len(vec_b) or len(vec_a) == 0:
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return max(0.0, dot / (norm_a * norm_b))


# ═══════════════════════════════════════════════════════════
# Transition scoring
# ═══════════════════════════════════════════════════════════

TrackFeatures = dict


def score_transition(
    track_a: TrackFeatures,
    track_b: TrackFeatures,
    direction: str = "build",
    *,
    w_camelot: float = 0.40,
    w_bpm: float = 0.35,
    w_energy: float = 0.15,
    w_vibe: float = 0.10,
) -> dict:
    """
    Score a transition from track_a to track_b.

    Returns a dict with total score (0.0–1.0) and breakdown.
    """
    bpm_a = track_a.get("bpm", 120) or 120
    bpm_b = track_b.get("bpm", 120) or 120
    camelot_a = track_a.get("camelot", "") or ""
    camelot_b = track_b.get("camelot", "") or ""
    energy_a = track_a.get("energy", 0.5) or 0.5
    energy_b = track_b.get("energy", 0.5) or 0.5
    vec_a = track_a.get("vector", [0.5] * 8) or [0.5] * 8
    vec_b = track_b.get("vector", [0.5] * 8) or [0.5] * 8

    cs = camelot_score(camelot_a, camelot_b) if camelot_a and camelot_b else 0.5
    bs = bpm_score(bpm_a, bpm_b)
    eb = energy_bonus(energy_a, energy_b, direction)
    vs = cosine_similarity(vec_a, vec_b)

    total = w_camelot * cs + w_bpm * bs + w_energy * (0.5 + eb) + w_vibe * vs
    total = max(0.0, min(1.0, total))

    return {
        "total": round(total, 4),
        "camelot": round(cs, 4),
        "camelot_distance": camelot_distance(camelot_a, camelot_b),
        "bpm": round(bs, 4),
        "bpm_ratio": round(abs(bpm_a - bpm_b) / max(bpm_a, bpm_b), 4) if max(bpm_a, bpm_b) > 0 else 1.0,
        "energy": round(0.5 + eb, 4),
        "vibe": round(vs, 4),
        "track_a": track_a.get("title", "?")[:40],
        "track_b": track_b.get("title", "?")[:40],
    }


# ═══════════════════════════════════════════════════════════
# Chain builder
# ═══════════════════════════════════════════════════════════

def build_chain(
    start_track: TrackFeatures,
    candidates: list[TrackFeatures],
    chain_length: int = 10,
    direction: str = "build",
    *,
    diversity_penalty: float = 0.0,
    top_k: int = 50,
) -> list[dict]:
    """
    Build a DJ set chain from start_track using greedy best-next selection.

    At each step, scores all remaining candidates and picks the best.
    Never repeats the same track.

    Args:
        start_track: the opening track (dict with bpm, camelot, energy, vector, title)
        candidates: pool of available tracks
        chain_length: how many tracks to select
        direction: "build", "cool", "peak", or "neutral"
        diversity_penalty: penalty for same-artist transitions (0.0 = allow, 1.0 = block)
        top_k: consider only top_k candidates per step (speed optimization)

    Returns:
        list of transition dicts, each with total score and breakdown
    """
    remaining = list(candidates)
    current = start_track
    chain: list[dict] = []

    for _ in range(chain_length):
        if not remaining:
            break

        # Score all transitions
        scored = []
        for candidate in remaining:
            if candidate.get("track_id") == current.get("track_id"):
                continue

            result = score_transition(current, candidate, direction)

            # Artist diversity penalty
            artist_a = (current.get("artist") or "").split(",")[0].strip().lower()
            artist_b = (candidate.get("artist") or "").split(",")[0].strip().lower()
            if diversity_penalty > 0 and artist_a == artist_b:
                result["total"] -= diversity_penalty * result["total"]

            scored.append((result, candidate))

        # Sort by total score descending
        scored.sort(key=lambda x: x[0]["total"], reverse=True)

        # Pick best
        best_result, best_candidate = scored[0]
        chain.append(best_result)

        # Remove chosen track from pool
        remaining = [c for c in remaining if c.get("track_id") != best_candidate.get("track_id")]
        current = best_candidate

    return chain


# ═══════════════════════════════════════════════════════════
# Chain analysis
# ═══════════════════════════════════════════════════════════

def analyze_chain(chain: list[dict]) -> dict:
    """
    Compute aggregate statistics for a built chain.

    Returns:
        dict with mean_score, min_score, max_score, energy_curve, bpm_curve
    """
    if not chain:
        return {"error": "empty chain"}

    scores = [t["total"] for t in chain]
    bpms = [
        t["bpm_ratio"] for t in chain
    ]

    # Energy curve: track how energy evolves across the chain
    energies = [t["energy"] for t in chain]

    return {
        "length": len(chain),
        "mean_score": round(sum(scores) / len(scores), 4),
        "min_score": round(min(scores), 4),
        "max_score": round(max(scores), 4),
        "mean_bpm_ratio": round(sum(bpms) / len(bpms) if bpms else 0, 4),
        "energy_start": energies[0] if energies else 0,
        "energy_end": energies[-1] if energies else 0,
        "energy_trend": "rising" if energies[-1] > energies[0] + 0.1 else (
            "falling" if energies[-1] < energies[0] - 0.1 else "flat"
        ),
    }
