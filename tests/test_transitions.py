"""
test_transitions.py — Unit tests for DJ transition scoring and chain builder.

Tests camelot_distance, camelot_score, bpm_score, score_transition,
build_chain, and analyze_chain.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from musiclab.transitions import (
    analyze_chain,
    bpm_score,
    build_chain,
    camelot_distance,
    camelot_score,
    cosine_similarity,
    energy_bonus,
    score_transition,
)

# ═══════════════════════════════════════════════════════════
# camelot_distance
# ═══════════════════════════════════════════════════════════

class TestCamelotDistance:
    def test_same_key(self):
        assert camelot_distance("8A", "8A") == 0

    def test_relative_major_minor(self):
        """8A (A minor) and 8B (C major) share same pitch class."""
        assert camelot_distance("8A", "8B") == 0

    def test_adjacent_same_ring(self):
        assert camelot_distance("8A", "9A") == 1

    def test_two_steps(self):
        assert camelot_distance("8A", "10A") == 2

    def test_tritone(self):
        """6 steps apart = opposite on wheel."""
        assert camelot_distance("1A", "7A") == 6

    def test_circular_wrap(self):
        """12A → 1A should be distance 1 (circular)."""
        assert camelot_distance("12A", "1A") == 1

    def test_invalid_code(self):
        assert camelot_distance("", "8A") == 6

    def test_out_of_range(self):
        assert camelot_distance("13A", "8A") == 6

    def test_all_distances(self):
        """Distance matrix for 8A: every distance should appear."""
        targets = {
            "8A": 0, "8B": 0,
            "7A": 1, "9A": 1,
            "6A": 2, "10A": 2,
            "5A": 3, "11A": 3,
            "4A": 4, "12A": 4,
        }
        for code, expected in targets.items():
            assert camelot_distance("8A", code) == expected, f"8A→{code}"


# ═══════════════════════════════════════════════════════════
# camelot_score
# ═══════════════════════════════════════════════════════════

class TestCamelotScore:
    def test_perfect(self):
        assert camelot_score("8A", "8B") == 1.0

    def test_good(self):
        assert camelot_score("8A", "9A") == 0.8

    def test_acceptable(self):
        assert camelot_score("8A", "10A") == 0.4

    def test_poor(self):
        assert camelot_score("8A", "11A") == 0.1

    def test_bad(self):
        assert camelot_score("8A", "12A") == 0.0


# ═══════════════════════════════════════════════════════════
# bpm_score
# ═══════════════════════════════════════════════════════════

class TestBpmScore:
    def test_perfect(self):
        assert bpm_score(128, 128) == 1.0
        assert bpm_score(128, 130) == 1.0  # 1.5%

    def test_good(self):
        assert bpm_score(128, 134) == 0.7  # ~4.7%

    def test_acceptable(self):
        assert bpm_score(128, 137) == pytest.approx(0.3, abs=0.01)  # ~7%

    def test_bad(self):
        assert bpm_score(128, 160) == 0.0  # 25%

    def test_zero_bpm(self):
        assert bpm_score(0, 128) == 0.0
        assert bpm_score(128, 0) == 0.0

    def test_boundary_3pct(self):
        """Exactly 3% should still be 1.0."""
        assert bpm_score(100, 103) == 1.0


# ═══════════════════════════════════════════════════════════
# energy_bonus
# ═══════════════════════════════════════════════════════════

class TestEnergyBonus:
    def test_build_rising(self):
        bonus = energy_bonus(0.3, 0.7, "build")
        assert bonus > 0

    def test_build_falling(self):
        bonus = energy_bonus(0.7, 0.3, "build")
        assert bonus == 0.0

    def test_cool_rising(self):
        """Cooling mode: rising energy gets no bonus."""
        bonus = energy_bonus(0.3, 0.7, "cool")
        assert bonus == 0.0  # wrong direction, no bonus

    def test_cool_falling(self):
        """Cooling mode: falling energy gets positive bonus (as negative delta)."""
        bonus = energy_bonus(0.7, 0.3, "cool")
        assert bonus < 0  # energy drops, cooling bonus applies

    def test_neutral(self):
        assert energy_bonus(0.3, 0.7, "neutral") == 0.0

    def test_peak(self):
        bonus_high = energy_bonus(0.5, 0.9, "peak")
        bonus_low = energy_bonus(0.5, 0.3, "peak")
        assert bonus_high > bonus_low


# ═══════════════════════════════════════════════════════════
# cosine_similarity
# ═══════════════════════════════════════════════════════════

class TestCosineSimilarity:
    def test_identical(self):
        assert cosine_similarity([1, 0, 0], [1, 0, 0]) == pytest.approx(1.0)

    def test_orthogonal(self):
        assert cosine_similarity([1, 0], [0, 1]) == 0.0

    def test_opposite(self):
        sim = cosine_similarity([1, 0], [-1, 0])
        assert sim == 0.0

    def test_empty(self):
        assert cosine_similarity([], []) == 0.0

    def test_zero_vector(self):
        assert cosine_similarity([0, 0], [1, 0]) == 0.0


# ═══════════════════════════════════════════════════════════
# score_transition
# ═══════════════════════════════════════════════════════════

class TestScoreTransition:
    def _track(self, bpm=128, camelot="8A", energy=0.5, vector=None):
        vec = vector or [0.5] * 8
        return {"bpm": bpm, "camelot": camelot, "energy": energy, "vector": vec, "title": "Test"}

    def test_perfect_transition(self):
        a = self._track(128, "8A", 0.5)
        b = self._track(128, "8A", 0.5)
        result = score_transition(a, b)
        assert result["total"] > 0.8  # near-perfect

    def test_bad_transition(self):
        a = self._track(128, "8A", 0.5)
        b = self._track(160, "2A", 0.5)
        result = score_transition(a, b)
        assert result["total"] < 0.5

    def test_missing_camelot(self):
        a = self._track(128, "", 0.5)
        b = self._track(128, "8A", 0.5)
        result = score_transition(a, b)
        assert 0.0 <= result["total"] <= 1.0

    def test_returns_breakdown(self):
        a = self._track(128, "8A", 0.5)
        b = self._track(128, "8A", 0.5)
        result = score_transition(a, b)
        for key in ("camelot", "bpm", "energy", "vibe", "camelot_distance", "bpm_ratio"):
            assert key in result, f"Missing key: {key}"


# ═══════════════════════════════════════════════════════════
# build_chain
# ═══════════════════════════════════════════════════════════

class TestBuildChain:
    def _make_track(self, idx, bpm=128, camelot="8A", energy=0.5, artist="Artist"):
        return {
            "track_id": f"track_{idx}",
            "artist": artist,
            "title": f"Track {idx}",
            "bpm": bpm + idx * 2,
            "camelot": camelot,
            "energy": energy,
            "vector": [0.5] * 8,
        }

    def test_builds_chain(self):
        start = self._make_track(0, bpm=128, camelot="8A")
        candidates = [self._make_track(i, bpm=128 + i, camelot="8A") for i in range(1, 20)]
        chain = build_chain(start, candidates, chain_length=5)

        assert len(chain) == 5
        for t in chain:
            assert "total" in t
            assert 0.0 <= t["total"] <= 1.0

    def test_no_duplicates(self):
        start = self._make_track(0)
        candidates = [self._make_track(i) for i in range(1, 5)]
        chain = build_chain(start, candidates, chain_length=3)

        seen_titles = {start["title"]}
        for t in chain:
            assert t["track_b"] not in seen_titles
            seen_titles.add(t["track_b"])

    def test_empty_candidates(self):
        start = self._make_track(0)
        chain = build_chain(start, [], chain_length=5)
        assert len(chain) == 0

    def test_direction_build(self):
        start = self._make_track(0, energy=0.2)
        candidates = [
            self._make_track(1, energy=0.9),
            self._make_track(2, energy=0.1),
            self._make_track(3, energy=0.5),
        ]
        chain = build_chain(start, candidates, chain_length=1, direction="build")
        # Should prefer higher energy track when building
        assert chain[0]["track_b"] == "Track 1"  # highest energy

    def test_direction_cool(self):
        start = self._make_track(0, energy=0.9)
        candidates = [
            self._make_track(1, energy=0.2),
            self._make_track(2, energy=0.9),
            self._make_track(3, energy=0.5),
        ]
        chain = build_chain(start, candidates, chain_length=1, direction="cool")
        # Should prefer lower energy track when cooling
        assert chain[0]["track_b"] == "Track 1"


# ═══════════════════════════════════════════════════════════
# analyze_chain
# ═══════════════════════════════════════════════════════════

class TestAnalyzeChain:
    def test_empty(self):
        assert analyze_chain([]) == {"error": "empty chain"}

    def test_rising_energy(self):
        chain = [
            {"total": 0.8, "energy": 0.2, "bpm_ratio": 0.02},
            {"total": 0.7, "energy": 0.5, "bpm_ratio": 0.03},
            {"total": 0.9, "energy": 0.8, "bpm_ratio": 0.01},
        ]
        stats = analyze_chain(chain)
        assert stats["length"] == 3
        assert stats["mean_score"] == pytest.approx(0.8)
        assert stats["energy_trend"] == "rising"

    def test_falling_energy(self):
        chain = [
            {"total": 0.9, "energy": 0.9, "bpm_ratio": 0.01},
            {"total": 0.7, "energy": 0.5, "bpm_ratio": 0.02},
            {"total": 0.6, "energy": 0.2, "bpm_ratio": 0.03},
        ]
        stats = analyze_chain(chain)
        assert stats["energy_trend"] == "falling"
