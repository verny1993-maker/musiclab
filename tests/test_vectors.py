"""
test_vectors.py — Unit tests for vector normalization and Camelot encoding.

Imports from musiclab.utils (zero dependencies).
"""

import math
import sys
from pathlib import Path

import pytest

# Add AudioLab root so `from musiclab.utils import ...` works
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from musiclab.utils import (
    build_8d_vector,
    camelot_to_angle,
    norm,
    sanitize_filename,
    track_to_vector,
)

# ═══════════════════════════════════════════════════════════
# norm
# ═══════════════════════════════════════════════════════════


class TestNorm:
    def test_min_value(self):
        assert norm(0, 0, 100) == 0.0

    def test_max_value(self):
        assert norm(100, 0, 100) == 1.0

    def test_midpoint(self):
        assert norm(50, 0, 100) == 0.5

    def test_below_min_clamped(self):
        assert norm(-10, 0, 100) == 0.0

    def test_above_max_clamped(self):
        assert norm(200, 0, 100) == 1.0

    def test_equal_bounds(self):
        assert norm(50, 50, 50) == 0.5

    def test_negative_range(self):
        assert norm(-30, -50, 0) == 0.4

    def test_bpm_real_values(self):
        assert norm(60, 60, 200) == 0.0
        assert norm(200, 60, 200) == 1.0
        assert norm(130, 60, 200) == 0.5


# ═══════════════════════════════════════════════════════════
# camelot_to_angle
# ═══════════════════════════════════════════════════════════


class TestCamelotToAngle:
    def test_1a(self):
        cos_a, sin_a = camelot_to_angle("1A")
        assert cos_a == pytest.approx(math.cos(math.pi / 12), abs=1e-6)
        assert sin_a == pytest.approx(math.sin(math.pi / 12), abs=1e-6)

    def test_1b(self):
        cos_b, sin_b = camelot_to_angle("1B")
        assert cos_b == pytest.approx(1.0, abs=1e-6)
        assert sin_b == pytest.approx(0.0, abs=1e-6)

    def test_12b(self):
        cos_b, sin_b = camelot_to_angle("12B")
        angle = 11 * 2 * math.pi / 12
        assert cos_b == pytest.approx(math.cos(angle), abs=1e-6)
        assert sin_b == pytest.approx(math.sin(angle), abs=1e-6)

    def test_empty_string(self):
        cos_a, sin_a = camelot_to_angle("")
        assert cos_a == 0.0
        assert sin_a == 0.0

    def test_none(self):
        cos_a, sin_a = camelot_to_angle(None)
        assert cos_a == 0.0
        assert sin_a == 0.0

    def test_invalid_short(self):
        cos_a, sin_a = camelot_to_angle("A")
        assert cos_a == 0.0

    def test_out_of_range(self):
        cos_a, sin_a = camelot_to_angle("13A")
        assert cos_a == 0.0

    def test_zero(self):
        cos_a, sin_a = camelot_to_angle("0A")
        assert cos_a == 0.0

    def test_non_numeric(self):
        cos_a, sin_a = camelot_to_angle("XAB")
        assert cos_a == 0.0

    def test_unit_circle(self):
        for n in range(1, 13):
            for ring in ("A", "B"):
                code = f"{n}{ring}"
                cos_v, sin_v = camelot_to_angle(code)
                dist = (cos_v**2 + sin_v**2) ** 0.5
                assert dist == pytest.approx(1.0, abs=1e-6), f"{code} dist={dist}"

    def test_opposite_keys_tritone(self):
        c1, s1 = camelot_to_angle("1B")
        c7, s7 = camelot_to_angle("7B")
        assert c1 == pytest.approx(-c7, abs=1e-6)
        assert s1 == pytest.approx(-s7, abs=1e-6)


# ═══════════════════════════════════════════════════════════
# build_8d_vector
# ═══════════════════════════════════════════════════════════


class TestBuild8dVector:
    def test_length(self):
        v = build_8d_vector(128, "8A", 0.7, 0.65, [-140, 85, -12])
        assert len(v) == 8

    def test_all_in_range(self):
        v = build_8d_vector(128, "8A", 0.7, 0.65, [-140, 85, -12])
        for i, val in enumerate(v):
            if i in (1, 2):
                assert -1.0 <= val <= 1.0
            else:
                assert 0.0 <= val <= 1.0

    def test_edge_min(self):
        v = build_8d_vector(60, "1B", 0.0, 0.0, [-358, 55, -73])
        assert all(-1.0 <= x <= 1.0 for x in v)
        assert v[0] == 0.0  # min BPM

    def test_edge_max(self):
        v = build_8d_vector(200, "12A", 1.0, 1.0, [-24, 194, 63])
        assert all(-1.0 <= x <= 1.0 for x in v)
        assert v[0] == 1.0  # max BPM

    def test_none_bpm(self):
        v = build_8d_vector(None, "8A", 0.5, 0.5, [0, 0, 0])
        assert len(v) == 8

    def test_none_camelot(self):
        v = build_8d_vector(128, None, 0.5, 0.5, [0, 0, 0])
        assert v[1] == 0.0
        assert v[2] == 0.0

    def test_none_mfcc(self):
        v = build_8d_vector(128, "8A", 0.5, 0.5, None)
        assert len(v) == 8
        for i, val in enumerate(v):
            if i in (1, 2):  # cos/sin on unit circle: [-1, 1]
                assert -1.0 <= val <= 1.0
            else:
                assert 0.0 <= val <= 1.0


# ═══════════════════════════════════════════════════════════
# track_to_vector
# ═══════════════════════════════════════════════════════════


class TestTrackToVector:
    def test_full_track(self):
        track = {
            "audio": {
                "bpm": 128,
                "camelot": "8A",
                "energy": 0.72,
                "danceability": 0.65,
                "mfcc_mean": [-140.2, 85.3, -12.1],
            }
        }
        v = track_to_vector(track)
        assert len(v) == 8
        for i, val in enumerate(v):
            if i in (1, 2):
                assert -1.0 <= val <= 1.0
            else:
                assert 0.0 <= val <= 1.0

    def test_missing_audio(self):
        v = track_to_vector({})
        assert len(v) == 8

    def test_null_fields(self):
        track = {
            "audio": {
                "bpm": None,
                "camelot": None,
                "energy": None,
                "danceability": None,
                "mfcc_mean": None,
            }
        }
        v = track_to_vector(track)
        assert len(v) == 8

    def test_bpm_range(self):
        slow = track_to_vector(
            {
                "audio": {
                    "bpm": 60,
                    "camelot": "1B",
                    "energy": 0.5,
                    "danceability": 0.5,
                    "mfcc_mean": [0, 0, 0],
                }
            }
        )
        fast = track_to_vector(
            {
                "audio": {
                    "bpm": 200,
                    "camelot": "1B",
                    "energy": 0.5,
                    "danceability": 0.5,
                    "mfcc_mean": [0, 0, 0],
                }
            }
        )
        assert slow[0] == pytest.approx(0.0, abs=0.01)
        assert fast[0] == pytest.approx(1.0, abs=0.01)


# ═══════════════════════════════════════════════════════════
# sanitize_filename
# ═══════════════════════════════════════════════════════════


class TestSanitizeFilename:
    def test_basic(self):
        assert sanitize_filename("Artist - Title") == "Artist - Title"

    def test_removes_special(self):
        result = sanitize_filename("Artist / Title: Remix")
        assert "/" not in result
        assert ":" not in result

    def test_truncation(self):
        result = sanitize_filename("A" * 200)
        assert len(result) <= 120

    def test_strips_edges(self):
        result = sanitize_filename(" - Artist - ")
        assert not result.startswith("-")
        assert not result.endswith("-")

    def test_multiple_dashes(self):
        result = sanitize_filename("Artist---Title")
        assert "---" not in result
