"""
test_camelot.py — Unit tests for Camelot wheel mapping (pitch_to_camelot).

Imports from musiclab.utils (zero dependencies).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from musiclab.utils import pitch_to_camelot


class TestPitchToCamelot:
    def test_a_minor(self):
        assert pitch_to_camelot("A", "minor") == "8A"

    def test_c_major(self):
        assert pitch_to_camelot("C", "major") == "8B"

    def test_g_major(self):
        assert pitch_to_camelot("G", "major") == "9B"

    def test_d_minor(self):
        assert pitch_to_camelot("D", "minor") == "7A"

    def test_f_sharp_minor(self):
        assert pitch_to_camelot("F#", "minor") == "11A"

    def test_e_minor(self):
        assert pitch_to_camelot("E", "minor") == "9A"

    def test_b_flat_minor(self):
        result = pitch_to_camelot("Bb", "minor")
        assert result == "3A"

    def test_e_flat_major(self):
        result = pitch_to_camelot("Eb", "major")
        assert result == "5B"

    def test_unknown_key(self):
        assert pitch_to_camelot("H", "major") == "?"

    def test_all_major_keys(self):
        expected = {
            "B": "1B", "F#": "2B", "C#": "3B", "G#": "4B",
            "D#": "5B", "A#": "6B", "F": "7B", "C": "8B",
            "G": "9B", "D": "10B", "A": "11B", "E": "12B",
        }
        for key, exp in expected.items():
            assert pitch_to_camelot(key, "major") == exp, f"{key} major"

    def test_all_minor_keys(self):
        expected = {
            "G#": "1A", "D#": "2A", "A#": "3A", "F": "4A",
            "C": "5A", "G": "6A", "D": "7A", "A": "8A",
            "E": "9A", "B": "10A", "F#": "11A", "C#": "12A",
        }
        for key, exp in expected.items():
            assert pitch_to_camelot(key, "minor") == exp, f"{key} minor"

    def test_relative_keys(self):
        """C major and A minor share Camelot position 8."""
        assert pitch_to_camelot("C", "major") == "8B"
        assert pitch_to_camelot("A", "minor") == "8A"
