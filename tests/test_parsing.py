"""
test_parsing.py — Unit tests for timecode parsing and string utilities.

Imports from musiclab.utils (zero dependencies).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from musiclab.utils import parse_timecode, split_artist_title, slugify


class TestParseTimecode:
    def test_mm_ss(self):
        assert parse_timecode("01:23") == 83.0

    def test_hh_mm_ss(self):
        assert parse_timecode("01:23:45") == 5025.0

    def test_zero(self):
        assert parse_timecode("00:00") == 0.0

    def test_single_digit_minutes(self):
        assert parse_timecode("5:30") == 330.0

    def test_large_hours(self):
        assert parse_timecode("10:00:00") == 36000.0

    def test_empty(self):
        assert parse_timecode("") == 0.0

    def test_spaces(self):
        assert parse_timecode(" 01:23 ") == 83.0


class TestSplitArtistTitle:
    def test_standard_dash(self):
        a, t = split_artist_title("Aphex Twin - Windowlicker")
        assert a == "Aphex Twin"
        assert t == "Windowlicker"

    def test_em_dash(self):
        a, t = split_artist_title("Daniel Avery – Drone Logic")
        assert a == "Daniel Avery"
        assert t == "Drone Logic"

    def test_no_separator(self):
        a, t = split_artist_title("Untitled Track")
        assert a == ""
        assert t == "Untitled Track"

    def test_title_first(self):
        a, t = split_artist_title("Windowlicker - Aphex Twin", title_first=True)
        assert a == "Aphex Twin"
        assert t == "Windowlicker"

    def test_multiple_dashes(self):
        a, t = split_artist_title("Artist - Title - Remix")
        assert a == "Artist"
        assert t == "Title - Remix"


class TestSlugify:
    def test_basic(self):
        assert slugify("Yousuke Yukimatsu") == "yousuke-yukimatsu"

    def test_special_chars(self):
        slug = slugify("YOU SUKE YUKIMATSU")
        assert slug == "you-suke-yukimatsu"

    def test_multiple_spaces(self):
        assert slugify("Boiler  Room") == "boiler-room"

    def test_truncation(self):
        slug = slugify("A" * 100)
        assert len(slug) <= 60
