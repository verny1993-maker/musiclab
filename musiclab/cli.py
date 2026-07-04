"""
musiclab.cli — Command-line interface wrappers.

Each function wraps the corresponding pipeline script's main(),
allowing `musiclab-<command>` after `pip install musiclab`.
"""

import argparse
import sys
from pathlib import Path

# Project root for script discovery
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _run_script(script_name: str, args: list[str] | None = None):
    """Import and run a pipeline script by name, forwarding CLI args."""
    sys.path.insert(0, str(_PROJECT_ROOT))

    if args is None:
        args = sys.argv[1:]

    # Script names that start with digits need importlib
    if script_name[0].isdigit():
        import importlib

        mod = importlib.import_module(script_name)
    else:
        mod = __import__(script_name)

    # Override sys.argv for argparse-based scripts
    original_argv = sys.argv
    sys.argv = [script_name] + args
    try:
        mod.main()
    finally:
        sys.argv = original_argv


def main_parse():
    """musiclab-parse — Parse a DJ set tracklist from set79.com."""
    parser = argparse.ArgumentParser(
        description="Parse a DJ set tracklist",
        usage="musiclab-parse --url <url> [--artist <name>] [--venue <name>]",
    )
    parser.add_argument("--url", required=True, help="set79 tracklist URL")
    parser.add_argument("--artist", help="Override artist name")
    parser.add_argument("--venue", help="Override venue name")
    parser.add_argument("--date", help="Override date (YYYY-MM-DD)")
    parser.add_argument("--output", help="Output JSON path")
    parsed, remaining = parser.parse_known_args()

    args_list = [f"--url={parsed.url}"]
    if parsed.artist:
        args_list.append(f"--artist={parsed.artist}")
    if parsed.venue:
        args_list.append(f"--venue={parsed.venue}")
    if parsed.date:
        args_list.append(f"--date={parsed.date}")
    if parsed.output:
        args_list.append(f"--output={parsed.output}")

    _run_script("01_parse", args_list)


def main_enrich_audio():
    """musiclab-enrich-audio — Download and analyze audio for a set."""
    parser = argparse.ArgumentParser(
        description="Download and analyze audio for a set",
        usage="musiclab-enrich-audio --set <path>",
    )
    parser.add_argument("--set", required=True, help="Path to set JSON")
    parsed, _ = parser.parse_known_args()
    _run_script("02_enrich_audio", [f"--set={parsed.__dict__['set']}"])


def main_enrich_meta():
    """musiclab-enrich-meta — Enrich tracks with Discogs/Last.fm/MusicBrainz."""
    parser = argparse.ArgumentParser(
        description="Enrich track metadata",
        usage="musiclab-enrich-meta --set <path>",
    )
    parser.add_argument("--set", required=True, help="Path to set JSON")
    parsed, _ = parser.parse_known_args()
    _run_script("02_enrich_meta", [f"--set={parsed.__dict__['set']}"])


def main_enrich_library():
    """musiclab-enrich-library — Batch audio analysis for Spotify liked tracks."""
    _run_script(
        "03_enrich_audio_library",
        sys.argv[1:] if len(sys.argv) > 1 else [],
    )


def main_build_cards():
    """musiclab-build-cards — Build artist and venue intelligence cards."""
    _run_script(
        "05_build_cards",
        sys.argv[1:] if len(sys.argv) > 1 else [],
    )
