"""
04_hypothesis.py — Generate and evaluate hypotheses from enriched set data.

Each hypothesis is a falsifiable claim backed by N observed sets.

Usage:
    python 04_hypothesis.py --artist <slug>   # generate hypotheses for an artist
    python 04_hypothesis.py --all              # process all artists
"""

from __future__ import annotations

import argparse


def generate_hypotheses(artist_slug: str) -> list[dict]:
    """
    Analyze all enriched sets for `artist_slug` and produce hypotheses.

    Returns a list of hypothesis dicts conforming to hypothesis.schema.json.
    Each hypothesis includes based_on_n_sets and confidence.
    """
    # TODO: implement hypothesis generation
    raise NotImplementedError("04_hypothesis not implemented yet — TODO")


def main():
    parser = argparse.ArgumentParser(
        description="Generate hypotheses from enriched DJ sets"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--artist", type=str, help="Artist slug")
    group.add_argument("--all", action="store_true", help="Process all artists")
    args = parser.parse_args()
    # TODO: dispatch
    print(
        f"[04_hypothesis] artist={args.artist} all={args.all} — NOT IMPLEMENTED"
    )


if __name__ == "__main__":
    main()
