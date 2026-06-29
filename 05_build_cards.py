"""
05_build_cards.py — Build artist and set "cards" for display/export.

A card is a curated summary: artist profile cards, set overview cards,
energy curve visualizations, key transition maps, etc.

Energy curve archetype classification (set.schema.json):
  - type: "rising" | "peak_valley" | "flat" | null
  - segments: [] (placeholder, will be populated with actual segment boundaries)

Usage:
    python 05_build_cards.py --artist <slug>
    python 05_build_cards.py --set <set_id>
"""

from __future__ import annotations

import argparse


def build_artist_card(artist_slug: str) -> dict:
    """
    Build a summary card for an artist from computed + manual fields.

    Returns a dict suitable for rendering (JSON → frontend).
    """
    # TODO: implement card builder
    raise NotImplementedError("05_build_cards not implemented yet — TODO")


def build_set_card(set_id: str) -> dict:
    """
    Build a summary card for a single set, including energy curve archetype.

    Returns a dict with bpm, key spread, energy plot data, archetype, errors summary.
    """
    # TODO: implement set card builder
    raise NotImplementedError("05_build_cards not implemented yet — TODO")


def main():
    parser = argparse.ArgumentParser(description="Build artist/set cards")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--artist", type=str, help="Artist slug")
    group.add_argument("--set", type=str, help="Set ID")
    args = parser.parse_args()
    # TODO: dispatch
    print(
        f"[05_build_cards] artist={args.artist} set={args.set} — NOT IMPLEMENTED"
    )


if __name__ == "__main__":
    main()
