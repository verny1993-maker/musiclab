"""
06_transitions.py — Analyze transitions between consecutive tracks in a set.

A transition is a separate entity:
  - track_from (position i) → track_to (position i+1)
  - Overlap zone (if any)
  - Energy delta, BPM delta, key compatibility

*** STUB — NOT IMPLEMENTED ***
Will be activated when we reach manual transition markup stage.

Usage:
    python 06_transitions.py --set <set_id>
"""

from __future__ import annotations

import argparse


def analyze_transitions(set_id: str) -> list[dict]:
    """
    Analyze every transition in the set.

    Returns a list of transition dicts (schema TBD when implemented).
    Each transition spans track i → i+1 with an overlap zone.
    """
    raise NotImplementedError(
        "06_transitions is a stub — transitions are a separate entity, "
        "not implemented until transition markup stage."
    )


def main():
    parser = argparse.ArgumentParser(
        description="Analyze transitions between consecutive tracks (STUB)"
    )
    parser.add_argument("--set", required=True, help="Set ID")
    args = parser.parse_args()
    print(f"[06_transitions] set={args.set} — STUB, not implemented")


if __name__ == "__main__":
    main()
