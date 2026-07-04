"""
lib_io.py — Single writer for all Music Lab JSON artifacts.

Every script that produces a .json file must go through this module.
Guarantees:
  1. Validation against the declared JSON Schema (rejects invalid objects).
  2. Atomic write: writes to a .tmp file, then os.rename (no partial files on disk).
  3. Canonical path resolution relative to the project root.

Usage:
    from lib.lib_io import write_json

    write_json(data, schema="set", path="data/sets/my-set.json")
    # Raises ValidationError if data doesn't conform to schemas/set.schema.json
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Literal

import jsonschema

# ── Project root detection ──────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SCHEMAS_DIR = _PROJECT_ROOT / "schemas"

# Schema file mapping
_SCHEMA_FILES: dict[str, Path] = {
    "set": _SCHEMAS_DIR / "set.schema.json",
    "artist": _SCHEMAS_DIR / "artist.schema.json",
    "hypothesis": _SCHEMAS_DIR / "hypothesis.schema.json",
}

# Cache for loaded schemas (loaded once, reused)
_schema_cache: dict[str, dict] = {}


def _load_schema(name: str) -> dict:
    """Load a JSON Schema from schemas/, with caching."""
    if name not in _schema_cache:
        schema_path = _SCHEMA_FILES.get(name)
        if schema_path is None:
            raise ValueError(
                f"Unknown schema '{name}'. Valid: {sorted(_SCHEMA_FILES.keys())}"
            )
        if not schema_path.exists():
            raise FileNotFoundError(f"Schema file not found: {schema_path}")
        with open(schema_path, encoding="utf-8") as f:
            _schema_cache[name] = json.load(f)
    return _schema_cache[name]


def validate(data: dict, schema_name: Literal["set", "artist", "hypothesis"]) -> None:
    """Validate `data` against a named schema. Raises jsonschema.ValidationError on failure."""
    schema = _load_schema(schema_name)
    jsonschema.validate(instance=data, schema=schema)


def write_json(
    data: dict,
    schema: Literal["set", "artist", "hypothesis"],
    path: str | Path,
    *,
    indent: int = 2,
    ensure_ascii: bool = False,
) -> Path:
    """
    Validate `data` against `schema`, then write atomically to `path`.

    Args:
        data: The Python dict to write.
        schema: Schema name — 'set', 'artist', or 'hypothesis'.
        path: Destination path (absolute, or relative to project root).
        indent: JSON indentation (default 2).
        ensure_ascii: Passed to json.dump (default False — allows Unicode).

    Returns:
        The resolved Path that was written.

    Raises:
        jsonschema.ValidationError: if `data` fails schema validation.
        OSError: on filesystem errors.
    """
    # 1. Validate
    validate(data, schema)

    # 2. Resolve path (relative → absolute under project root)
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = _PROJECT_ROOT / resolved

    # 3. Ensure parent directory exists
    resolved.parent.mkdir(parents=True, exist_ok=True)

    # 4. Write to a temporary file in the same directory, then rename
    #    (atomic on the same filesystem — no partial files visible)
    tmp_fd, tmp_path = tempfile.mkstemp(
        suffix=".tmp",
        prefix=f".{resolved.name}.",
        dir=str(resolved.parent),
    )
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii)
            f.write("\n")  # trailing newline
        os.replace(tmp_path, resolved)  # atomic on same fs
    except Exception:
        # Clean up temp file on failure
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise

    return resolved


def read_json(path: str | Path) -> dict:
    """
    Read and parse a JSON file. Returns the parsed dict.
    Does NOT validate — use load_json() for validated reads.
    """
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = _PROJECT_ROOT / resolved
    with open(resolved, encoding="utf-8") as f:
        return json.load(f)


def load_json(
    path: str | Path,
    schema: Literal["set", "artist", "hypothesis"],
) -> dict:
    """
    Read a JSON file AND validate it against a schema.
    Use this when loading existing data that must conform.
    """
    data = read_json(path)
    validate(data, schema)
    return data
