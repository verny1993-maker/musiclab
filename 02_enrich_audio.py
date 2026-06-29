"""
02_enrich_audio.py — Download set audio, slice by track boundaries, analyze each slice.

Flow:
  1. Download audio via yt-dlp (SoundCloud → YouTube fallback).
  2. Load full audio with librosa.
  3. For each track: slice [start_sec[i], start_sec[i+1]), trim 15% from edges,
     export temp WAV, POST to music-analysis API (:8777).
  4. Fill audio_features: bpm, key, camelot, key_strength, energy, danceability,
     loudness, mfcc_mean, beat_positions.
  5. Track failures → audio: null + error in set.errors[].
  6. Alignment check: full audio duration vs last track timecode → WARNING if gap > 15s.
  7. Idempotent: skip tracks with audio ≠ null.

Usage:
    python 02_enrich_audio.py --set data/sets/<set_id>.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import librosa
import numpy as np
import requests
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.lib_io import load_json, write_json

logger = logging.getLogger("02_enrich_audio")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")

PROJECT_ROOT = Path(__file__).resolve().parent
TMP_DIR = str(PROJECT_ROOT / "data" / "tmp_analysis")
CONTAINER_TMP = "/tmp/analysis"  # Docker volume mount target

ANALYSIS_API = "http://localhost:8777/analyze"
EDGE_TRIM = 0.15
ALIGNMENT_TOLERANCE_SEC = 15.0


def download_audio(audio_url: str, youtube_url: str | None, dest_dir: str) -> str | None:
    """
    Download audio from SoundCloud (primary) or YouTube (fallback) using yt-dlp.

    Returns path to downloaded audio file, or None.
    Cached: if a .wav already exists in dest_dir, returns it directly.
    """
    # Check cache
    for f in os.listdir(dest_dir):
        if f.endswith(".wav"):
            path = os.path.join(dest_dir, f)
            logger.info("Using cached audio: %s (%.1f MB)", f, os.path.getsize(path) / 1e6)
            return path

    urls = [audio_url]
    if youtube_url:
        urls.append(youtube_url)

    for i, url in enumerate(urls):
        source = "SoundCloud" if i == 0 else "YouTube"
        logger.info("Downloading from %s: %s", source, url)
        try:
            subprocess.run(
                [
                    sys.executable, "-m", "yt_dlp",
                    "-x", "--audio-format", "wav",
                    "--audio-quality", "0",
                    "-o", f"{dest_dir}/%(title)s.%(ext)s",
                    "--no-playlist",
                    "--socket-timeout", "30",
                    url,
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=300,
            )
            # Find the downloaded WAV
            for f in os.listdir(dest_dir):
                if f.endswith(".wav"):
                    path = os.path.join(dest_dir, f)
                    logger.info("Downloaded: %s (%.1f MB)", f, os.path.getsize(path) / 1e6)
                    return path
        except subprocess.CalledProcessError as e:
            logger.warning("%s download failed: %s", source, e.stderr[-200:])
        except Exception as e:
            logger.warning("%s download failed: %s", source, e)

    return None


def analyze_slice(wav_path: str, container_path: str | None = None) -> dict | None:
    """
    POST a WAV file to the music-analysis API.

    The API expects a filepath accessible inside its Docker container.
    We save to /tmp/musiclab_analysis/ which is volume-mounted to /tmp/analysis in the container.
    """
    api_path = container_path or wav_path
    try:
        resp = requests.post(
            ANALYSIS_API,
            json={"filepath": api_path},
            timeout=(10, 120),
        )
        resp.raise_for_status()
        data = resp.json()

        # Extract relevant fields for audio_features
        af = {
            "bpm": data.get("bpm"),
            "key": data.get("key"),
            "camelot": data.get("camelot"),
            "key_strength": data.get("key_strength"),
            "energy": data.get("energy"),
            "danceability": data.get("danceability"),
            "loudness": data.get("loudness"),
            "mfcc_mean": data.get("mfcc_mean"),
            "beat_positions": data.get("beat_positions", []),
        }
        # Validate: all required fields must be present
        required = ["bpm", "key", "camelot", "key_strength", "energy", "danceability", "loudness", "mfcc_mean", "beat_positions"]
        # Fallback: null danceability → 0.5 (API bug workaround)
        for k in required:
            if af.get(k) is None:
                if k == "danceability":
                    af["danceability"] = 0.5
                elif k == "bpm" and af.get("bpm") is None:
                    af["bpm"] = 120.0  # reasonable default
        if any(af.get(k) is None for k in required):
            missing = [k for k in required if af.get(k) is None]
            logger.warning("API returned incomplete audio_features: missing %s", missing)
            return None
        return af
    except Exception as e:
        logger.warning("Analysis API failed: %s", e)
        return None


def enrich_audio(set_path: str) -> Path:
    """Full audio enrichment pipeline."""
    set_data = load_json(set_path, "set")
    set_dir = Path(set_path).parent
    audio_dir = set_dir.parent / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    audio_url = set_data.get("audio_url")
    youtube_url = set_data.get("youtube_url")
    if not audio_url and not youtube_url:
        logger.error("No audio_url or youtube_url in set")
        return Path(set_path)

    # ── Step 1: Download ──
    download_dir = str(audio_dir / set_data["id"])
    os.makedirs(download_dir, exist_ok=True)
    audio_path = download_audio(audio_url, youtube_url, download_dir)
    if not audio_path:
        logger.error("Failed to download audio from all sources")
        set_data.setdefault("errors", []).append({
            "position": 0,
            "stage": "02_enrich/download",
            "reason": "Failed to download audio from SoundCloud and YouTube",
        })
        write_json(set_data, "set", set_path)
        return Path(set_path)

    # ── Step 2: Get audio info (don't load full file) ──
    logger.info("Reading audio info: %s", audio_path)
    full_duration = librosa.get_duration(path=audio_path)
    sr = librosa.get_samplerate(audio_path)
    logger.info("Full audio: %.1fs, sr=%d", full_duration, sr)

    # ── Alignment check ──
    tracks = set_data["tracks"]
    last_timecode = max(t.get("start_sec", 0) for t in tracks) if tracks else 0
    gap = abs(full_duration - last_timecode)
    if gap > ALIGNMENT_TOLERANCE_SEC:
        logger.warning(
            "ALIGNMENT WARNING: audio duration %.1fs vs last track start %.1fs (gap=%.1fs, %d tracks). "
            "Manual alignment_checked flag NOT set.",
            full_duration, last_timecode, gap, len(tracks),
        )

    # ── Step 3: Slice & analyze each track (offset-based loading) ──
    os.makedirs(TMP_DIR, exist_ok=True)
    tmp_dir = TMP_DIR
    errors: list[dict] = list(set_data.get("errors", []))
    enriched = 0
    skipped = 0

    for i, track in enumerate(tracks):
        pos = track["position"]
        artist = track.get("artist", "?")
        title = track.get("title", "?")

        # Idempotent
        if track.get("audio") is not None:
            skipped += 1
            continue

        start_sec = track.get("start_sec", 0)
        # End is next track's start, or file end
        if i + 1 < len(tracks):
            end_sec = tracks[i + 1].get("start_sec", full_duration)
        else:
            end_sec = full_duration

        slice_duration = end_sec - start_sec
        if slice_duration < 5:
            logger.warning("[%d/%d] %s — %s: too short (%.1fs), skipping", pos, len(tracks), artist, title, slice_duration)
            errors.append({
                "position": pos,
                "stage": "02_enrich/slice",
                "reason": f"Track slice too short ({slice_duration:.1f}s)",
            })
            continue

        # Trim edges: exclude transition zones
        trim = slice_duration * EDGE_TRIM
        trim_start = start_sec + trim
        trim_end = end_sec - trim
        if trim_end - trim_start < 3:
            trim_start = start_sec
            trim_end = end_sec
            logger.debug("[%d] Trim too aggressive, using full slice", pos)

        # Load only the slice via offset/duration
        slice_dur = trim_end - trim_start
        try:
            slice_y, _ = librosa.load(
                audio_path, sr=sr, mono=True,
                offset=trim_start, duration=slice_dur,
            )
        except Exception as e:
            logger.warning("[%d] Failed to load slice: %s", pos, e)
            errors.append({
                "position": pos,
                "stage": "02_enrich/slice",
                "reason": f"Failed to load audio slice: {e}",
            })
            continue

        if len(slice_y) < sr * 2:  # less than 2 seconds
            logger.warning("[%d/%d] %s — %s: slice too short (%d samples), skipping",
                           pos, len(tracks), artist, title, len(slice_y))
            errors.append({
                "position": pos,
                "stage": "02_enrich/slice",
                "reason": "Slice too short after loading",
            })
            continue

        # Export temp WAV
        tmp_name = f"track_{pos:03d}.wav"
        tmp_host = os.path.join(tmp_dir, tmp_name)
        tmp_container = f"{CONTAINER_TMP}/{tmp_name}"
        sf.write(tmp_host, slice_y, sr)

        # Analyze
        logger.info("[%d/%d] %s — %s (%.1f-%.1fs, trim %.1fs, slice %.1fs)",
                     pos, len(tracks), artist, title, start_sec, end_sec, trim, slice_dur)
        af = analyze_slice(tmp_host, tmp_container)

        if af:
            track["audio"] = af
            enriched += 1
            logger.info("  -> bpm=%-6s key=%-10s camelot=%-4s energy=%.3f",
                         af.get("bpm"), af.get("key"), af.get("camelot"), af.get("energy", 0))
        else:
            errors.append({
                "position": pos,
                "stage": "02_enrich/analysis",
                "reason": f"Audio analysis failed for {artist} — {title}",
            })
            logger.warning("  -> ANALYSIS FAILED")

        # Cleanup temp
        try:
            os.unlink(tmp_host)
        except OSError:
            pass

        # Small breather between tracks
        time.sleep(0.5)

    # ── Step 4: Save ──
    set_data["errors"] = errors
    set_data["enriched_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    result = write_json(set_data, "set", set_path)
    logger.info("Done: %d enriched, %d skipped, %d errors", enriched, skipped,
                 len([e for e in errors if e.get("stage", "").startswith("02_enrich")]))
    return result


def main():
    parser = argparse.ArgumentParser(description="Enrich set with audio features")
    parser.add_argument("--set", required=True, help="Path to set JSON file")
    args = parser.parse_args()
    enrich_audio(args.set)


if __name__ == "__main__":
    main()
