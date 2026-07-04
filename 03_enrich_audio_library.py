"""
03_enrich_audio_library.py — Batch audio analysis pipeline for Spotify liked tracks.

Downloads from SoundCloud (first) or YouTube (fallback), analyzes via music-analysis:8777,
computes MusicLab 8D vectors, stores in SQLite.

Usage:
    python 03_enrich_audio_library.py --limit 10 --dry-run
    python 03_enrich_audio_library.py --workers 4 --resume
"""
from __future__ import annotations

import argparse, hashlib, json, logging, math, os, re, shutil, sqlite3, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

logger = logging.getLogger("03_enrich_audio")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")

PROJECT_ROOT = Path(__file__).resolve().parent
TEMP_AUDIO_DIR = PROJECT_ROOT / "data" / "temp_audio"
TMP_ANALYSIS_DIR = PROJECT_ROOT / "data" / "tmp_analysis"
DB_PATH = PROJECT_ROOT / "data" / "library" / "audio_analysis.db"
ANALYZE_URL = "http://localhost:8777/analyze"
COOKIES_PATH = PROJECT_ROOT / "data" / "cookies.txt"

BPM_MIN, BPM_MAX = 60.0, 200.0
MFCC1_MIN, MFCC1_MAX = -358.0, -24.0
MFCC2_MIN, MFCC2_MAX = 55.0, 194.0
MFCC3_MIN, MFCC3_MAX = -73.0, 63.0

SANITIZE_RE = re.compile(r'[&/\\:*?"<>|()\[\]{}—–' "']+")
SANITIZE_DASH_RE = re.compile(r'-{2,}')


def sanitize_filename(name: str) -> str:
    name = SANITIZE_RE.sub('', name)
    name = SANITIZE_DASH_RE.sub('-', name)
    return name.strip(' -_.')[:120]


def _norm(value: float, vmin: float, vmax: float) -> float:
    if vmax == vmin: return 0.5
    return max(0.0, min(1.0, (value - vmin) / (vmax - vmin)))


def camelot_to_angle(camelot: str) -> tuple[float, float]:
    if not camelot or len(camelot) < 2: return 0.0, 0.0
    try:
        n = int(camelot[:-1]); ring = camelot[-1].upper()
    except (ValueError, IndexError): return 0.0, 0.0
    if n < 1 or n > 12: return 0.0, 0.0
    angle = (n - 1) * 2 * math.pi / 12
    if ring == 'A': angle += math.pi / 12
    return math.cos(angle), math.sin(angle)


def build_8d_vector(bpm, camelot, energy, danceability, mfcc_mean):
    key_cos, key_sin = camelot_to_angle(camelot)
    mfcc = (mfcc_mean or [0,0,0])[:3]
    return [
        round(_norm(bpm or 0, BPM_MIN, BPM_MAX), 6),
        round(key_cos, 6), round(key_sin, 6),
        round(energy or 0, 6), round(danceability or 0, 6),
        round(_norm(mfcc[0], MFCC1_MIN, MFCC1_MAX), 6),
        round(_norm(mfcc[1], MFCC2_MIN, MFCC2_MAX), 6),
        round(_norm(mfcc[2], MFCC3_MIN, MFCC3_MAX), 6),
    ]


def search_audio_track(query: str) -> tuple[str | None, str]:
    for src, prefix, timeout in [("soundcloud", "scsearch1:", 20), ("youtube", "ytsearch1:", 30)]:
        try:
            cmd = [sys.executable, "-m", "yt_dlp", "--flat-playlist", "--print", "url",
                   f"{prefix}{query}", "--no-playlist"]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                                 env={**os.environ, "PYTHONIOENCODING": "utf-8"})
            for line in proc.stdout.strip().split("\n"):
                line = line.strip()
                if src == "soundcloud" and "soundcloud" in line.lower():
                    return line, "soundcloud"
                if src == "youtube" and line.startswith("https://"):
                    return line, "youtube"
        except Exception as e:
            logger.debug("%s search failed: %s", src, e)
    return None, "none"


def _search_yt_only(query: str) -> tuple[str | None, str]:
    """YouTube-only search (used as fallback when SC download fails)."""
    try:
        cmd = [sys.executable, "-m", "yt_dlp", "--flat-playlist", "--print", "url",
               f"ytsearch1:{query}", "--no-playlist"]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30,
                             env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        for line in proc.stdout.strip().split("\n"):
            line = line.strip()
            if line.startswith("https://"):
                return line, "youtube"
    except Exception as e:
        logger.debug("YT fallback search failed: %s", e)
    return None, "none"


def download_audio(url: str, output_path: Path) -> bool:
    """Download audio and convert to mono 22kHz WAV via FFmpeg."""
    raw_stem = str(output_path).replace('.wav', '_raw')
    try:
        cmd = [sys.executable, "-m", "yt_dlp",
               "-f", "bestaudio[filesize<30M]",  # hard limit, no fallback
               "-o", raw_stem + ".%(ext)s", url,
               "--no-playlist", "--max-filesize", "50m"]
        if COOKIES_PATH.exists():
            url_idx = cmd.index(url)
            cmd.insert(url_idx, "--cookies")
            cmd.insert(url_idx + 1, str(COOKIES_PATH))
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600,
                             env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        if proc.returncode != 0:
            return False
        # Find the downloaded file (yt-dlp may add .webm/.m4a/.opus etc.)
        raw_files = list(output_path.parent.glob(Path(raw_stem).name + ".*"))
        if not raw_files:
            return False
        raw_file = raw_files[0]
        # Convert to mono 22kHz WAV via FFmpeg
        ffmpeg_cmd = ["ffmpeg", "-y", "-i", str(raw_file),
                      "-ac", "1", "-ar", "22050", str(output_path)]
        ffproc = subprocess.run(ffmpeg_cmd, capture_output=True, timeout=300,
                               env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        raw_file.unlink(missing_ok=True)
        return ffproc.returncode == 0 and output_path.exists()
    except Exception as e:
        logger.warning("Download failed: %s", e)
        return False


def call_analyze(wav_path: Path) -> dict | None:
    docker_path = f"/tmp/analysis/{wav_path.name}"
    last_error = None
    for attempt in range(3):
        try:
            resp = requests.post(ANALYZE_URL, json={"filepath": docker_path}, timeout=300)
            if resp.status_code == 200:
                return resp.json()
            last_error = f"HTTP {resp.status_code}"
            logger.debug("/analyze attempt %d: HTTP %d", attempt + 1, resp.status_code)
        except Exception as e:
            last_error = e
            logger.debug("/analyze attempt %d failed: %s", attempt + 1, e)
        if attempt < 2:
            time.sleep(5.0)  # Docker container restart delay
    logger.warning("/analyze failed after 3 attempts: %s", last_error)
    return None


def init_db(db_path: Path):
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS audio_analysis (
            track_id TEXT PRIMARY KEY, artist TEXT NOT NULL, title TEXT NOT NULL,
            bpm REAL, key_text TEXT, camelot TEXT, key_strength REAL,
            energy REAL, danceability REAL, loudness REAL, duration_s REAL,
            mfcc_mean TEXT, spectral_centroid REAL, zero_crossing_rate REAL,
            vector_8d TEXT, source_url TEXT, source_type TEXT,
            analyzed_at TEXT, analysis_time_ms INTEGER
        );
        CREATE TABLE IF NOT EXISTS audio_analysis_errors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            track_id TEXT NOT NULL, artist TEXT, title TEXT,
            error TEXT, stage TEXT, timestamp TEXT DEFAULT (datetime('now'))
        );
        CREATE INDEX IF NOT EXISTS idx_analyzed_at ON audio_analysis(analyzed_at);
    """)
    conn.commit()
    return conn


def is_analyzed(conn, track_id): return conn.execute("SELECT 1 FROM audio_analysis WHERE track_id=?", (track_id,)).fetchone() is not None

def save_result(conn, track_id, artist, title, analysis, source_url, source_type):
    vector = build_8d_vector(analysis.get("bpm"), analysis.get("camelot"),
                             analysis.get("energy"), analysis.get("danceability"),
                             analysis.get("mfcc_mean"))
    conn.execute("""INSERT OR REPLACE INTO audio_analysis
        (track_id,artist,title,bpm,key_text,camelot,key_strength,energy,danceability,loudness,duration_s,mfcc_mean,spectral_centroid,zero_crossing_rate,vector_8d,source_url,source_type,analyzed_at,analysis_time_ms)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,datetime('now'),?)""",
        (track_id, artist, title, analysis.get("bpm"), analysis.get("key"), analysis.get("camelot"),
         analysis.get("key_strength"), analysis.get("energy"), analysis.get("danceability"),
         analysis.get("loudness"), analysis.get("duration"), json.dumps(analysis.get("mfcc_mean")),
         analysis.get("spectral_centroid"), analysis.get("zero_crossing_rate"),
         json.dumps(vector), source_url, source_type, analysis.get("_analysis_time_ms")))
    conn.commit()

def save_error(conn, track_id, artist, title, error, stage):
    conn.execute("INSERT INTO audio_analysis_errors (track_id,artist,title,error,stage) VALUES (?,?,?,?,?)",
                 (track_id, artist, title, error, stage))
    conn.commit()


def analyze_track(track: dict, db_path: Path, dry_run: bool = False) -> dict:
    """Thread-safe: creates own DB connection."""
    conn = sqlite3.connect(str(db_path))
    try:
        track_id = track["id"]
        artist = track["artist"].split(",")[0].strip()
        title = track["name"]
        query = f"{artist} {title}"
        result = {"track_id": track_id, "artist": artist, "title": title, "status": "pending"}

        if is_analyzed(conn, track_id):
            result["status"] = "skipped"
            return result
        if dry_run:
            result["status"] = "dry_run"
            return result

        safe_name = sanitize_filename(f"{artist}_{title}")[:80]
        temp_wav = TEMP_AUDIO_DIR / f"{safe_name}.wav"

        try:
            TEMP_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
            TMP_ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

            audio_url, source_type = search_audio_track(query)
            if not audio_url:
                save_error(conn, track_id, artist, title, "No results (SC+YT)", "search")
                result["status"] = "error_search"
                return result

            downloaded = download_audio(audio_url, temp_wav) and temp_wav.exists()
            # SC download failed (likely no cookies) — fallback to YouTube
            if not downloaded and source_type == "soundcloud":
                logger.debug("SC download failed, trying YT fallback for %s", query[:40])
                yt_url, yt_source = _search_yt_only(query)
                if yt_url:
                    audio_url, source_type = yt_url, yt_source
                    downloaded = download_audio(audio_url, temp_wav) and temp_wav.exists()

            if not downloaded:
                save_error(conn, track_id, artist, title, "Download failed", "download")
                result["status"] = "error_download"
                return result

            docker_wav = TMP_ANALYSIS_DIR / temp_wav.name
            shutil.copy2(temp_wav, docker_wav)

            analysis = call_analyze(docker_wav)
            if not analysis or analysis.get("bpm") is None:
                save_error(conn, track_id, artist, title, "Empty analysis", "analyze")
                result["status"] = "error_analyze"
                return result

            save_result(conn, track_id, artist, title, analysis, audio_url, source_type)
            result["status"] = "ok"; result["bpm"] = analysis.get("bpm")
            result["camelot"] = analysis.get("camelot"); result["source"] = source_type
            return result

        except Exception as e:
            save_error(conn, track_id, artist, title, str(e), "exception")
            result["status"] = "error_exception"; result["error"] = str(e)
            return result
        finally:
            # Windows: yt-dlp/FFmpeg may not release file handles immediately
            for f in [temp_wav, TMP_ANALYSIS_DIR / temp_wav.name]:
                for attempt in range(3):
                    try:
                        f.unlink(missing_ok=True)
                        break
                    except PermissionError:
                        time.sleep(1.0)
            # Clean leftover partial downloads from yt-dlp (.m4a, .part, .ytdl, etc.)
            for leftover in temp_wav.parent.glob(temp_wav.stem + '.*'):
                for attempt in range(3):
                    try:
                        leftover.unlink(missing_ok=True)
                        break
                    except PermissionError:
                        time.sleep(1.0)
    finally:
        conn.close()


def load_tracks(source: str, limit: int = 0) -> list[dict]:
    if source.endswith(".json"):
        with open(source, encoding="utf-8") as f: tracks = json.load(f)
    elif source.endswith(".db") or source.endswith(".sqlite"):
        conn = sqlite3.connect(source)
        tracks = [{"id": r[0], "artist": r[1], "name": r[2]}
                  for r in conn.execute("SELECT track_id, artist, title FROM tracks").fetchall()]
        conn.close()
    else: raise ValueError(f"Unsupported source: {source}")
    return tracks[:limit] if limit > 0 else tracks


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="data/library/tracks_enriched.json")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--db", default=str(DB_PATH))
    args = p.parse_args()

    db_path = Path(args.db)
    conn = init_db(db_path)
    source_path = PROJECT_ROOT / args.source if not os.path.isabs(args.source) else Path(args.source)
    tracks = load_tracks(str(source_path), args.limit)
    logger.info("Loaded %d tracks", len(tracks))

    to_process = [t for t in tracks if args.dry_run or not is_analyzed(conn, t["id"])]
    logger.info("To process: %d (already: %d)", len(to_process), len(tracks) - len(to_process))
    conn.close()

    if args.dry_run:
        for t in to_process[:10]: logger.info("  %s — %s", t["artist"], t["name"])
        return

    t0 = time.monotonic(); ok = err = 0
    total = len(to_process)

    if args.workers > 1:
        processed = 0
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(analyze_track, t, db_path, False): t for t in to_process}
            for fut in as_completed(futures):
                r = fut.result()
                processed += 1
                if r["status"] == "ok":
                    ok += 1
                    logger.info("✓ [%d/%d] %s  BPM=%s %s", ok+err, total, r['artist'][:20], r.get('bpm','?'), r.get('camelot',''))
                else:
                    err += 1
                    logger.warning("✗ [%d/%d] %s  %s", ok+err, total, futures[fut]['artist'][:20], r['status'])
                # Periodic Docker restart to prevent memory leak
                if processed % 40 == 0:
                    logger.info("Restarting Docker (processed %d tracks)...", processed)
                    subprocess.run(["docker", "restart", "music-analysis-music-analysis-1"], capture_output=True, timeout=30)
                    time.sleep(10)
    else:
        for i, t in enumerate(to_process):
            r = analyze_track(t, db_path)
            if r["status"] == "ok":
                ok += 1
                logger.info("✓ [%d/%d] %s  BPM=%s %s", i+1, total, t['artist'][:20], r.get('bpm','?'), r.get('camelot',''))
            else:
                err += 1
                logger.warning("✗ [%d/%d] %s  %s", i+1, total, t['artist'][:20], r['status'])

    elapsed = time.monotonic() - t0

    vconn = sqlite3.connect(str(db_path))
    total_db = vconn.execute("SELECT COUNT(*) FROM audio_analysis").fetchone()[0]
    errs_db = vconn.execute("SELECT COUNT(*) FROM audio_analysis_errors").fetchone()[0]
    vconn.close()

    logger.info("=" * 50)
    logger.info("DONE: %d OK, %d errors, %.0fs (%.0fs/track)", ok, err, elapsed, elapsed/max(1,total))
    logger.info("DB: %d analyzed, %d errors", total_db, errs_db)


if __name__ == "__main__":
    main()
