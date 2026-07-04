"""Music Analysis Service v2 — beat-grid, danceability, essentia-enhanced."""

import json, os, time
import numpy as np
import librosa
import soundfile
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Optional: essentia (Docker container)
ESSENTIA = False
try:
    import essentia.standard as es
    import essentia.streaming as es_stream
    ESSENTIA = True
except ImportError:
    pass

app = FastAPI(title="Music Analysis Service", version="2.0.0")

# Pitch class to circle-of-fifths index (0=C at top, clockwise: G,D,A,E,B,F#,C#,G#,D#,A#,F)
# Used only for legacy chroma fallback — real key detection now uses essentia KeyExtractor
KEYS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Camelot wheel mapping: (pitch_class_index, scale) → Camelot code
# Standard circle-of-fifths numbering:
#   Major (B ring):  1B=B, 2B=F#, 3B=C#, 4B=G#, 5B=D#, 6B=A#, 7B=F, 8B=C, 9B=G, 10B=D, 11B=A, 12B=E
#   Minor (A ring):  1A=G#, 2A=D#, 3A=A#, 4A=F,  5A=C,  6A=G,  7A=D, 8A=A,  9A=E, 10A=B, 11A=F#, 12A=C#
_CAMELOT_MAJOR = {
    # pitch_index: camelot_number
    11: 1,   # B → 1B
    6: 2,    # F# → 2B
    1: 3,    # C# → 3B
    8: 4,    # G# → 4B
    3: 5,    # D# → 5B
    10: 6,   # A# → 6B
    5: 7,    # F → 7B
    0: 8,    # C → 8B
    7: 9,    # G → 9B
    2: 10,   # D → 10B
    9: 11,   # A → 11B
    4: 12,   # E → 12B
}
_CAMELOT_MINOR = {
    8: 1,    # G# → 1A
    3: 2,    # D# → 2A
    10: 3,   # A# → 3A
    5: 4,    # F → 4A
    0: 5,    # C → 5A
    7: 6,    # G → 6A
    2: 7,    # D → 7A
    9: 8,    # A → 8A
    4: 9,    # E → 9A
    11: 10,  # B → 10A
    6: 11,   # F# → 11A
    1: 12,   # C# → 12A
}

# Reverse: pitch class → index (for KeyExtractor output like "A", "C#", "Ab")
_PITCH_TO_INDEX = {name: i for i, name in enumerate(KEYS)}
# Essentia KeyExtractor can return flats — map them to sharps
_FLAT_TO_SHARP = {"Ab": "G#", "Bb": "A#", "Db": "C#", "Eb": "D#", "Gb": "F#"}


def pitch_to_camelot(key_name: str, scale: str) -> str:
    """Convert essentia KeyExtractor output to Camelot notation (e.g. 'A','minor' → '8A')."""
    # Normalize flats → sharps (KeyExtractor may return "Ab" instead of "G#")
    key_name = _FLAT_TO_SHARP.get(key_name, key_name)
    idx = _PITCH_TO_INDEX.get(key_name)
    if idx is None:
        return "?"
    if scale == "major":
        num = _CAMELOT_MAJOR.get(idx, 0)
        return f"{num}B" if num else "?"
    else:  # minor
        num = _CAMELOT_MINOR.get(idx, 0)
        return f"{num}A" if num else "?"


def beat_grid_essentia(audio_path):
    """Beat positions + onsets via essentia BeatTrackerDegara."""
    loader = es.MonoLoader(filename=audio_path)
    audio = loader()
    sr = 44100  # essentia default

    # Beat tracker
    bt = es.BeatTrackerDegara()
    beats = bt(audio)
    beat_times = [round(float(b), 3) for b in beats]

    # Onset detection
    od = es.OnsetDetection(method='complex')
    onsets = []
    frame_size = 1024
    hop_size = 512
    for frame in es.FrameGenerator(audio, frameSize=frame_size, hopSize=hop_size):
        onset_val = od(frame, frame)
        onsets.append(float(onset_val))
    onset_times = [round(float(i * hop_size / sr), 3) for i, v in enumerate(onsets)
                   if v > 0.5]

    return beat_times, onset_times


def beat_grid_librosa(y, sr):
    """Beat positions + onsets via librosa."""
    # Beat positions
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr).tolist()
    beat_times = [round(float(b), 3) for b in beat_times]

    # Onset detection
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, backtrack=True)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr).tolist()
    onset_times = [round(float(o), 3) for o in onset_times]

    return beat_times, onset_times


def danceability_essentia(audio_path):
    """Danceability via essentia."""
    loader = es.MonoLoader(filename=audio_path)
    audio = loader()
    dance = es.Danceability()
    d = dance(audio)
    # Danceability returns (danceability, dfa) — both may be arrays
    d_val = float(d[0].mean()) if hasattr(d[0], 'mean') else float(d[0])
    c_val = float(d[1].mean()) if hasattr(d[1], 'mean') else float(d[1])
    return round(d_val, 4), round(c_val, 4)


def danceability_librosa(y, sr):
    """Danceability estimate via librosa: beat strength + spectral flux variation."""
    # Beat strength as proxy
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    beat_strength = float(tempo.item()) if hasattr(tempo, 'item') else float(tempo) if tempo else 0.0
    # Normalize to 0-1 range
    beat_score = min(1.0, max(0.0, beat_strength / 200.0)) if beat_strength else 0.5

    # Spectral flux variation (more variation = more danceable)
    spec = np.abs(librosa.stft(y))
    flux = np.diff(spec, axis=1)
    flux_var = float(np.var(flux))
    flux_score = min(1.0, flux_var * 100) if flux_var else 0.5

    danceability = round((beat_score * 0.6 + flux_score * 0.4), 4)
    confidence = 0.5  # librosa estimate is rough
    return danceability, confidence


def analyze_audio(filepath: str) -> dict:
    errors = []
    result = {
        "filepath": filepath,
        "duration": 0.0, "sample_rate": 0,
        "bpm": None, "key": None, "camelot": None, "key_strength": None,
        "loudness": None, "energy": None,
        "spectral_centroid": None, "zero_crossing_rate": None,
        "mfcc_mean": None,
        "danceability": None, "danceability_confidence": None,
        "beat_positions": [], "onset_times": [],
    }

    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail=f"File not found: {filepath}")

    try:
        y, sr = librosa.load(filepath, sr=None, mono=True)
        result["duration"] = float(librosa.get_duration(y=y, sr=sr))
        result["sample_rate"] = int(sr)
    except Exception as e:
        errors.append(f"load: {e}")
        return {**result, "errors": errors}

    # BPM
    try:
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        t = float(tempo[0]) if hasattr(tempo, 'size') and tempo.size > 0 else float(tempo) if tempo else None
        if t: result["bpm"] = round(t, 1)
    except Exception as e: errors.append(f"bpm: {e}")

    # Key with essentia KeyExtractor → pitch + scale + strength → Camelot
    try:
        if ESSENTIA:
            # Essentia KeyExtractor: returns (key_name, scale, strength)
            y32 = y.astype(np.float32)
            ke = es.KeyExtractor()
            key_name, scale, strength = ke(y32)
            result["key"] = f"{key_name} {scale}"  # e.g. "F# minor"
            result["camelot"] = pitch_to_camelot(key_name, scale)
            result["key_strength"] = round(float(strength), 4)
        else:
            # Fallback: chroma argmax (pitch class only, no mode)
            chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
            pitch_idx = int(np.argmax(chroma.mean(axis=1))) % 12
            result["key"] = KEYS[pitch_idx]
            result["camelot"] = None
            result["key_strength"] = None
    except Exception as e:
        errors.append(f"key: {e}")

    # Loudness
    try:
        rms = librosa.feature.rms(y=y)
        result["loudness"] = round(float(20 * np.log10(np.maximum(rms.mean(), 1e-10))), 1)
    except Exception as e: errors.append(f"loudness: {e}")

    # Energy — onset strength (spectral flux) normalized to [0, 1].
    # Measures track density/busyness, not raw amplitude.
    # Dense breakbeat → high, sparse ambient → low.
    # Independent from loudness (unlike raw RMS which has 0.97 correlation).
    try:
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        if len(onset_env) > 0:
            raw = float(np.mean(onset_env))
            # Normalize: onset_strength typically ranges 0.5–3.0 for music.
            # Map to [0, 1] linearly: (raw - 0.5) / 2.5
            result["energy"] = round(min(1.0, max(0.0, (raw - 0.5) / 2.5)), 4)
        else:
            result["energy"] = 0.0
    except Exception as e:
        errors.append(f"energy: {e}")

    # Spectral centroid
    try:
        cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        result["spectral_centroid"] = round(float(cent.mean()), 1)
    except Exception as e: errors.append(f"centroid: {e}")

    # ZCR
    try:
        zcr = librosa.feature.zero_crossing_rate(y)
        result["zero_crossing_rate"] = round(float(zcr.mean()), 6)
    except Exception as e: errors.append(f"zcr: {e}")

    # MFCC
    try:
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        result["mfcc_mean"] = [round(float(v), 2) for v in mfcc.mean(axis=1)]
    except Exception as e: errors.append(f"mfcc: {e}")

    # Beat grid + onsets
    if ESSENTIA:
        try:
            bt, ot = beat_grid_essentia(filepath)
            result["beat_positions"] = bt[:200]
            result["onset_times"] = ot[:200]
        except Exception as e: errors.append(f"beat_grid_essentia: {e}")
    if not result["beat_positions"]:
        try:
            bt, ot = beat_grid_librosa(y, sr)
            result["beat_positions"] = bt[:200]
            result["onset_times"] = ot[:200]
        except Exception as e: errors.append(f"beat_grid_librosa: {e}")

    # BPM correction: cross-check librosa BPM against beat_positions.
    # Half-time / double-time detection is a common librosa artifact.
    # When the ratio is ~2x or ~0.5x, trust the beat-grid BPM.
    if result["bpm"] and result["beat_positions"] and len(result["beat_positions"]) >= 3:
        try:
            beats = result["beat_positions"]
            intervals = [beats[i+1] - beats[i] for i in range(len(beats)-1)]
            # Filter outliers
            med = sorted(intervals)[len(intervals)//2]
            clean = [x for x in intervals if 0.3*med < x < 3.0*med]
            if clean:
                avg_interval = sum(clean) / len(clean)
                grid_bpm = 60.0 / avg_interval if avg_interval > 0 else 0
                if grid_bpm > 0:
                    ratio = result["bpm"] / grid_bpm
                    if 0.45 < ratio < 0.78 or 1.30 < ratio < 2.20:
                        # Half-time or double-time artifact — use grid BPM
                        result["bpm"] = round(grid_bpm, 1)
        except Exception:
            pass  # correction is best-effort, never fail the analysis

    # Danceability — prefer librosa (essentia danceability not normalized in 2.1b6)
    if ESSENTIA:
        try:
            # Use DFA from essentia as confidence, librosa for the score
            d, c = danceability_librosa(y, sr)
            result["danceability"] = d
            # Boost confidence with essentia DFA if available
            loader = es.MonoLoader(filename=filepath)
            audio = loader()
            dance = es.Danceability()
            dfa = float(dance(audio)[1].mean()) if hasattr(dance(audio)[1], 'mean') else 0.5
            result["danceability_confidence"] = round(min(1.0, max(0.0, 1.0 - dfa * 10)), 4)
        except Exception as e:
            errors.append(f"danceability: {e}")
    if result["danceability"] is None:
        try:
            d, c = danceability_librosa(y, sr)
            result["danceability"] = d
            result["danceability_confidence"] = c
        except Exception as e:
            errors.append(f"danceability_librosa: {e}")

    result["errors"] = errors
    return result


@app.get("/health")
def health():
    return {"status": "ok", "librosa": librosa.__version__, "essentia": ESSENTIA}


class AnalyzeRequest(BaseModel):
    filepath: str


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    t0 = time.time()
    data = analyze_audio(req.filepath)
    data["_analysis_time_ms"] = round((time.time() - t0) * 1000)
    return data


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8777)
