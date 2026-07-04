"""
test_analysis.py — Integration tests for audio analysis.

Uses synthetic WAV files (sine waves) to verify the music-analysis API.
These tests require the music-analysis Docker container running on :8777,
or the server.py to be importable locally.

Marked with @pytest.mark.integration — run with: pytest -m integration
"""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def sine_wav_440():
    """Generate a 5-second 440 Hz sine wave (A4) as a temp WAV file."""
    try:
        import soundfile as sf
    except ImportError:
        pytest.skip("soundfile not installed")

    sr = 22050
    duration = 5.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    y = np.sin(2 * np.pi * 440.0 * t).astype(np.float32)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, y, sr)
        yield f.name
    os.unlink(f.name)


@pytest.fixture
def sine_wav_120bpm():
    """Generate a 5-second audio with 120 BPM pulse (transient every 0.5s)."""
    try:
        import soundfile as sf
    except ImportError:
        pytest.skip("soundfile not installed")

    sr = 22050
    duration = 5.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    beat_interval = 60.0 / 120.0
    y = np.zeros_like(t).astype(np.float32)
    for i in range(int(duration / beat_interval)):
        start = int(i * beat_interval * sr)
        end = min(start + int(0.05 * sr), len(y))
        y[start:end] = np.sin(2 * np.pi * 200 * t[start:end]) * np.exp(
            -t[start:end] * 30
        )

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, y.astype(np.float32), sr)
        yield f.name
    os.unlink(f.name)


@pytest.mark.integration
class TestAnalyzeAudio:
    """Tests that require the music-analysis API or server.py to be available."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        """Try to import analyze_audio; skip if unavailable."""
        try:
            _ma = (
                Path(__file__).resolve().parent.parent.parent
                / "Hermes"
                / "data"
                / "music-analysis"
            )
            if _ma.exists():
                sys.path.insert(0, str(_ma))
            from server import analyze_audio

            self.analyze_audio = analyze_audio
        except (ImportError, ModuleNotFoundError):
            pytest.skip("music-analysis server.py not available")

    def test_returns_dict(self, sine_wav_440):
        result = self.analyze_audio(sine_wav_440)
        assert isinstance(result, dict)
        assert "filepath" in result

    def test_has_duration(self, sine_wav_440):
        result = self.analyze_audio(sine_wav_440)
        assert result["duration"] > 0

    def test_has_sample_rate(self, sine_wav_440):
        result = self.analyze_audio(sine_wav_440)
        assert result["sample_rate"] > 0

    def test_bpm_is_reasonable(self, sine_wav_120bpm):
        result = self.analyze_audio(sine_wav_120bpm)
        bpm = result.get("bpm")
        assert bpm is not None, "BPM should not be None"
        assert 60 < bpm < 200, f"BPM {bpm} out of range"

    def test_energy_in_range(self, sine_wav_440):
        result = self.analyze_audio(sine_wav_440)
        energy = result.get("energy")
        assert energy is not None
        assert 0.0 <= energy <= 1.0

    def test_mfcc_has_coeffs(self, sine_wav_440):
        result = self.analyze_audio(sine_wav_440)
        mfcc = result.get("mfcc_mean")
        assert mfcc is not None
        assert len(mfcc) >= 3

    def test_beat_positions_not_empty(self, sine_wav_120bpm):
        result = self.analyze_audio(sine_wav_120bpm)
        beats = result.get("beat_positions", [])
        assert len(beats) > 0, "Should detect beats on 120 BPM pulse"
