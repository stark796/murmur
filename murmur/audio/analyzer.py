"""
Audio analyzer for murmur.

Loads an audio file, extracts features with librosa, and resamples them
to match the GIF frame rate — producing an AudioTimeline of per-frame features.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

_NUMBA_CACHE_DIR = Path(os.environ.get("NUMBA_CACHE_DIR", "/tmp/murmur_numba_cache"))
_NUMBA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("NUMBA_CACHE_DIR", str(_NUMBA_CACHE_DIR))

import librosa

from murmur.audio.features import AudioFrame, AudioTimeline


def _normalize(arr: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Min-max normalize an array to [0, 1]."""
    mn, mx = arr.min(), arr.max()
    if mx - mn < eps:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)


def _resample_to_frames(feature: np.ndarray, n_frames: int) -> np.ndarray:
    """
    Resample a 1-D feature array (in librosa frame-rate space) to n_frames
    output frames using linear interpolation.
    """
    n_src = len(feature)
    x_src = np.linspace(0, 1, n_src)
    x_dst = np.linspace(0, 1, n_frames)
    return np.interp(x_dst, x_src, feature)


def _resample_chroma(chroma: np.ndarray, n_frames: int) -> np.ndarray:
    """
    Resample a (12, T) chroma matrix to (n_frames, 12).
    """
    n_src = chroma.shape[1]
    x_src = np.linspace(0, 1, n_src)
    x_dst = np.linspace(0, 1, n_frames)
    out = np.zeros((n_frames, 12), dtype=np.float32)
    for i in range(12):
        out[:, i] = np.interp(x_dst, x_src, chroma[i])
    return out


class AudioAnalyzer:
    """
    Extracts per-frame audio features from an audio file for a given time window.

    Usage:
        analyzer = AudioAnalyzer(audio_path, start_time, duration, fps)
        timeline = analyzer.analyze()
    """

    def __init__(
        self,
        audio_path: str,
        start_time: float,
        duration: float | None,
        fps: int,
    ) -> None:
        self.audio_path = audio_path
        self.start_time = start_time
        self.duration = duration
        self.fps = fps
        self.n_frames = int(duration * fps) if duration else 0

    def analyze_global(self) -> GlobalAudioFeatures:
        """
        Analyze the audio globally and return macro-level features for a static render.
        """
        from murmur.audio.features import GlobalAudioFeatures

        print(f"  Loading global audio: {self.audio_path}")

        y, sr = librosa.load(self.audio_path, sr=None, mono=True)
        print(f"  Sample rate: {sr} Hz, {len(y)} samples")

        bpm, _ = librosa.beat.beat_track(y=y, sr=sr)
        # librosa 0.10+ beat_track returns a 1D array for tempo if not aggregated, but usually a scalar or array of size 1
        bpm = float(np.mean(bpm)) if isinstance(bpm, np.ndarray) else float(bpm)

        rms = np.mean(librosa.feature.rms(y=y))
        centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))

        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        mean_chroma = np.mean(chroma, axis=1)
        dominant_chroma = int(np.argmax(mean_chroma))

        # Normalize continuous features roughly based on typical ranges
        # RMS: typically 0 to 0.3 for loud tracks
        norm_rms = min(rms / 0.3, 1.0)
        # Centroid: typical 500 to 4000 Hz
        norm_centroid = np.clip((centroid - 500) / 3500.0, 0.0, 1.0)
        # Bandwidth: typical 1000 to 3500 Hz
        norm_bandwidth = np.clip((bandwidth - 1000) / 2500.0, 0.0, 1.0)

        return GlobalAudioFeatures(
            bpm=bpm,
            mean_rms=norm_rms,
            mean_centroid=norm_centroid,
            mean_bandwidth=norm_bandwidth,
            dominant_chroma=dominant_chroma,
        )

    def analyze_dna(self) -> "SongDNA":
        """
        Extract the 5-axis Song DNA fingerprint from the audio.
        Returns a SongDNA with all axes in [0, 1] and a deterministic seed.
        Uses: onset variance, RMS, bandwidth, centroid, MFCCs, spectral
        contrast, HPSS, and chroma entropy.
        """
        from murmur.audio.features import SongDNA
        import hashlib

        print(f"  Extracting Song DNA from: {self.audio_path}")
        y, sr = librosa.load(self.audio_path, sr=None, mono=True)
        print(f"  Loaded {len(y)/sr:.1f}s @ {sr}Hz")

        # --- BPM ---
        bpm_raw, _ = librosa.beat.beat_track(y=y, sr=sr)
        bpm = float(np.mean(bpm_raw))

        # --- Onset strength → rhythm (variance = chaos) ---
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        onset_mean = np.mean(onset_env) + 1e-8
        onset_var = np.std(onset_env) / onset_mean
        rhythm = float(np.clip(onset_var / 3.5, 0.0, 1.0))

        # --- RMS + bandwidth → density ---
        rms_raw = librosa.feature.rms(y=y)[0]
        mean_rms = float(np.mean(rms_raw))
        norm_rms = np.clip(mean_rms / 0.25, 0.0, 1.0)

        bw_raw = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        norm_bw = np.clip((np.mean(bw_raw) - 800.0) / 2800.0, 0.0, 1.0)
        density = float(np.clip((norm_rms * 0.6 + norm_bw * 0.4), 0.0, 1.0))

        # --- Spectral centroid → brightness ---
        centroid_raw = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        brightness = float(np.clip((np.mean(centroid_raw) - 300.0) / 4200.0, 0.0, 1.0))

        # --- MFCCs + spectral contrast → complexity ---
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_var = float(np.mean(np.var(mfccs, axis=1)))
        norm_mfcc = np.clip(mfcc_var / 1500.0, 0.0, 1.0)

        contrast_raw = librosa.feature.spectral_contrast(y=y, sr=sr)
        norm_contrast = np.clip(np.mean(contrast_raw) / 35.0, 0.0, 1.0)
        complexity = float(np.clip((norm_mfcc * 0.5 + norm_contrast * 0.5), 0.0, 1.0))

        # --- HPSS + chroma entropy → harmonic ---
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        h_power = float(np.mean(y_harmonic**2))
        p_power = float(np.mean(y_percussive**2))
        hpss_ratio = h_power / (h_power + p_power + 1e-9)

        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        mean_chroma = np.mean(chroma, axis=1)
        mean_chroma_norm = mean_chroma / (mean_chroma.sum() + 1e-9)
        chroma_entropy = float(
            -np.sum(mean_chroma_norm * np.log2(mean_chroma_norm + 1e-9)) / np.log2(12)
        )
        dominant_chroma = int(np.argmax(mean_chroma))

        # High hpss_ratio (tonal) + low chroma_entropy (one key dominates) = harmonic
        harmonic = float(
            np.clip((hpss_ratio * 0.6 + (1.0 - chroma_entropy) * 0.4), 0.0, 1.0)
        )

        # --- Deterministic seed from fingerprint ---
        fingerprint = (
            f"{bpm:.2f}|{mean_rms:.6f}|{np.mean(centroid_raw):.2f}|{dominant_chroma}"
        )
        seed = int(hashlib.sha256(fingerprint.encode()).hexdigest()[:16], 16) % (2**31)

        dna = SongDNA(
            rhythm=rhythm,
            density=density,
            brightness=brightness,
            complexity=complexity,
            harmonic=harmonic,
            dominant_chroma=dominant_chroma,
            bpm=bpm,
            seed=seed,
        )
        print(f"  {dna.describe()}")
        return dna

    def analyze(self) -> AudioTimeline:
        """
        Load audio, extract features, resample to frame rate.
        Returns an AudioTimeline ready to drive the simulation.
        """
        print(f"  Loading audio: {self.audio_path}")
        dur_str = f"{self.start_time + self.duration:.1f}s" if self.duration else "End"
        print(f"  Clip: {self.start_time:.1f}s → {dur_str}")

        # Load the audio slice
        y, sr = librosa.load(
            self.audio_path,
            offset=self.start_time,
            duration=self.duration,
            sr=None,  # keep native sample rate
            mono=True,
        )
        print(f"  Sample rate: {sr} Hz, {len(y)} samples")

        # ---- Extract raw features ----
        # Hop length for librosa analysis (controls temporal resolution)
        hop_length = 512

        # RMS energy
        rms_raw = librosa.feature.rms(y=y, hop_length=hop_length)[0]

        # Spectral centroid
        centroid_raw = librosa.feature.spectral_centroid(
            y=y, sr=sr, hop_length=hop_length
        )[0]

        # Spectral bandwidth
        bandwidth_raw = librosa.feature.spectral_bandwidth(
            y=y, sr=sr, hop_length=hop_length
        )[0]

        # Spectral flatness
        flatness_raw = librosa.feature.spectral_flatness(y=y, hop_length=hop_length)[0]

        # Zero crossing rate
        zcr_raw = librosa.feature.zero_crossing_rate(y, hop_length=hop_length)[0]

        # Onset strength envelope
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)

        # Beat tracking
        _, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)
        beat_mask = np.zeros(len(onset_env), dtype=bool)
        valid_beats = beat_frames[beat_frames < len(onset_env)]
        beat_mask[valid_beats] = True

        # Chroma (12 pitch classes)
        chroma_raw = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length)
        # shape: (12, T_librosa)

        # ---- Normalize features to [0, 1] ----
        rms_norm = _normalize(rms_raw)
        centroid_norm = _normalize(centroid_raw)
        bandwidth_norm = _normalize(bandwidth_raw)
        flatness_norm = _normalize(flatness_raw)
        zcr_norm = _normalize(zcr_raw)
        onset_norm = _normalize(onset_env)
        # Chroma: normalize each row independently
        chroma_norm = np.zeros_like(chroma_raw)
        for i in range(12):
            chroma_norm[i] = _normalize(chroma_raw[i])

        # ---- Resample everything to n_frames ----
        n = self.n_frames
        rms_r = _resample_to_frames(rms_norm, n)
        centroid_r = _resample_to_frames(centroid_norm, n)
        bandwidth_r = _resample_to_frames(bandwidth_norm, n)
        flatness_r = _resample_to_frames(flatness_norm, n)
        zcr_r = _resample_to_frames(zcr_norm, n)
        onset_r = _resample_to_frames(onset_norm, n)
        beat_r = _resample_to_frames(beat_mask.astype(float), n) > 0.5
        chroma_r = _resample_chroma(chroma_norm, n)

        # ---- Build AudioFrame list ----
        frames = []
        for i in range(n):
            chroma_vec = chroma_r[i]  # shape (12,)
            dom_chroma = int(np.argmax(chroma_vec))
            frames.append(
                AudioFrame(
                    frame_idx=i,
                    rms=float(rms_r[i]),
                    spectral_centroid=float(centroid_r[i]),
                    spectral_bandwidth=float(bandwidth_r[i]),
                    spectral_flatness=float(flatness_r[i]),
                    onset_strength=float(onset_r[i]),
                    is_beat=bool(beat_r[i]),
                    chroma=chroma_vec.astype(np.float32),
                    dominant_chroma=dom_chroma,
                    zero_crossing_rate=float(zcr_r[i]),
                )
            )

        print(f"  Extracted {n} audio frames @ {self.fps} fps")
        return AudioTimeline(
            frames=frames,
            sample_rate=sr,
            duration=self.duration,
            fps=self.fps,
        )
