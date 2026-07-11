"""Audio feature dataclasses for murmur."""

from __future__ import annotations
import hashlib

from dataclasses import dataclass
from typing import List
import colorsys

import numpy as np


@dataclass
class GlobalAudioFeatures:
    """
    Macro-level audio features extracted from the entire track (or clip).
    Used to drive global generative image parameters.
    """

    bpm: float  # Tempo in beats per minute
    mean_rms: float  # Overall loudness
    mean_centroid: float  # Overall brightness
    mean_bandwidth: float  # Overall texture/width
    dominant_chroma: int  # Overall dominant musical key/pitch class


def audio_to_hex(features: GlobalAudioFeatures) -> str:
    """Map global audio features to a synesthesia base hex color."""
    # Hue: Map dominant chroma (0-11) directly to hue (0.0 - 1.0)
    hue = features.dominant_chroma / 12.0

    # Lightness: louder (RMS) = brighter. Range [0.2, 0.8] to avoid pure black/white
    lightness = 0.2 + (features.mean_rms * 0.6)

    # Saturation: High bandwidth (noisy/wide) = lower saturation (pastel/grey)
    # Low bandwidth (pure/narrow) = high saturation (vibrant)
    saturation = 1.0 - (features.mean_bandwidth * 0.6)
    saturation = max(0.1, min(1.0, saturation))

    r, g, b = colorsys.hls_to_rgb(hue, lightness, saturation)
    return "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))


@dataclass
class AudioFrame:
    """
    Normalized audio features for a single output frame (all values in [0, 1]).

    These are computed by the analyzer and resampled to match the GIF frame rate.
    """

    frame_idx: int

    # Amplitude
    rms: float  # root-mean-square energy (loudness)

    # Spectral
    spectral_centroid: float  # "brightness" — center of mass of spectrum
    spectral_bandwidth: float  # "width" of the spectrum
    spectral_flatness: float  # 0=tonal (pure tone), 1=noise-like

    # Rhythm
    onset_strength: float  # strength of onset at this moment
    is_beat: bool  # True if a beat falls on this frame

    # Pitch
    chroma: np.ndarray  # shape (12,) — normalized chroma vector (pitch classes)
    dominant_chroma: int  # index 0–11 of most active pitch class

    # Noisiness
    zero_crossing_rate: float  # rate at which waveform crosses zero

    @property
    def dominant_hue_shift(self) -> float:
        """
        Maps the dominant chroma (0–11 pitch classes) to a hue shift in [0, 1].
        C=0, C#=1/12, D=2/12, ... B=11/12.
        """
        return self.dominant_chroma / 12.0


@dataclass
class AudioTimeline:
    """
    Full sequence of AudioFrames for the rendered clip, indexed by frame number.
    """

    frames: List[AudioFrame]
    sample_rate: int
    duration: float
    fps: int

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx: int) -> AudioFrame:
        return self.frames[idx]

    def __iter__(self):
        return iter(self.frames)


# ---------------------------------------------------------------------------
# Song DNA — the continuous 5-axis fingerprint that drives murmur v2
# ---------------------------------------------------------------------------


@dataclass
class SongDNA:
    """
    A 5-axis continuous fingerprint extracted from a song.
    All axes are floats in [0.0, 1.0]. No categories, no presets.
    These 5 values, combined with dominant_chroma and seed, fully determine
    every visual parameter of the generated mandala.
    """

    rhythm: float  # 0=steady/regular beat, 1=chaotic/irregular
    density: float  # 0=sparse/quiet, 1=full/loud mix
    brightness: float  # 0=dark/muffled, 1=bright/shimmery
    complexity: float  # 0=pure/simple tone, 1=complex/dissonant
    harmonic: float  # 0=percussive/noisy, 1=tonal/melodic

    # Supporting data
    dominant_chroma: int  # 0–11: musical key (C=0, C#=1, ... B=11)
    bpm: float  # beats per minute
    seed: int  # deterministic reproducibility seed

    def describe(self) -> str:
        return (
            f"DNA(rhythm={self.rhythm:.3f} density={self.density:.3f} "
            f"brightness={self.brightness:.3f} complexity={self.complexity:.3f} "
            f"harmonic={self.harmonic:.3f} key={self.dominant_chroma} bpm={self.bpm:.1f})"
        )
