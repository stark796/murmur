"""Configuration dataclasses and vibe presets for murmur."""

from __future__ import annotations

import colorsys
from dataclasses import dataclass, field
from typing import Tuple, Optional

# ---------------------------------------------------------------------------
# Vibe presets — named aliases for the float vibe parameter
# ---------------------------------------------------------------------------
VIBE_PRESETS: dict[str, float] = {
    "dreamy": 0.1,
    "organic": 0.4,
    "electric": 0.7,
    "chaotic": 0.95,
}


def resolve_vibe(vibe: str | float) -> float:
    """Accept a named preset string or a float in [0, 1]."""
    if isinstance(vibe, str):
        key = vibe.lower()
        if key in VIBE_PRESETS:
            return VIBE_PRESETS[key]
        try:
            v = float(key)
        except ValueError:
            raise ValueError(
                f"Unknown vibe preset '{vibe}'. "
                f"Choose from: {list(VIBE_PRESETS.keys())}"
            )
    else:
        v = float(vibe)
    if not 0.0 <= v <= 1.0:
        raise ValueError(f"Vibe must be in [0.0, 1.0], got {v}")
    return v


def parse_hex_color(hex_color: str) -> Tuple[int, int, int]:
    """Parse a hex string like '#4a1942' → (R, G, B) ints."""
    h = hex_color.lstrip("#")
    if len(h) != 6:
        raise ValueError(f"Expected 6-digit hex color, got '{hex_color}'")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


# ---------------------------------------------------------------------------
# Per-frame simulation parameters (derived from audio + vibe each frame)
# ---------------------------------------------------------------------------
@dataclass
class SimParams:
    """Attractor simulation parameters for a single frame, derived from audio."""

    a: float
    b: float
    c: float
    d: float


# ---------------------------------------------------------------------------
# Main configuration
# ---------------------------------------------------------------------------
@dataclass
class MurmurConfig:
    """Top-level configuration for a murmur render."""

    audio_path: str
    vibe: float  # resolved 0.0–1.0
    color: Optional[str] = (
        None  # hex string, e.g. "#4a1942". If None, generated via auto-color
    )
    engine: str = "field"  # field = abstract math field, veins = vascular art
    start_time: float = 0.0  # seconds into the audio to start
    duration: Optional[float] = None  # seconds of output (None = full length)
    fps: int = 24
    resolution: int = 512
    output_path: str = "output.gif"

    # Derived
    rgb: Tuple[int, int, int] = field(default=(0, 0, 0), init=False)

    def __post_init__(self) -> None:
        if self.color:
            self.rgb = parse_hex_color(self.color)

    @property
    def total_frames(self) -> int:
        if self.duration is None:
            return 0  # not known ahead of time
        return int(self.duration * self.fps)

    @property
    def hsl(self) -> Tuple[float, float, float]:
        """Base color as HSL (h in [0,1], s in [0,1], l in [0,1])."""
        r, g, b = (c / 255.0 for c in self.rgb)
        h, l, s = colorsys.rgb_to_hls(r, g, b)
        return h, s, l

    def make_sim_params(
        self,
        rms: float,
        spectral_centroid: float,
        onset_strength: float,
        spectral_bandwidth: float,
        zero_crossing_rate: float,
    ) -> SimParams:
        """
        Map normalized audio features (all in [0, 1]) + vibe → SimParams.
        """
        v = self.vibe

        # Base Clifford parameters (a classic beautiful starting shape)
        a_base = -1.4 + v * 0.5
        b_base = 1.6 - v * 0.2
        c_base = 1.0 + v * 0.3
        d_base = 0.7 - v * 0.5

        # Audio Modulation:
        # Loudness (rms) expands/contracts the folds
        a = a_base + rms * 0.3
        b = b_base - rms * 0.2

        # Timbre (centroid) twists the ribbons
        c = c_base + spectral_centroid * 0.5

        # Beats (onset) cause sharp rhythmic shape shifts
        d = d_base + onset_strength * 0.8

        return SimParams(a=a, b=b, c=c, d=d)
