"""
Color palette builder for murmur.

Takes the user's base hex color and generates a full painterly palette:
  - A near-black background shade
  - Mid tones around the base color
  - Bright highlight / accent
  - Audio chroma shifts the active hue over time
"""

from __future__ import annotations

import colorsys
import math
from typing import List, Tuple

import numpy as np


RGB = Tuple[int, int, int]


def hex_to_hsl(hex_color: str) -> Tuple[float, float, float]:
    """Convert '#rrggbb' → (h, s, l) all in [0, 1]."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16) / 255.0, int(h[2:4], 16) / 255.0, int(h[4:6], 16) / 255.0
    _h, _l, _s = colorsys.rgb_to_hls(r, g, b)
    return _h, _s, _l


def hsl_to_rgb(h: float, s: float, l: float) -> RGB:
    """Convert HSL → (R, G, B) integers in [0, 255]."""
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return (
        max(0, min(255, int(r * 255))),
        max(0, min(255, int(g * 255))),
        max(0, min(255, int(b * 255))),
    )


def build_colormap(
    base_hex: str,
    n_stops: int = 256,
    hue_shift: float = 0.0,
) -> np.ndarray:
    """
    Build a (n_stops, 3) uint8 colormap from trail_map value [0→1].

    The colormap goes from:
      0.0 → dark background (very low lightness, base hue)
      0.3 → mid-tone base color
      0.7 → bright saturated highlight
      1.0 → near-white tinted accent

    Parameters
    ----------
    base_hex  : user's base color e.g. '#4a1942'
    n_stops   : number of colormap entries (256 = 8-bit precision)
    hue_shift : shift in hue [0, 1] driven by audio chroma
    """
    h, s, l = hex_to_hsl(base_hex)

    # Apply audio-driven hue shift (wrap around)
    h = (h + hue_shift) % 1.0

    # Complementary hue (opposite on color wheel)
    h_comp = (h + 0.5) % 1.0

    # Ensure lightness doesn't clip to white too early for bright base colors
    l_base = min(l, 0.65)

    # Define 5 control points: (trail_value, hsl)
    #   Dark background → shadows → base → bright → highlight
    stops: List[Tuple[float, float, float, float]] = [
        # (t,  hue,     sat,    lightness)
        (0.00, h, s * 0.3, 0.04),  # near-black with base hue tint
        (0.30, h, s * 0.7, l_base * 0.5),  # dark mid
        (0.60, h, s, l_base),  # base color
        (0.85, h_comp, s * 0.8, min(l_base * 1.3, 0.85)),  # bright complementary
        (1.00, h_comp, 0.2, 0.95),  # near-white highlight
    ]

    colormap = np.zeros((n_stops, 3), dtype=np.uint8)

    for i in range(n_stops):
        t = i / (n_stops - 1)

        # Find surrounding stops
        s0 = stops[0]
        s1 = stops[-1]
        for j in range(len(stops) - 1):
            if stops[j][0] <= t <= stops[j + 1][0]:
                s0, s1 = stops[j], stops[j + 1]
                break

        # Interpolate
        span = s1[0] - s0[0]
        alpha = (t - s0[0]) / span if span > 1e-8 else 0.0
        # Smooth step for nicer gradient
        alpha = alpha * alpha * (3 - 2 * alpha)

        hi = s0[1] + (s1[1] - s0[1]) * alpha
        # Hue: interpolate through shorter arc on color wheel
        dh = (s1[1] - s0[1] + 1.5) % 1.0 - 0.5
        hi = (s0[1] + dh * alpha) % 1.0

        si = s0[2] + (s1[2] - s0[2]) * alpha
        li_val = s0[3] + (s1[3] - s0[3]) * alpha
        li_val = max(0.0, min(1.0, li_val))
        si = max(0.0, min(1.0, si))

        colormap[i] = hsl_to_rgb(hi, si, li_val)

    return colormap  # shape (256, 3) uint8


class PaletteManager:
    """
    Manages the color palette for a render, allowing frame-by-frame hue shift.
    """

    def __init__(self, base_hex: str) -> None:
        self.base_hex = base_hex
        self._current_hue_shift = 0.0
        self._current_colormap = build_colormap(base_hex, hue_shift=0.0)

    def update(self, dominant_chroma: int, chroma_strength: float) -> None:
        """
        Update the active hue shift based on audio chroma.

        Parameters
        ----------
        dominant_chroma  : 0–11 (pitch class)
        chroma_strength  : 0–1 (how confident the chroma detection is)
        """
        # Map pitch class to a small hue shift (±0.08 around base)
        target_shift = (dominant_chroma / 12.0) * 0.15 * chroma_strength
        # Smooth interpolation to avoid sudden jumps
        self._current_hue_shift += (target_shift - self._current_hue_shift) * 0.1

    def get_colormap(self) -> np.ndarray:
        """Return the current (256, 3) uint8 colormap."""
        # Rebuild only if hue shift changed meaningfully
        self._current_colormap = build_colormap(
            self.base_hex,
            hue_shift=self._current_hue_shift,
        )
        return self._current_colormap
