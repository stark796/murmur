"""
murmur v2 — Song-driven 5-stop color palette.

Every palette is derived entirely from the SongDNA — no manual
color input. The dominant musical key maps to a base hue; the
other axes control saturation, lightness, and accent hue.

Output: a (256, 3) uint8 gradient array for direct index lookup.
"""

from __future__ import annotations

import colorsys
import numpy as np
from murmur.audio.features import SongDNA


def _hsl_to_rgb255(h: float, s: float, l: float) -> tuple[int, int, int]:
    """HSL → (R, G, B) as uint8 ints."""
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return (
        int(np.clip(r * 255, 0, 255)),
        int(np.clip(g * 255, 0, 255)),
        int(np.clip(b * 255, 0, 255)),
    )


def build_song_palette(dna: SongDNA, n_stops: int = 256) -> np.ndarray:
    """
    Build a 5-stop perceptual gradient from SongDNA.

    CRITICAL FRONKONSTIN INSIGHT: The palette goes BRIGHT → DARK.
    - t=0.00 (void / low density)  → BRIGHT background (cream, teal, coral)
    - t=1.00 (dense / high pheromone) → DARK (black, charcoal, deep color)

    This is the opposite of typical "glow on black" rendering. It creates
    the signature Fronkonstin look: filled painted blobs with dark networks
    carving through bright colored regions.
    """
    key_mod = dna.dominant_chroma % 4

    if key_mod == 0:
        # Coral Sunset: Cream → Coral → Red → Maroon → Black
        h1, s1, l1 = 0.08, 0.3, 0.90  # Cream/off-white background
        h2, s2, l2 = 0.05, 0.8, 0.70  # Coral / salmon
        h3, s3, l3 = 0.02, 0.9, 0.50  # Bright red
        h4, s4, l4 = 0.98, 0.8, 0.25  # Deep maroon
        h5, s5, l5 = 0.70, 0.6, 0.05  # Near-black (dark blue-ish)
    elif key_mod == 1:
        # Teal Ocean: Light cyan → Teal → Blue → Navy → Charcoal
        h1, s1, l1 = 0.50, 0.4, 0.85  # Light cyan background
        h2, s2, l2 = 0.48, 0.7, 0.65  # Bright teal
        h3, s3, l3 = 0.55, 0.8, 0.45  # Rich blue
        h4, s4, l4 = 0.65, 0.7, 0.20  # Deep navy
        h5, s5, l5 = 0.60, 0.3, 0.05  # Near-black
    elif key_mod == 2:
        # Warm Gold: Pale yellow → Gold → Orange → Burnt sienna → Dark
        h1, s1, l1 = 0.12, 0.5, 0.88  # Pale warm background
        h2, s2, l2 = 0.10, 0.8, 0.65  # Gold
        h3, s3, l3 = 0.06, 0.9, 0.50  # Bright orange
        h4, s4, l4 = 0.02, 0.7, 0.25  # Burnt sienna
        h5, s5, l5 = 0.95, 0.4, 0.05  # Near-black (warm)
    else:
        # Cool Lavender: Pale lilac → Violet → Purple → Plum → Dark
        h1, s1, l1 = 0.78, 0.3, 0.88  # Pale lavender background
        h2, s2, l2 = 0.80, 0.6, 0.65  # Soft violet
        h3, s3, l3 = 0.82, 0.8, 0.45  # Rich purple
        h4, s4, l4 = 0.85, 0.7, 0.20  # Deep plum
        h5, s5, l5 = 0.75, 0.4, 0.05  # Near-black

    # Tweak saturation based on harmonic content
    s_mod = 0.6 + dna.harmonic * 0.4

    stops = [
        (0.00, h1, s1 * s_mod, l1),  # Bright void
        (0.15, h2, s2 * s_mod, l2),  # Mid-light
        (0.40, h3, s3 * s_mod, l3),  # Vivid mid
        (0.70, h4, s4 * s_mod, l4),  # Deep shadow
        (1.00, h5, s5 * s_mod, l5),  # Near-black dense core
    ]

    palette = np.zeros((n_stops, 3), dtype=np.uint8)

    for i in range(n_stops):
        t = i / max(n_stops - 1, 1)

        # Find surrounding stops
        left, right = stops[0], stops[-1]
        for j in range(len(stops) - 1):
            if stops[j][0] <= t <= stops[j + 1][0]:
                left, right = stops[j], stops[j + 1]
                break

        span = right[0] - left[0]
        alpha = (t - left[0]) / span if span > 1e-9 else 0.0
        alpha = alpha * alpha * (3.0 - 2.0 * alpha)  # smoothstep

        # Hue interpolation via shortest arc on color wheel
        dh = (right[1] - left[1] + 1.5) % 1.0 - 0.5
        h = (left[1] + dh * alpha) % 1.0
        s = left[2] + (right[2] - left[2]) * alpha
        l = left[3] + (right[3] - left[3]) * alpha

        palette[i] = _hsl_to_rgb255(h, np.clip(s, 0, 1), np.clip(l, 0, 1))

    return palette
