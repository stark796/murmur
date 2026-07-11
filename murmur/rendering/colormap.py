"""
Trail map → RGB frame compositor for murmur.

Maps the Physarum trail map (float32 [0, 1]) to an RGB image using
the palette colormap, with audio-driven brightness modulation.
"""

from __future__ import annotations

import numpy as np


def trail_to_rgb(
    trail_map: np.ndarray,
    colormap: np.ndarray,
    rms: float = 0.5,
    gamma: float = 1.6,
) -> np.ndarray:
    """
    Convert a (H, W) float32 trail map to a (H, W, 3) uint8 RGB image.

    Uses log-scale normalization so both sparse edges and dense cores
    show color detail — prevents saturation to uniform grey/white.

    Parameters
    ----------
    trail_map : (H, W) float32, values in [0, 1]
    colormap  : (256, 3) uint8 — maps trail value to RGB
    rms       : normalized audio loudness [0, 1] — boosts brightness on peaks
    gamma     : power applied after log-norm (higher = more contrast in dim areas)

    Returns
    -------
    rgb : (H, W, 3) uint8
    """
    t = np.clip(trail_map, 0.0, 1.0)

    # Log-scale: compress the high end, expand the low end
    # log1p(t * k) / log1p(k)  maps [0,1] → [0,1] non-linearly
    k = 12.0  # compression strength — higher = more detail in sparse regions
    t_log = np.log1p(t * k) / np.log1p(k)

    # Gamma correction for final contrast
    t_gamma = np.power(t_log, 1.0 / gamma).astype(np.float32)

    # Audio RMS slightly lifts the exposure on loud moments
    exposure = 1.0 + rms * 0.2
    t_final = np.clip(t_gamma * exposure, 0.0, 1.0)

    # Map to 8-bit colormap indices
    indices = (t_final * 255).astype(np.uint8)

    # Look up in colormap
    rgb = colormap[indices]  # (H, W, 3) uint8

    return rgb
