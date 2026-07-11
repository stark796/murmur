"""
murmur v2 — Trail map rendering.

Replicates Fronkonstin's R rendering:

  ggplot(data = df %>% filter(v>0), aes(x = x, y = y, fill = log(v))) + 
    geom_raster(interpolate = TRUE) +
    scale_fill_gradientn(colours = colors)

The key trick: geom_raster(interpolate=TRUE) does bilinear interpolation
of the raw raster data. We replicate this by blurring the raw trail_map
BEFORE log and palette lookup. This physically thickens thin 1px trails
into soft gradient regions.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter


def render_trail(
    trail_map: np.ndarray,
    palette: np.ndarray,
) -> np.ndarray:
    """
    Convert a (H, W) float32 trail map to a (H, W, 3) uint8 RGB image.
    """
    H, W = trail_map.shape

    # Step 1: Blur the RAW trail map to thicken thin lines
    # This is the equivalent of geom_raster(interpolate=TRUE) which
    # does bilinear interpolation on the data before color mapping.
    # sigma=1.5 on an 800x600 canvas gives ~3px effective trail width.
    blurred = gaussian_filter(trail_map, sigma=1.5)

    # Step 2: filter(v > 0) — only color pixels with pheromone
    mask = blurred > 0

    if not np.any(mask):
        return np.tile(palette[0], (H, W, 1))

    # Step 3: log(v) — compress dynamic range
    log_vals = np.log(blurred[mask])

    # Step 4: scale_fill_gradientn — map FULL range linearly to [0, 255]
    v_min = log_vals.min()
    v_max = log_vals.max()

    # Start with background
    rgb = np.tile(palette[0], (H, W, 1))

    if v_max > v_min:
        indices = ((log_vals - v_min) / (v_max - v_min) * 255.0).astype(np.uint8)
    else:
        indices = np.zeros_like(log_vals, dtype=np.uint8)

    rgb[mask] = palette[indices]

    return rgb
