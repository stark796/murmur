"""
Fluid advection for murmur.

Applies a velocity field to the Physarum trail map using semi-Lagrangian
advection — making trails drift and flow rather than being static.
This gives the simulation the "amoeba in water" quality.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates


def advect_trail_map(
    trail_map: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    strength: float,
) -> np.ndarray:
    """
    Advect a 2D scalar field (trail_map) through a velocity field (vx, vy).

    Uses semi-Lagrangian advection: for each output pixel, we look back
    along the velocity to find where it came from, and sample there.

    Parameters
    ----------
    trail_map : (H, W) float32 — Physarum pheromone concentration
    vx        : (H, W) float32 — x-component of velocity (normalized [-1, 1])
    vy        : (H, W) float32 — y-component of velocity (normalized [-1, 1])
    strength  : float — how strongly to apply the advection (pixels per frame)

    Returns
    -------
    Advected trail map, same shape as input.
    """
    H, W = trail_map.shape

    # Pixel coordinate grids
    row_coords, col_coords = np.mgrid[0:H, 0:W].astype(np.float32)

    # Back-trace: where did each pixel come from?
    # We subtract velocity * strength to look backward
    src_rows = row_coords - vy * strength
    src_cols = col_coords - vx * strength

    # Wrap coordinates (toroidal boundary)
    src_rows = src_rows % H
    src_cols = src_cols % W

    # Sample the trail map at source coordinates using bicubic interpolation
    coords = np.array([src_rows.ravel(), src_cols.ravel()])
    advected = (
        map_coordinates(
            trail_map,
            coords,
            order=3,
            mode="wrap",
            prefilter=True,
        )
        .reshape(H, W)
        .astype(np.float32)
    )

    return advected
