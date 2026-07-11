"""
murmur v2 — Radial symmetry enforcement.

After every N simulation steps, average the trail map with its
n-fold rotated copies. This creates mandala-like radial patterns
without forcing exact symmetry (softer blend = more organic).

blend=0.0 → no symmetry enforcement (pure agent behavior)
blend=1.0 → hard n-fold symmetry enforced
"""

from __future__ import annotations

import math
import numpy as np
from scipy.ndimage import rotate as scipy_rotate


def enforce_symmetry(
    trail_map: np.ndarray,
    n_fold: int,
    blend: float,
) -> np.ndarray:
    """
    Average the trail map with its n-fold rotationally symmetric copies.

    Parameters
    ----------
    trail_map : (H, W) float32
    n_fold    : number of symmetry folds (2–12)
    blend     : 0=keep original, 1=fully averaged (mandala strength)

    Returns
    -------
    (H, W) float32 — symmetrized trail map
    """
    if n_fold < 2 or blend <= 0.0:
        return trail_map

    # For n=4, use fast numpy rotations (exact 90° multiples)
    if n_fold == 4:
        accumulated = (
            trail_map
            + np.rot90(trail_map, 1)
            + np.rot90(trail_map, 2)
            + np.rot90(trail_map, 3)
        ) / 4.0
    elif n_fold == 2:
        accumulated = (trail_map + np.rot90(trail_map, 2)) / 2.0
    else:
        # General case: scipy rotate for arbitrary angles
        accumulated = np.zeros_like(trail_map)
        angle_step = 360.0 / n_fold
        for k in range(n_fold):
            angle = k * angle_step
            if angle == 0.0:
                accumulated += trail_map
            else:
                rotated = scipy_rotate(
                    trail_map,
                    angle,
                    reshape=False,
                    order=1,
                    mode="wrap",
                    prefilter=False,
                )
                accumulated += rotated
        accumulated /= n_fold

    result = trail_map * (1.0 - blend) + accumulated * blend
    return result.astype(np.float32)
