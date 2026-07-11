"""
Physarum slime mold simulation with the exact Fronkonstin memory mapping glitch.
"""

from __future__ import annotations

import math
import numpy as np

try:
    from numba import njit, prange

    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False


@njit
def get_index(M, val):
    val = int(round(val))
    if val < 0:
        return (M + val % M) % M
    if val >= M:
        return val % M
    return val


if _HAS_NUMBA:

    @njit(parallel=True, fastmath=True)
    def _agent_step_glitch_numba(
        ax: np.ndarray,  # (N,) float32 — agent x positions
        ay: np.ndarray,  # (N,) float32 — agent y positions
        angles: np.ndarray,  # (N,) float32 — agent headings (radians)
        envM: np.ndarray,  # (W*H,) float32 — 1D flat pheromone array for glitch
        sensor_angle: np.ndarray,
        sensor_distance: np.ndarray,
        rotation_angle: np.ndarray,
        step_size: np.ndarray,
        deposit_amount: np.ndarray,
        H: int,
        W: int,
        random_vals: np.ndarray,  # (N,) float32 — pre-generated random [0,1]
    ) -> None:
        """In-place update of agent positions, angles, and trail map WITH BUG."""
        memory_size = W * H

        # Sense and Steer
        for i in prange(len(ax)):
            sa = sensor_angle[i]
            sd = sensor_distance[i]
            ra = rotation_angle[i]
            angle = angles[i]

            # The exact Fronkonstin bug: Fx uses get_index with m (H) but adds to x
            # Fy uses get_index with n (W) but adds to y
            Fx = get_index(H, ax[i] + sd * math.cos(angle))
            Fy = get_index(W, ay[i] + sd * math.sin(angle))

            FLx = get_index(H, ax[i] + sd * math.cos(angle + sa))
            FLy = get_index(W, ay[i] + sd * math.sin(angle + sa))

            FRx = get_index(H, ax[i] + sd * math.cos(angle - sa))
            FRy = get_index(W, ay[i] + sd * math.sin(angle - sa))

            # The index into envM is computed as Fx + Fy * H
            # because he reads envM(Fx, Fy) from an HxW matrix in Armadillo
            F = envM[Fx + Fy * H]
            FL = envM[FLx + FLy * H]
            FR = envM[FRx + FRy * H]

            if F > FL and F > FR:
                pass
            elif F < FL and F < FR:
                if random_vals[i] < 0.5:
                    angle += ra
                else:
                    angle -= ra
            elif FL < FR:
                angle -= ra
            elif FR < FL:
                angle += ra

            angles[i] = angle

        # Move
        for i in prange(len(ax)):
            ss = step_size[i]
            ax[i] = get_index(W, ax[i] + ss * math.cos(angles[i]))
            ay[i] = get_index(H, ay[i] + ss * math.sin(angles[i]))

        # Deposit
        # (NOT in parallel to avoid race conditions, exactly like our normal engine)
        for i in range(len(ax)):
            idx = int(ax[i]) + int(ay[i]) * H
            if idx < memory_size:
                envM[idx] += deposit_amount[i]
