"""
murmur v2 — MandalaEngine.

EXACT Fronkonstin replication. Hardcoded to his literal R/C++ values.
No DNA scaling, no species, no attractors.
"""

from __future__ import annotations

import math
import numpy as np
from types import SimpleNamespace

from murmur.audio.features import SongDNA
from murmur.v2.init_geometry import build_init
from murmur.v2.symmetry import enforce_symmetry

from murmur.simulation.physarum_glitch import _agent_step_glitch_numba


class MandalaEngine:
    """
    Grows a Fronkonstin-style Physarum abstraction using the exact
    memory mapping glitch that gives it its signature thick, blobby look.
    """

    def __init__(self, dna: SongDNA, width: int, height: int) -> None:
        self.dna = dna
        self.W = width
        self.H = height

        trail_map, ax, ay, angles = build_init(dna, width, height)
        # Convert trail_map (2D) to envM (1D glitch mapping)
        self.envM = np.zeros(width * height, dtype=np.float32)
        for i in range(height):
            for j in range(width):
                self.envM[i + j * height] = trail_map[i, j]

        self.ax = ax
        self.ay = ay
        self.angles = angles
        self.rng = np.random.default_rng(dna.seed)

        self.p = self._params(len(ax))

        print(f"  Agents: {len(self.ax):,}")
        print(f"  Steps: {self.p.n_steps}")
        print(f"  Canvas: {self.W}×{self.H}")
        print(
            f"  SA: {math.degrees(self.p.sensor_angle[0]):.1f}°  "
            f"RA: {math.degrees(self.p.rotation_angle[0]):.1f}°  "
            f"SO: {self.p.sensor_distance[0]:.0f}px  "
            f"SS: {self.p.step_size[0]:.0f}px  "
            f"depT: {self.p.deposit_amount[0]:.0f}  "
            f"decay: {self.p.decay_factor:.2f}"
        )

    def _params(self, n_agents: int) -> SimpleNamespace:
        """Fronkonstin's exact R script parameters. No variation."""
        sa = np.full(n_agents, 22.5 * math.pi / 180.0, dtype=np.float32)
        ra = np.full(n_agents, 45.0 * math.pi / 180.0, dtype=np.float32)
        sd = np.full(n_agents, 6.0, dtype=np.float32)
        ss = np.full(n_agents, 1.0, dtype=np.float32)
        da = np.full(n_agents, 15.0, dtype=np.float32)

        return SimpleNamespace(
            sensor_angle=sa,
            rotation_angle=ra,
            sensor_distance=sd,
            step_size=ss,
            deposit_amount=da,
            decay_factor=0.90,  # 1 - decayT where decayT = 0.1
            n_steps=2000,  # Exact Fronkonstin
        )

    def run(self, progress_cb=None) -> np.ndarray:
        """Run the exact Fronkonstin buggy simulation loop."""
        p = self.p

        for step in range(p.n_steps):
            random_vals = self.rng.random(len(self.ax)).astype(np.float32)
            _agent_step_glitch_numba(
                self.ax,
                self.ay,
                self.angles,
                self.envM,
                p.sensor_angle,
                p.sensor_distance,
                p.rotation_angle,
                p.step_size,
                p.deposit_amount,
                self.H,
                self.W,
                random_vals,
            )

            # Decay
            self.envM *= p.decay_factor

            if progress_cb is not None:
                progress_cb(step, p.n_steps)

        # Reconstruct the 2D trail map from the corrupted 1D space
        # R's melt(envM) with envM having dimensions HxW
        # maps envM(i, j) to x=i, y=j. Then he plotted with x=x, y=y.
        # But we'll reconstruct the visual representation as a WxH (horizontal x vertical) grid.
        trail_map = np.zeros((self.H, self.W), dtype=np.float32)
        for i in range(self.H):
            for j in range(self.W):
                trail_map[i, j] = self.envM[i + j * self.H]

        return trail_map
