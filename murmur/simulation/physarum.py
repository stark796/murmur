"""
Physarum slime mold simulation for murmur.

Implements the Jeff Jones (2010) multi-agent model:
  "Characteristics of pattern formation and evolution in approximations
   of Physarum transport networks."

Each agent senses pheromone at three points ahead (left, center, right),
steers toward the strongest signal, moves forward, and deposits trail.
The trail diffuses and decays each step, creating organic growing networks.

Performance: Numba JIT compiles the agent loop to native code, enabling
200K+ agents at interactive speeds on CPU.
"""

from __future__ import annotations

import math
import numpy as np
from scipy.ndimage import gaussian_filter

from murmur.core.config import SimParams
from murmur.simulation.advection import advect_trail_map
from murmur.simulation.noise import get_velocity_field

# ---------------------------------------------------------------------------
# Attempt to import Numba; fall back to pure NumPy if unavailable
# ---------------------------------------------------------------------------
try:
    from numba import njit, prange

    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False
    print("[murmur] Warning: numba not found, falling back to pure NumPy (slower)")


if _HAS_NUMBA:

    @njit(parallel=True, fastmath=True)
    def _agent_step_numba(
        ax: np.ndarray,  # (N,) float32 — agent x positions
        ay: np.ndarray,  # (N,) float32 — agent y positions
        angles: np.ndarray,  # (N,) float32 — agent headings (radians)
        trail: np.ndarray,  # (H, W) float32 — pheromone grid
        noise_field: np.ndarray,  # (H, W) float32 — per-pixel angle noise
        sensor_angle: np.ndarray,
        sensor_distance: np.ndarray,
        rotation_angle: np.ndarray,
        step_size: np.ndarray,
        deposit_amount: np.ndarray,
        noise_amount: float,
        H: int,
        W: int,
        random_vals: np.ndarray,  # (N,) float32 — pre-generated random [0,1]
    ) -> None:
        """In-place update of agent positions, angles, and trail map."""
        for i in prange(len(ax)):
            angle = angles[i]

            # Apply perlin noise perturbation to angle
            px = int(ax[i]) % W
            py = int(ay[i]) % H
            angle += noise_field[py, px] * noise_amount

            # ---- SENSE ----
            # Sample trail at left, center, right sensor positions
            sa = sensor_angle[i]
            sd = sensor_distance[i]

            lx = ax[i] + sd * math.cos(angle + sa)
            ly = ay[i] + sd * math.sin(angle + sa)
            cx = ax[i] + sd * math.cos(angle)
            cy = ay[i] + sd * math.sin(angle)
            rx = ax[i] + sd * math.cos(angle - sa)
            ry = ay[i] + sd * math.sin(angle - sa)

            # Wrap sensor positions
            l_val = trail[int(ly) % H, int(lx) % W]
            c_val = trail[int(cy) % H, int(cx) % W]
            r_val = trail[int(ry) % H, int(rx) % W]

            # ---- STEER ----
            # Match Jeff Jones (2010) and Fronkonstin abstractions logic exactly
            ra = rotation_angle[i]
            if c_val > l_val and c_val > r_val:
                pass  # Center is strongest -> go straight
            elif c_val < l_val and c_val < r_val:
                # Both sides stronger than center -> turn randomly
                if random_vals[i] < 0.5:
                    angle -= ra
                else:
                    angle += ra
            elif l_val > r_val:
                angle += ra  # Left is stronger -> turn left (+ra)
            elif r_val > l_val:
                angle -= ra  # Right is stronger -> turn right (-ra)
            else:
                # They are equal (e.g. all 0). Keep going straight!
                pass

            angles[i] = angle

            # ---- MOVE ----
            ss = step_size[i]
            ax[i] = (ax[i] + ss * math.cos(angle)) % W
            ay[i] = (ay[i] + ss * math.sin(angle)) % H

            # ---- DEPOSIT ----
            trail[int(ay[i]) % H, int(ax[i]) % W] += deposit_amount[i]

else:

    def _agent_step_numba(
        ax,
        ay,
        angles,
        trail,
        noise_field,
        sensor_angle,
        sensor_distance,
        rotation_angle,
        step_size,
        deposit_amount,
        noise_amount,
        H,
        W,
        random_vals,
    ):
        """Pure-NumPy vectorized fallback (no Numba)."""
        N = len(ax)

        # Apply noise perturbation
        px = ax.astype(np.int32) % W
        py = ay.astype(np.int32) % H
        angles = angles + noise_field[py, px] * noise_amount

        # ---- SENSE (vectorized) ----
        sa, sd = sensor_angle, sensor_distance

        lx = (ax + sd * np.cos(angles + sa)).astype(np.int32) % W
        ly = (ay + sd * np.sin(angles + sa)).astype(np.int32) % H
        cx = (ax + sd * np.cos(angles)).astype(np.int32) % W
        cy = (ay + sd * np.sin(angles)).astype(np.int32) % H
        rx = (ax + sd * np.cos(angles - sa)).astype(np.int32) % W
        ry = (ay + sd * np.sin(angles - sa)).astype(np.int32) % H

        l_val = trail[ly, lx]
        c_val = trail[cy, cx]
        r_val = trail[ry, rx]

        # ---- STEER (vectorized) ----
        ra = rotation_angle
        # Masks
        go_left = (l_val > r_val) & ~(c_val > l_val)
        go_right = (r_val > l_val) & ~(c_val > r_val)
        go_rand = (l_val == r_val) & (c_val <= l_val)

        rand_turn = (random_vals < 0.5) * 2 - 1  # -1 or +1

        angles = angles.copy()
        angles[go_left] -= ra
        angles[go_right] += ra
        angles[go_rand] += ra * rand_turn[go_rand]

        # ---- MOVE ----
        ax[:] = (ax + step_size * np.cos(angles)) % W
        ay[:] = (ay + step_size * np.sin(angles)) % H

        # ---- DEPOSIT (scatter) ----
        xi = ax.astype(np.int32) % W
        yi = ay.astype(np.int32) % H
        np.add.at(trail, (yi, xi), deposit_amount)


# ---------------------------------------------------------------------------
# Physarum simulation class
# ---------------------------------------------------------------------------


class PhysarumSimulation:
    """
    Manages the Physarum simulation state and step logic.

    State:
        trail_map : (H, W) float32 — pheromone concentrations
        ax, ay    : (N,) float32   — agent positions
        angles    : (N,) float32   — agent headings
    """

    def __init__(
        self,
        width: int,
        height: int,
        n_agents: int = 100_000,
        seed: int = 42,
        initial_map: np.ndarray = None,
        ax: np.ndarray = None,
        ay: np.ndarray = None,
        angles: np.ndarray = None,
    ) -> None:
        self.W = width
        self.H = height
        self.N = n_agents
        self.rng = np.random.default_rng(seed)
        self._t = 0.0  # time counter for noise evolution

        # Initialize trail map
        if initial_map is not None:
            self.trail_map = initial_map.astype(np.float32).copy()
        else:
            self.trail_map = self.rng.random((height, width), dtype=np.float32) * 0.005

        # Initialize agents
        if ax is not None and ay is not None and angles is not None:
            self.ax = ax.astype(np.float32).copy()
            self.ay = ay.astype(np.float32).copy()
            self.angles = angles.astype(np.float32).copy()
            # If using custom structured initialization, skip pre-warming
            # to preserve the pristine structure.
        else:
            self.ax = self.rng.uniform(0, width, n_agents).astype(np.float32)
            self.ay = self.rng.uniform(0, height, n_agents).astype(np.float32)
            self.angles = self.rng.uniform(0, 2 * math.pi, n_agents).astype(np.float32)
            # Pre-warm the simulation to establish initial trail patterns
            self._prewarm()

    def _prewarm(self, steps: int = 15) -> None:
        """Run a few simulation steps to seed trails."""

        class DummyParams:
            sensor_distance = 15.0
            sensor_angle = 0.5
            rotation_angle = 0.5
            step_size = 2.0
            deposit_amount = 0.1
            decay_factor = 0.95
            diffusion_sigma = 1.0
            advection_strength = 0.0
            noise_amount = 0.0

        default_params = DummyParams()
        noise_field = np.zeros((self.H, self.W), dtype=np.float32)
        for _ in range(steps):
            self._step_core(default_params, noise_field)

    def step(self, params: SimParams) -> None:
        """
        Advance the simulation by one frame using the given audio-driven params.
        Also applies fluid advection and diffuse/decay.
        """
        self._t += 0.1

        # Get noise field for this timestep
        noise_field = _get_noise_field_cached(self.H, self.W, self._t)

        # Core agent step
        self._step_core(params, noise_field)

        # Fluid advection on the trail map (makes it flow)
        if params.advection_strength > 0.01:
            vx, vy = get_velocity_field(self.H, self.W, self._t)
            self.trail_map = advect_trail_map(
                self.trail_map, vx, vy, params.advection_strength
            )

    def _step_core(self, params: SimParams, noise_field: np.ndarray) -> None:
        """Inner step: sense→steer→move→deposit→diffuse→decay."""
        random_vals = self.rng.random(self.N).astype(np.float32)

        _agent_step_numba(
            self.ax,
            self.ay,
            self.angles,
            self.trail_map,
            noise_field,
            params.sensor_angle,
            params.sensor_distance,
            params.rotation_angle,
            params.step_size,
            params.deposit_amount,
            params.noise_amount,
            self.H,
            self.W,
            random_vals,
        )

        # Diffuse (Gaussian blur) — trails spread outward
        if params.diffusion_sigma > 0.0:
            self.trail_map = gaussian_filter(
                self.trail_map,
                sigma=params.diffusion_sigma,
                mode="wrap",
            )

        # Decay — old trails fade
        self.trail_map *= params.decay_factor

        # Clamp
        np.clip(self.trail_map, 0.0, 1.0, out=self.trail_map)

    @property
    def trail(self) -> np.ndarray:
        """Return the current trail map (H, W) float32 in [0, 1]."""
        return self.trail_map


# ---------------------------------------------------------------------------
# Noise field cache (avoid recomputing for same timestep)
# ---------------------------------------------------------------------------
_noise_cache: dict[float, np.ndarray] = {}
_noise_cache_t: float = -9999.0
_noise_cache_field: np.ndarray | None = None


def _get_noise_field_cached(H: int, W: int, t: float) -> np.ndarray:
    """Return a noise field, reusing the last one if t is close enough."""
    global _noise_cache_t, _noise_cache_field
    # Round t to reduce cache misses (update noise every ~5 frames)
    t_key = round(t * 2) / 2.0
    if t_key != _noise_cache_t or _noise_cache_field is None:
        from murmur.simulation.noise import get_noise_field_fast

        _noise_cache_field = get_noise_field_fast(H, W, t_key, frequency=4.0, octaves=3)
        _noise_cache_t = t_key
    return _noise_cache_field
