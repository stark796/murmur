"""
Perlin noise field generator for murmur.

Generates time-evolving 2D noise fields used to:
  1. Perturb individual agent sensor angles (stochastic variation)
  2. Build the advection velocity field (drift/flow)
"""

from __future__ import annotations

import numpy as np

try:
    from noise import pnoise3

    _HAS_NOISE = True
except ImportError:
    _HAS_NOISE = False


def _fallback_noise(H: int, W: int, t: float, freq: float, seed: int) -> np.ndarray:
    """
    Simple fallback if the 'noise' library is not available.
    Uses NumPy's random with a time-varying seed to approximate coherent noise.
    """
    rng = np.random.default_rng(seed=int(t * 100) % 2**31)
    base = rng.standard_normal((H, W)).astype(np.float32)
    from scipy.ndimage import gaussian_filter

    sigma = max(1, int(min(H, W) / (freq * 4)))
    return gaussian_filter(base, sigma=sigma)


def get_noise_field(
    H: int,
    W: int,
    t: float,
    frequency: float = 3.0,
    octaves: int = 4,
    seed: int = 42,
) -> np.ndarray:
    """
    Return a (H, W) float32 array of Perlin noise in [-1, 1].

    Parameters
    ----------
    H, W     : grid dimensions
    t        : time coordinate — drives evolution of the field
    frequency: spatial frequency (higher = more variation / smaller features)
    octaves  : number of noise octaves (more = more detail)
    seed     : offset for the noise space (acts like a random seed)
    """
    if not _HAS_NOISE:
        return _fallback_noise(H, W, t, frequency, seed)

    scale_x = frequency / W
    scale_y = frequency / H
    t_scaled = t * 0.15  # slow temporal evolution

    # Pre-compute coordinate grids
    xs = np.arange(W, dtype=np.float64) * scale_x + seed * 100
    ys = np.arange(H, dtype=np.float64) * scale_y + seed * 100

    field = np.empty((H, W), dtype=np.float32)
    for row_idx, y in enumerate(ys):
        for col_idx, x in enumerate(xs):
            field[row_idx, col_idx] = pnoise3(
                x,
                y,
                t_scaled,
                octaves=octaves,
                persistence=0.5,
                lacunarity=2.0,
                repeatx=1024,
                repeaty=1024,
                repeatz=1024,
                base=seed,
            )
    return field


def get_noise_field_fast(
    H: int,
    W: int,
    t: float,
    frequency: float = 3.0,
    octaves: int = 3,
    seed: int = 42,
) -> np.ndarray:
    """
    Faster version using vectorized approach with scipy for the fallback,
    or a tiled Perlin approach for speed.

    For large grids (512+) the full pnoise3 loop is slow.
    This version uses a coarser grid and upsamples.
    """
    if not _HAS_NOISE:
        return _fallback_noise(H, W, t, frequency, seed)

    from scipy.ndimage import zoom

    # Sample at 1/4 resolution then upscale
    factor = 4
    h_s, w_s = max(1, H // factor), max(1, W // factor)
    small = get_noise_field(h_s, w_s, t, frequency, octaves, seed)

    # Upscale with bicubic interpolation
    zoom_h = H / h_s
    zoom_w = W / w_s
    return zoom(small, (zoom_h, zoom_w), order=3).astype(np.float32)


def get_velocity_field(
    H: int,
    W: int,
    t: float,
    frequency: float = 2.5,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a divergence-free-ish 2D velocity field (vx, vy) from two
    offset Perlin noise fields. Both arrays are shape (H, W), float32.

    Using two separate noise fields (offset by seed) gives us x and y
    velocity components that vary smoothly in space and time.
    """
    vx = get_noise_field_fast(H, W, t, frequency=frequency, octaves=3, seed=seed)
    vy = get_noise_field_fast(H, W, t, frequency=frequency, octaves=3, seed=seed + 7919)
    return vx, vy
