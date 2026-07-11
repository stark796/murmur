"""
Post-processing visual effects for murmur.

Applied per-frame after the trail map is converted to RGB:
  - Bloom / glow: soft light bleed on bright areas
  - Vignette: darkened edges for a "painting" feel
  - Film grain: subtle texture
  - Color grading: final mood adjustments
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter


# ---------------------------------------------------------------------------
# Pre-compute the vignette mask once per resolution
# ---------------------------------------------------------------------------
_vignette_cache: dict[tuple, np.ndarray] = {}


def _get_vignette(H: int, W: int, strength: float = 0.5) -> np.ndarray:
    """Return a (H, W) float32 vignette mask in [0, 1] — darker at edges."""
    key = (H, W, round(strength, 2))
    if key not in _vignette_cache:
        cy, cx = H / 2, W / 2
        y = np.linspace(-1, 1, H)
        x = np.linspace(-1, 1, W)
        xx, yy = np.meshgrid(x, y)
        dist = np.sqrt(xx**2 + yy**2)
        # Smooth cosine vignette
        mask = np.clip(1.0 - dist * strength, 0.0, 1.0).astype(np.float32)
        mask = mask**2  # strengthen the falloff
        _vignette_cache[key] = mask
    return _vignette_cache[key]


def apply_bloom(
    rgb: np.ndarray,
    strength: float = 0.25,
    sigma: float = 4.0,
    threshold: float = 0.6,
) -> np.ndarray:
    """
    Additive bloom: blur the bright areas and blend back in.

    Parameters
    ----------
    rgb       : (H, W, 3) float32 in [0, 1]
    strength  : how strongly to blend the bloom
    sigma     : Gaussian blur radius for the bloom glow
    threshold : only pixels brighter than this contribute to bloom
    """
    # Luminance mask: only bright areas bloom
    lum = rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114
    bright_mask = np.clip((lum - threshold) / (1.0 - threshold + 1e-6), 0.0, 1.0)

    # Isolate bright regions
    bright = rgb * bright_mask[..., np.newaxis]

    # Blur them
    bloom = np.stack(
        [gaussian_filter(bright[..., c], sigma=sigma) for c in range(3)], axis=-1
    ).astype(np.float32)

    # Additive blend
    result = rgb + bloom * strength
    return np.clip(result, 0.0, 1.0)


def apply_vignette(
    rgb: np.ndarray,
    strength: float = 0.55,
) -> np.ndarray:
    """
    Darken the edges of the image.

    Parameters
    ----------
    rgb      : (H, W, 3) float32 in [0, 1]
    strength : 0 = no vignette, 1 = heavy darkening at edges
    """
    H, W = rgb.shape[:2]
    mask = _get_vignette(H, W, strength)[..., np.newaxis]  # (H, W, 1)
    return (rgb * mask).astype(np.float32)


def apply_grain(
    rgb: np.ndarray,
    amount: float = 0.025,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """
    Add subtle film grain (luminance noise).

    Parameters
    ----------
    rgb    : (H, W, 3) float32 in [0, 1]
    amount : standard deviation of the grain noise
    rng    : optional numpy Generator for reproducibility
    """
    if rng is None:
        rng = np.random.default_rng()
    noise = rng.standard_normal(rgb.shape[:2]).astype(np.float32) * amount
    grained = rgb + noise[..., np.newaxis]
    return np.clip(grained, 0.0, 1.0)


def apply_color_grade(
    rgb: np.ndarray,
    saturation_boost: float = 1.15,
    shadow_lift: float = 0.01,
) -> np.ndarray:
    """
    Final color grading pass.

    Parameters
    ----------
    rgb              : (H, W, 3) float32 in [0, 1]
    saturation_boost : > 1 = more saturated, < 1 = desaturated
    shadow_lift      : slight lift to pure black (prevents crushed blacks)
    """
    # Lift shadows slightly
    rgb = rgb + shadow_lift * (1.0 - rgb)

    # Saturation boost: mix between grayscale and color
    lum = rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114
    lum3 = lum[..., np.newaxis]
    rgb = lum3 + (rgb - lum3) * saturation_boost

    return np.clip(rgb, 0.0, 1.0)


def apply_all_effects(
    rgb_uint8: np.ndarray,
    vibe: float,
    rms: float,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """
    Apply all post-processing effects in sequence and return uint8.

    Parameters
    ----------
    rgb_uint8 : (H, W, 3) uint8
    vibe      : 0–1 — controls effect intensity
    rms       : normalized audio loudness — boosts bloom on loud moments
    rng       : optional random generator for grain

    Returns
    -------
    (H, W, 3) uint8
    """
    # Convert to float for processing
    rgb = rgb_uint8.astype(np.float32) / 255.0

    # Bloom: stronger on louder moments and higher vibe
    bloom_strength = 0.15 + rms * 0.25 + vibe * 0.1
    rgb = apply_bloom(rgb, strength=bloom_strength, sigma=3.5 + vibe * 3.0)

    # Vignette: turn off for clean glassy look
    rgb = apply_vignette(rgb, strength=0.0)

    # Film grain: turn off for clean glassy look
    grain_amount = 0.0
    rgb = apply_grain(rgb, amount=grain_amount, rng=rng)

    # Color grade
    rgb = apply_color_grade(
        rgb,
        saturation_boost=1.1 + vibe * 0.1,
        shadow_lift=0.008,
    )

    # Back to uint8
    return (rgb * 255).clip(0, 255).astype(np.uint8)
