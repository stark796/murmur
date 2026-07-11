import numpy as np
from noise import pnoise2
from murmur.audio.features import GlobalAudioFeatures
from murmur.rendering.palette import build_colormap
from murmur.rendering.palette import hex_to_hsl, hsl_to_rgb
from murmur.rendering.effects import apply_all_effects
from scipy.ndimage import gaussian_filter


def _song_seed(features: GlobalAudioFeatures) -> int:
    return int(
        (features.bpm * 1000)
        + features.mean_rms * 100000
        + features.mean_centroid * 1000000
        + features.dominant_chroma * 7919
    ) % (2**32 - 1)


def _sample_particle_layer(
    density: np.ndarray,
    n_particles: int,
    rng: np.random.Generator,
    jitter: float,
    weight_scale: float,
) -> np.ndarray:
    """Sample many tiny points from a density map and accumulate intensity."""
    H, W = density.shape
    weights = np.clip(density, 0.0, None).astype(np.float64)
    total = weights.sum()
    if total <= 0:
        return np.zeros((H, W), dtype=np.float32)

    probability = (weights / total).ravel()
    selected = rng.choice(H * W, size=n_particles, replace=True, p=probability)
    ys, xs = np.divmod(selected, W)
    xs = np.clip(np.rint(xs + rng.normal(0.0, jitter, n_particles)), 0, W - 1).astype(
        np.int32
    )
    ys = np.clip(np.rint(ys + rng.normal(0.0, jitter, n_particles)), 0, H - 1).astype(
        np.int32
    )

    values = rng.gamma(shape=1.6, scale=weight_scale, size=n_particles).astype(
        np.float32
    )
    layer = np.zeros((H, W), dtype=np.float32)
    np.add.at(layer, (ys, xs), values)

    # A small fraction of particles become visible neighboring sparks instead of blur.
    spark_count = max(0, n_particles // 8)
    if spark_count:
        spark_idx = rng.choice(n_particles, size=spark_count, replace=False)
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            sx = np.clip(xs[spark_idx] + dx, 0, W - 1)
            sy = np.clip(ys[spark_idx] + dy, 0, H - 1)
            np.add.at(layer, (sy, sx), values[spark_idx] * 0.24)

    p = np.percentile(layer[layer > 0], 99.5) if np.any(layer > 0) else 1.0
    return np.clip(layer / (p + 1e-8), 0.0, 1.0)


def generate_static_image(
    features: GlobalAudioFeatures,
    base_hex: str,
    vibe: float,
    width: int = 1920,
    height: int = 1080,
    stochastic_map: np.ndarray = None,
) -> np.ndarray:
    """
    Generate a high-res abstract painting (domain warped perlin noise)
    driven by global audio features.

    Returns:
        (H, W, 3) uint8 RGB array
    """
    print("  🎨 Generating domain-warped noise field...")

    # 1. Base parameters driven by audio
    # BPM scales the spatial frequency (how many shapes fit on the canvas)
    base_bpm = max(features.bpm, 60.0)
    freq = (base_bpm / 120.0) * 2.0 + vibe * 2.0

    # RMS (loudness) scales the warping amplitude (how twisted the shapes get)
    warp_amp = features.mean_rms * 4.0 + vibe * 2.0

    # Centroid (brightness) scales the grit/detail (octaves)
    octaves = int(3 + features.mean_centroid * 6)

    # 2. Build coordinate grid
    y = np.linspace(0, freq, height)
    x = np.linspace(0, freq * (width / height), width)
    yy, xx = np.meshgrid(y, x, indexing="ij")

    # 3. Domain warping
    # Generate two noise fields to offset the coordinates
    wx = np.zeros_like(xx)
    wy = np.zeros_like(yy)

    # We use a fast python loop, this takes ~2-3 seconds for 1080p
    for i in range(height):
        for j in range(width):
            wx[i, j] = pnoise2(yy[i, j] + 12.34, xx[i, j] + 56.78, octaves=2)
            wy[i, j] = pnoise2(yy[i, j] + 98.76, xx[i, j] + 54.32, octaves=2)

    ww_x = xx + wx * warp_amp
    ww_y = yy + wy * warp_amp

    # 4. Final Noise Field (serves as the density/trail map)
    field = np.zeros_like(xx)
    persistence = 0.4 + features.mean_bandwidth * 0.4

    # Apply stochastic displacement if provided
    if stochastic_map is not None:
        # Normalize map
        s_max = stochastic_map.max()
        if s_max > 0:
            stochastic_map = stochastic_map / s_max
        # Add the slime mold map directly to the warp fields!
        ww_x += stochastic_map * (warp_amp * 2.0)
        ww_y += stochastic_map * (warp_amp * 2.0)

    for i in range(height):
        for j in range(width):
            field[i, j] = pnoise2(
                ww_y[i, j], ww_x[i, j], octaves=octaves, persistence=persistence
            )

    # Normalize to [0, 1]
    mn, mx = field.min(), field.max()
    field = (field - mn) / (mx - mn + 1e-8)

    # Smooth the contrast (S-curve)
    field = field * field * (3 - 2 * field)

    # 5. Apply Colormap
    print("  🎨 Applying colormap and post-processing...")
    hue_shift = (features.dominant_chroma / 12.0) * 0.2
    cmap = build_colormap(base_hex, hue_shift=hue_shift)

    indices = (field * 255).astype(np.int32)
    rgb = cmap[indices]

    # 6. Apply standard post-processing (bloom, vignette, etc.)
    # We pass the mean_rms so the bloom scales with loudness
    rgb_final = apply_all_effects(rgb, vibe=vibe, rms=features.mean_rms)

    return rgb_final


def generate_vein_image(
    features: GlobalAudioFeatures,
    base_hex: str,
    vibe: float,
    vein_map: np.ndarray,
) -> np.ndarray:
    """
    Render a rigid vascular map as abstract song art.

    The geometry comes from the song fingerprint; the color comes from the same
    global audio features, with hard cores and a restrained glow.
    """
    print("  Rendering rigid vascular structure...")

    H, W = vein_map.shape
    h, s, l = hex_to_hsl(base_hex)

    energy = np.clip(features.mean_rms, 0.0, 1.0)
    brightness = np.clip(features.mean_centroid, 0.0, 1.0)
    texture = np.clip(features.mean_bandwidth, 0.0, 1.0)

    bg = np.array(hsl_to_rgb(h, s * 0.35, 0.025 + l * 0.05), dtype=np.float32) / 255.0
    body = (
        np.array(
            hsl_to_rgb(h, min(1.0, s * (0.95 + vibe * 0.25)), 0.32 + energy * 0.22),
            dtype=np.float32,
        )
        / 255.0
    )
    highlight_h = (h + 0.08 + features.dominant_chroma / 48.0) % 1.0
    highlight = (
        np.array(
            hsl_to_rgb(highlight_h, min(1.0, s * 0.75 + 0.2), 0.72 + brightness * 0.18),
            dtype=np.float32,
        )
        / 255.0
    )

    core = np.clip(vein_map, 0.0, 1.0)
    broad_field = gaussian_filter(core, sigma=7.0 + texture * 8.0 + vibe * 5.0)
    near_field = np.maximum(core, gaussian_filter(core, sigma=1.3))

    rng = np.random.default_rng(_song_seed(features))
    area = H * W
    n_core = int(area * (0.16 + energy * 0.08 + texture * 0.07 + vibe * 0.04))
    n_dust = int(area * (0.035 + texture * 0.035 + vibe * 0.02))
    n_core = int(np.clip(n_core, 45000, 900000))
    n_dust = int(np.clip(n_dust, 10000, 260000))

    print(f"  Seeding {n_core + n_dust:,} song particles...")

    dense_particles = _sample_particle_layer(
        density=np.power(near_field, 1.35),
        n_particles=n_core,
        rng=rng,
        jitter=0.45 + texture * 0.75,
        weight_scale=0.22,
    )
    dust_particles = _sample_particle_layer(
        density=np.power(broad_field + 1e-6, 0.8),
        n_particles=n_dust,
        rng=rng,
        jitter=2.0 + vibe * 2.0,
        weight_scale=0.08,
    )

    rgb = np.ones((H, W, 3), dtype=np.float32) * bg
    rgb += broad_field[..., np.newaxis] * body * (0.45 + energy * 0.55)
    rgb += dust_particles[..., np.newaxis] * body * (0.4 + texture * 0.4)
    rgb += dense_particles[..., np.newaxis] * body * (0.95 + energy * 0.65)

    highlight_mask = np.maximum(np.power(core, 1.8) * 0.28, dense_particles * 1.05)
    rgb = (
        rgb * (1.0 - highlight_mask[..., np.newaxis] * 0.55)
        + highlight * highlight_mask[..., np.newaxis] * 0.85
    )

    # Add a faint mineral grain that is deterministic for a given song profile.
    grain = rng.normal(0.0, 0.012 + texture * 0.01, (H, W, 1)).astype(np.float32)
    rgb = np.clip(rgb + grain, 0.0, 1.0)

    return (rgb * 255).clip(0, 255).astype(np.uint8)


def _soft_quantile_normalize(field: np.ndarray) -> np.ndarray:
    low, high = np.percentile(field, [1.0, 99.0])
    field = (field - low) / (high - low + 1e-8)
    return np.clip(field, 0.0, 1.0)


def _audio_gradient(
    base_hex: str, features: GlobalAudioFeatures, vibe: float
) -> np.ndarray:
    h, s, l = hex_to_hsl(base_hex)
    energy = np.clip(features.mean_rms, 0.0, 1.0)
    brightness = np.clip(features.mean_centroid, 0.0, 1.0)
    texture = np.clip(features.mean_bandwidth, 0.0, 1.0)

    shadow_h = (h - 0.08 - texture * 0.08) % 1.0
    glow_h = (h + 0.11 + features.dominant_chroma / 60.0) % 1.0
    hot_h = (h + 0.22 + brightness * 0.08) % 1.0

    stops = [
        (0.00, hsl_to_rgb(shadow_h, s * 0.28, 0.015)),
        (0.22, hsl_to_rgb(shadow_h, s * 0.65, 0.07 + l * 0.10)),
        (0.48, hsl_to_rgb(h, min(1.0, s * 1.05), 0.22 + energy * 0.18)),
        (0.68, hsl_to_rgb(glow_h, min(1.0, s * 0.9 + 0.15), 0.48 + brightness * 0.18)),
        (0.84, hsl_to_rgb(hot_h, min(1.0, s * 0.45 + 0.25), 0.78 + vibe * 0.10)),
        (1.00, hsl_to_rgb(hot_h, s * 0.18, 0.96)),
    ]

    cmap = np.zeros((256, 3), dtype=np.float32)
    for i in range(256):
        t = i / 255.0
        left, right = stops[0], stops[-1]
        for j in range(len(stops) - 1):
            if stops[j][0] <= t <= stops[j + 1][0]:
                left, right = stops[j], stops[j + 1]
                break
        span = right[0] - left[0]
        a = 0.0 if span <= 0 else (t - left[0]) / span
        a = a * a * (3.0 - 2.0 * a)
        c0 = np.array(left[1], dtype=np.float32) / 255.0
        c1 = np.array(right[1], dtype=np.float32) / 255.0
        cmap[i] = c0 * (1.0 - a) + c1 * a
    return cmap


def generate_field_image(
    features: GlobalAudioFeatures,
    base_hex: str,
    vibe: float,
    width: int,
    height: int,
) -> np.ndarray:
    """
    Render a song as a warped mathematical field.

    This engine is designed for large, recognizable abstract masses: the same
    palette can produce very different silhouettes because the song controls
    the layout of ellipses, rings, folds, and contour bands.
    """
    print("  Rendering warped song field...")

    rng = np.random.default_rng(_song_seed(features))
    y = np.linspace(-1.0, 1.0, height, dtype=np.float32)
    x = np.linspace(-1.0, 1.0, width, dtype=np.float32)
    X, Y = np.meshgrid(x, y)

    bpm_norm = np.clip((features.bpm - 60.0) / 120.0, 0.0, 1.0)
    energy = np.clip(features.mean_rms, 0.0, 1.0)
    brightness = np.clip(features.mean_centroid, 0.0, 1.0)
    texture = np.clip(features.mean_bandwidth, 0.0, 1.0)
    key_angle = (features.dominant_chroma / 12.0) * np.pi * 2.0

    # Song-specific coordinate warp: broad enough to change the composition
    # from far away, not just add surface texture.
    U = X.copy()
    V = Y.copy()
    warp_layers = 4 + int(texture * 3 + vibe * 2)
    for layer in range(warp_layers):
        freq = rng.uniform(0.9, 3.7) * (1.0 + bpm_norm * 0.8) * (layer + 1) ** 0.28
        amp = rng.uniform(0.035, 0.16) * (1.0 + energy * 0.9) / (layer + 1) ** 0.45
        phase = rng.uniform(0.0, np.pi * 2.0)
        angle = key_angle + rng.normal(0.0, 1.2)
        axis = np.cos(angle) * U + np.sin(angle) * V
        cross = -np.sin(angle) * U + np.cos(angle) * V
        U = U + np.sin(axis * np.pi * freq + phase) * amp
        V = V + np.cos(cross * np.pi * (freq * 0.8) - phase) * amp

    field = np.zeros((height, width), dtype=np.float32)

    # Large elliptical masses and voids. These are what make two songs read
    # differently from across the room.
    n_masses = 7 + int(round(bpm_norm * 4 + texture * 5 + vibe * 3))
    for idx in range(n_masses):
        cx = rng.uniform(-1.05, 1.05)
        cy = rng.uniform(-1.05, 1.05)
        a = rng.uniform(0.18, 0.58) * (1.0 + energy * 0.3)
        b = rng.uniform(0.06, 0.34) * (1.0 + texture * 0.4)
        theta = key_angle + rng.normal(0.0, np.pi)
        ct, st = np.cos(theta), np.sin(theta)
        px = (U - cx) * ct + (V - cy) * st
        py = -(U - cx) * st + (V - cy) * ct
        d = (px / a) ** 2 + (py / b) ** 2
        sign = -1.0 if idx % 3 == 0 else 1.0
        amp = rng.uniform(0.5, 1.5) * sign
        field += amp * np.exp(-d * rng.uniform(0.7, 1.7))

    # Ring contours and interference bands give the image that mathematical,
    # luminous edge quality similar to the reference image.
    n_rings = 3 + int(features.dominant_chroma % 4)
    for _ in range(n_rings):
        cx = rng.uniform(-0.9, 0.9)
        cy = rng.uniform(-0.9, 0.9)
        radius = rng.uniform(0.18, 0.85)
        thickness = rng.uniform(0.025, 0.11) * (1.0 + (1.0 - brightness) * 0.5)
        r = np.sqrt((U - cx) ** 2 + (V - cy) ** 2)
        field += rng.uniform(0.45, 1.2) * np.exp(
            -((r - radius) ** 2) / (2.0 * thickness**2)
        )

    band_freq = 4.0 + bpm_norm * 9.0 + texture * 5.0
    for band in range(3):
        angle = key_angle + band * np.pi / 3.0 + rng.normal(0.0, 0.7)
        axis = np.cos(angle) * U + np.sin(angle) * V
        field += 0.22 * np.sin(axis * np.pi * band_freq + rng.uniform(0, np.pi * 2.0))
        field += 0.12 * np.sin((U * V) * np.pi * band_freq * rng.uniform(0.6, 1.5))

    field = gaussian_filter(field, sigma=1.1 + (1.0 - texture) * 1.2)
    field = _soft_quantile_normalize(field)

    contrast = 1.15 + energy * 1.4 + vibe * 0.75
    field = 1.0 / (1.0 + np.exp(-(field - 0.5) * 8.0 * contrast))

    # Contour glow makes the edges glow without making the image look like a
    # network diagram.
    gy, gx = np.gradient(field)
    edge = np.sqrt(gx * gx + gy * gy)
    edge = _soft_quantile_normalize(edge)
    field = np.clip(field + edge * (0.22 + brightness * 0.28), 0.0, 1.0)

    # Fine song-controlled turbulence for close-up detail.
    grain_field = np.zeros_like(field)
    for octave in range(3):
        freq = (8.0 + octave * 9.0) * (1.0 + texture)
        phase = rng.uniform(0.0, np.pi * 2.0)
        grain_field += np.sin((U * freq + np.cos(V * freq * 0.7 + phase)) * np.pi)
    grain_field = _soft_quantile_normalize(grain_field)
    field = np.clip(field + (grain_field - 0.5) * (0.035 + texture * 0.05), 0.0, 1.0)

    cmap = _audio_gradient(base_hex, features, vibe)
    rgb = cmap[(field * 255).astype(np.uint8)]

    halo = gaussian_filter(field, sigma=2.5 + vibe * 3.0)
    rgb += halo[..., np.newaxis] * np.array([0.03, 0.04, 0.05], dtype=np.float32)

    rng = np.random.default_rng(_song_seed(features) + 17)
    grain = rng.normal(0.0, 0.008 + texture * 0.012, (height, width, 1)).astype(
        np.float32
    )
    rgb = np.clip(rgb + grain, 0.0, 1.0)

    return (rgb * 255).clip(0, 255).astype(np.uint8)
