import math
import numpy as np


def build_abstractions_setup(
    features, width: int, height: int, n_agents: int = 150_000
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate the exact geometric initial conditions from Fronkonstin's `abstractions` repository.
    Returns: (initial_map, ax, ay, angles)
    """
    # 1. Base food map (envM) initialized with zeros
    initial_map = np.zeros((height, width), dtype=np.float32)

    # We modulate the ring sizes slightly based on the song's energy and BPM
    # to give each song a unique abstraction shape.
    norm_bpm = max(0.0, min(1.0, (features.bpm - 60.0) / 120.0))
    energy = np.clip(features.mean_rms, 0.0, 1.0)

    # Fronkonstin base radii: imageH/8 and imageH/6
    base_inner_r = (height / 8.0) * (0.8 + 0.4 * norm_bpm)
    base_outer_r = (height / 6.0) * (0.8 + 0.4 * energy)

    # Ensure outer > inner
    if base_inner_r >= base_outer_r:
        base_outer_r = base_inner_r + (height / 20.0)

    cy, cx = height / 2.0, width / 2.0

    # Create the magnetic disc using numpy broadcasting
    y, x = np.ogrid[:height, :width]
    dist_sq = (x - cx) ** 2 + (y - cy) ** 2

    inner_r_sq = base_inner_r**2
    outer_r_sq = base_outer_r**2

    # Set the ring to strength 5.0 just like the R code
    mask = (dist_sq > inner_r_sq) & (dist_sq < outer_r_sq)
    initial_map[mask] = 5.0

    # 2. Agent spawn positions
    # Fronkonstin base radius: imageH/20
    spawn_radius = (height / 20.0) * (0.5 + 1.0 * norm_bpm)

    # Angles for the circle
    angles_circle = np.linspace(
        0, 2 * math.pi, n_agents, endpoint=False, dtype=np.float32
    )

    # Positions
    ax = (spawn_radius * np.cos(angles_circle) + cx).astype(np.float32)
    ay = (spawn_radius * np.sin(angles_circle) + cy).astype(np.float32)

    # Headings: Fronkonstin sets heading h = h + pi (facing inward)
    # Adding a tiny jitter based on song texture (Chroma) for chaos
    jitter_amt = (features.dominant_chroma / 12.0) * 0.1
    rng = np.random.default_rng(int(features.bpm * 100))
    jitter = rng.uniform(-jitter_amt, jitter_amt, n_agents).astype(np.float32)

    angles = (angles_circle + math.pi + jitter).astype(np.float32)

    return initial_map, ax, ay, angles
