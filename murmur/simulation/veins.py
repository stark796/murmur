"""Rigid vascular growth map for song-driven abstract images."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter

from murmur.audio.features import GlobalAudioFeatures


@dataclass
class VeinProfile:
    """Structural controls derived from a song's macro audio features."""

    seed: int
    root_count: int
    max_depth: int
    branch_probability: float
    angle_spread: float
    length_scale: float
    thickness_scale: float
    angular_snap: float
    main_angle: float


def build_vein_profile(features: GlobalAudioFeatures, vibe: float) -> VeinProfile:
    """Convert global song features into a repeatable vascular fingerprint."""
    fingerprint = (
        f"{features.bpm:.2f}|{features.mean_rms:.4f}|"
        f"{features.mean_centroid:.4f}|{features.mean_bandwidth:.4f}|"
        f"{features.dominant_chroma}"
    )
    digest = hashlib.sha256(fingerprint.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], "little", signed=False)

    bpm_norm = np.clip((features.bpm - 60.0) / 120.0, 0.0, 1.0)
    energy = np.clip(features.mean_rms, 0.0, 1.0)
    brightness = np.clip(features.mean_centroid, 0.0, 1.0)
    texture = np.clip(features.mean_bandwidth, 0.0, 1.0)

    return VeinProfile(
        seed=seed,
        root_count=9 + int(round(features.dominant_chroma % 5)) + int(texture * 6),
        max_depth=10 + int(round((energy + texture + vibe) * 5)),
        branch_probability=0.52 + texture * 0.25 + vibe * 0.18,
        angle_spread=math.radians(14.0 + brightness * 30.0 + vibe * 12.0),
        length_scale=0.13 + bpm_norm * 0.09 + energy * 0.04,
        thickness_scale=0.0035 + energy * 0.008 + vibe * 0.003,
        angular_snap=math.radians(24.0 - brightness * 12.0),
        main_angle=(features.dominant_chroma / 12.0) * math.tau,
    )


def _snap_angle(angle: float, step: float) -> float:
    return round(angle / step) * step


def _draw_branch(
    draw: ImageDraw.ImageDraw,
    rng: np.random.Generator,
    profile: VeinProfile,
    x: float,
    y: float,
    angle: float,
    length: float,
    thickness: float,
    depth: int,
    width: int,
    height: int,
) -> None:
    if depth <= 0 or length < 3.0 or thickness < 0.6:
        return

    n_segments = max(3, int(length / max(5.0, width * 0.006)))
    points = [(x, y)]
    segment_length = length / n_segments
    current_angle = angle

    for _ in range(n_segments):
        current_angle = _snap_angle(
            current_angle + rng.normal(0.0, profile.angle_spread * 0.12),
            profile.angular_snap,
        )
        x += math.cos(current_angle) * segment_length
        y += math.sin(current_angle) * segment_length
        points.append((x, y))

    line_width = max(1, int(round(thickness)))
    draw.line(points, fill=255, width=line_width, joint="curve")

    if rng.random() < 0.8:
        node_radius = max(1, int(round(thickness * 0.55)))
        draw.ellipse(
            (x - node_radius, y - node_radius, x + node_radius, y + node_radius),
            fill=255,
        )

    if x < -width * 0.15 or x > width * 1.15 or y < -height * 0.15 or y > height * 1.15:
        return

    child_count = 1
    if rng.random() < profile.branch_probability:
        child_count += 1
    if rng.random() < profile.branch_probability * 0.42:
        child_count += 1

    for child_idx in range(child_count):
        direction = -1.0 if child_idx % 2 == 0 else 1.0
        if child_count == 1:
            direction = rng.choice(np.array([-1.0, 1.0]))
        fork_angle = (
            current_angle
            + direction
            * rng.uniform(profile.angle_spread * 0.55, profile.angle_spread * 1.35)
            + rng.normal(0.0, profile.angle_spread * 0.15)
        )
        fork_angle = _snap_angle(fork_angle, profile.angular_snap)
        length_decay = rng.uniform(0.62, 0.84)
        thickness_decay = rng.uniform(0.56, 0.78)
        _draw_branch(
            draw=draw,
            rng=rng,
            profile=profile,
            x=x,
            y=y,
            angle=fork_angle,
            length=length * length_decay,
            thickness=thickness * thickness_decay,
            depth=depth - 1,
            width=width,
            height=height,
        )


def generate_vein_map(
    features: GlobalAudioFeatures,
    vibe: float,
    width: int,
    height: int,
) -> np.ndarray:
    """Generate a hard-edged vascular density map in [0, 1]."""
    profile = build_vein_profile(features, vibe)
    rng = np.random.default_rng(profile.seed)

    canvas = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(canvas)

    base_length = min(width, height) * profile.length_scale
    base_thickness = max(1.0, min(width, height) * profile.thickness_scale)

    source_count = 2 + int(np.clip(features.mean_bandwidth * 4 + vibe * 2, 0, 6))
    sources: list[tuple[float, float, float]] = []
    center_x = width * (0.50 + rng.normal(0.0, 0.035))
    center_y = height * (0.50 + rng.normal(0.0, 0.035))
    sources.append((center_x, center_y, profile.main_angle))

    for source_idx in range(1, source_count):
        theta = (
            profile.main_angle
            + source_idx * math.tau / source_count
            + rng.normal(0.0, 0.35)
        )
        radius = min(width, height) * rng.uniform(0.16, 0.42)
        sx = center_x + math.cos(theta) * radius
        sy = center_y + math.sin(theta) * radius
        sources.append((sx, sy, theta + math.pi))

    for source_idx, (sx, sy, source_angle) in enumerate(sources):
        roots_here = max(
            3, int(round(profile.root_count / source_count)) + rng.integers(-1, 3)
        )
        for root_idx in range(roots_here):
            root_angle = (
                source_angle
                + (root_idx / roots_here) * math.tau
                + rng.normal(0.0, profile.angle_spread * 0.35)
            )
            root_angle = _snap_angle(root_angle, profile.angular_snap)
            _draw_branch(
                draw=draw,
                rng=rng,
                profile=profile,
                x=sx,
                y=sy,
                angle=root_angle,
                length=base_length * rng.uniform(0.9, 1.7),
                thickness=base_thickness * rng.uniform(0.65, 1.45),
                depth=profile.max_depth - source_idx // 2,
                width=width,
                height=height,
            )

    vein_map = np.asarray(canvas, dtype=np.float32) / 255.0

    # Keep the core crisp; only weld sub-pixel gaps so the particle renderer has
    # a continuous probability field to sample from.
    vein_map = np.maximum(vein_map, gaussian_filter(vein_map, sigma=0.35) * 0.55)
    max_value = vein_map.max()
    if max_value > 0:
        vein_map /= max_value
    return vein_map.astype(np.float32)
