"""
True Fronkonstin Lombardi Network (The Sound of the Spheres)
Translated from Antonio Sánchez Chinchón's R implementation.
"""

from __future__ import annotations

import hashlib
import math
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter

from murmur.audio.features import GlobalAudioFeatures
from murmur.rendering.palette import hex_to_hsl, hsl_to_rgb


def create_arc(x, y, xend, yend, x0, y0, num_lines=5):
    """Generate points for a sketched arc between two points, around (x0,y0)."""
    r1 = math.hypot(x - x0, y - y0)
    r2 = math.hypot(xend - x0, yend - y0)
    base_radius = (r1 + r2) / 2.0

    ini_angle = math.atan2(y - y0, x - x0)
    end_angle = math.atan2(yend - y0, xend - x0)

    # Normalize angles to take the shortest path or specific winding
    if (end_angle - ini_angle) < -math.pi:
        end_angle += 2 * math.pi
    if (end_angle - ini_angle) > math.pi:
        ini_angle += 2 * math.pi

    end_angle = ini_angle + 2 * math.pi

    steps = 180
    angles = np.linspace(ini_angle, end_angle, steps)

    arcs = []
    # Create multiple parallel loops (sketched effect)
    for _ in range(num_lines):
        r_offset = np.random.uniform(0, 15)  # slight offset
        pts = []
        for a in angles:
            px = (base_radius + r_offset) * math.cos(a) + x0
            py = (base_radius + r_offset) * math.sin(a) + y0
            pts.append((px, py))
        arcs.append(pts)
    return arcs


def make_asteroid(rng, x0, y0, r, name, base_rgb):
    """Generate concentric offset circles for a planet/sun."""
    # We will return a list of polygons/circles to draw
    n = 7
    if name == "sun":
        # Variations of the base color, getting darker
        h, s, l = hex_to_hsl(base_rgb)
        pal = [hsl_to_rgb(h, s, max(0, l - i * 0.1)) for i in range(n)]
    else:
        # Grayscale
        pal = [hsl_to_rgb(0, 0, 0.8 - i * 0.1) for i in range(n)]

    angle = rng.uniform(0, 2 * math.pi)
    rc = rng.uniform(0, r * 0.6)

    centers = []
    for i in range(n):
        frac = i / (n - 1) if n > 1 else 0
        cx = x0 + (rc * frac) * math.cos(angle)
        cy = y0 + (rc * frac) * math.sin(angle)
        # Radius decreases
        curr_r = r - frac * (r * 0.8)  # Gets smaller
        centers.append((cx, cy, curr_r, pal[i]))

    # Reverse so largest is drawn first
    return centers


def build_spheres_image(
    features: GlobalAudioFeatures,
    base_hex: str,
    vibe: float,
    width: int,
    height: int,
) -> np.ndarray:
    """Generate the exact Fronkonstin Lombardi Network."""
    fingerprint = (
        f"{features.bpm:.2f}|{features.mean_rms:.4f}|{features.dominant_chroma}"
    )
    digest = hashlib.sha256(fingerprint.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], "little", signed=False)
    rng = np.random.default_rng(seed)

    # 1. Audio Mapping to Parameters
    bpm_norm = np.clip((features.bpm - 60.0) / 120.0, 0.0, 1.0)
    energy = np.clip(features.mean_rms, 0.0, 1.0)
    texture = np.clip(features.mean_bandwidth, 0.0, 1.0)

    # Fronkonstin params
    max_ratio = 48 + (energy * 30)  # Sun size
    nu_nodes = 15 + int(bpm_norm * 40)  # 15 to 55 planets
    nu_branches = 5 + int(texture * 20)  # 5 to 25 orbits
    if nu_branches >= nu_nodes:
        nu_branches = nu_nodes - 1

    # 2. Assign nodes to branches
    # Each branch gets 1 initial node
    nodes_raw_init = {
        i: branch
        for i, branch in zip(
            rng.choice(np.arange(1, nu_nodes + 1), nu_branches, replace=False),
            range(1, nu_branches + 1),
        )
    }

    # Remaining nodes assigned randomly
    nodes_raw = nodes_raw_init.copy()
    remaining = [n for n in range(1, nu_nodes + 1) if n not in nodes_raw]
    for n in remaining:
        nodes_raw[n] = int(rng.integers(1, nu_branches + 1))

    # Build list_branches (seeds)
    list_branches = {}
    for i in range(1, nu_branches + 1):
        # Find all nodes in branches < i
        available_seeds = [k for k, v in nodes_raw.items() if v < i]
        if not available_seeds:
            # Fallback for branch 1
            available_seeds = [k for k, v in nodes_raw.items() if v == i]

        seed_node = int(rng.choice(available_seeds))
        branch_nodes = [k for k, v in nodes_raw.items() if v == i]
        list_branches[i] = [seed_node] + branch_nodes

    # 3. Locate Nodes
    nodes_partial = {}  # branch_id -> (x0, y0, range)
    nodes_partial[1] = (0.0, 0.0, 2 * math.pi)

    nodes_xy = {}  # node_id -> (x, y)

    # Initialize first node
    first_seed = list_branches[min(nu_branches, 2)][0]
    nodes_xy[first_seed] = (0.0, 1.5 * max_ratio)

    for i in range(1, nu_branches + 1):
        x0, y0, rnge = nodes_partial[i]
        ids = list_branches[i]

        # Locate nodes for this branch
        anchor = ids[0]
        if anchor in nodes_xy:
            ax, ay = nodes_xy[anchor]
            radius = math.hypot(ax - x0, ay - y0)
            angle = math.atan2(ay - y0, ax - x0)

            nodes_to_locate = [n for n in ids if n not in nodes_xy]
            if nodes_to_locate:
                angles = np.linspace(angle, angle + rnge, len(nodes_to_locate) + 2)[
                    1:-1
                ]
                for idx, n in enumerate(nodes_to_locate):
                    nodes_xy[n] = (
                        x0 + radius * math.cos(angles[idx]),
                        y0 + radius * math.sin(angles[idx]),
                    )

        # Define next branch origin
        if i < nu_branches:
            next_seed = list_branches[i + 1][0]
            if next_seed in nodes_xy:
                sx, sy = nodes_xy[next_seed]
                angle_origin = math.atan2(sy, sx)  # Angle from global origin
                # Move towards global origin by a random amount
                dist = rng.uniform(1.5, 4.0) * max_ratio
                new_x0 = sx + dist * math.cos(angle_origin - math.pi)
                new_y0 = sy + dist * math.sin(angle_origin - math.pi)
                new_range = rng.uniform(math.pi / 2, 2 * math.pi) * rng.choice([-1, 1])
                nodes_partial[i + 1] = (new_x0, new_y0, new_range)
            else:
                nodes_partial[i + 1] = (0.0, 0.0, 2 * math.pi)

    # 4. Generate Edges (Arcs)
    arcs_to_draw = []
    for i in range(1, nu_branches + 1):
        branch_nodes = list_branches[i]
        x0, y0, _ = nodes_partial[i]
        for j in range(len(branch_nodes) - 1):
            n1 = branch_nodes[j]
            n2 = branch_nodes[j + 1]
            if n1 in nodes_xy and n2 in nodes_xy:
                x1, y1 = nodes_xy[n1]
                x2, y2 = nodes_xy[n2]
                arcs = create_arc(x1, y1, x2, y2, x0, y0, num_lines=3)
                arcs_to_draw.extend(arcs)

    # 5. Generate Asteroids
    asteroids_to_draw = []
    # Sun
    asteroids_to_draw.append(make_asteroid(rng, 0, 0, max_ratio, "sun", base_hex))
    # Planets
    for nid, (nx, ny) in nodes_xy.items():
        r = abs(rng.normal(14, 3))
        asteroids_to_draw.append(make_asteroid(rng, nx, ny, r, "planet", base_hex))

    # 6. Render to Image
    # Find bounding box to frame it perfectly
    min_x = min_y = float("inf")
    max_x = max_y = float("-inf")
    for ast in asteroids_to_draw:
        for cx, cy, r, _ in ast:
            min_x = min(min_x, cx - r)
            max_x = max(max_x, cx + r)
            min_y = min(min_y, cy - r)
            max_y = max(max_y, cy + r)

    # Add padding
    pad = min(max_x - min_x, max_y - min_y) * 0.15
    if pad == float("inf") or pad == 0:
        pad = 100
        min_x, max_x = -100, 100
        min_y, max_y = -100, 100

    min_x -= pad
    max_x += pad
    min_y -= pad
    max_y += pad

    # Scale to canvas (super-sample 4x for anti-aliasing)
    scale = min(width / (max_x - min_x), height / (max_y - min_y))
    ox = width / 2 - ((max_x + min_x) / 2) * scale
    oy = height / 2 - ((max_y + min_y) / 2) * scale

    SS = 4  # supersampling
    W, H = width * SS, height * SS
    canvas = Image.new("RGB", (W, H), (10, 10, 12))  # Dark background
    draw = ImageDraw.Draw(canvas)

    def transform(px, py):
        return (int((px * scale + ox) * SS), int((py * scale + oy) * SS))

    # Draw Arcs
    for arc_pts in arcs_to_draw:
        mapped = [transform(px, py) for (px, py) in arc_pts]
        draw.line(mapped, fill=(200, 200, 200), width=SS * 1, joint="curve")

    # Draw Asteroids
    for ast in asteroids_to_draw:
        for cx, cy, r, col in ast:
            mx, my = transform(cx, cy)
            mr = r * scale * SS
            bbox = (mx - mr, my - mr, mx + mr, my + mr)
            # Ensure col is integer tuple
            col_int = tuple(int(c) for c in col)
            draw.ellipse(bbox, fill=col_int, outline=(50, 50, 50), width=SS * 1)

    # Downsample
    canvas = canvas.resize((width, height), Image.Resampling.LANCZOS)
    return np.array(canvas)
