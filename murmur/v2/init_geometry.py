"""
murmur v2 — Agent initialization geometry.

EXACT Fronkonstin replication. No DNA scaling, no creativity.
Hardcoded to his literal R script values.
"""

from __future__ import annotations

import math
import numpy as np
from murmur.audio.features import SongDNA


def build_init(
    dna: SongDNA,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build EXACT Fronkonstin initial conditions.

    Returns
    -------
    trail_map : (H, W) float32
    ax, ay    : (N,) float32  — agent positions
    angles    : (N,) float32  — agent headings
    """
    # ================================================================
    # 1. Trail map — empty + magnetic ring
    # ================================================================
    # R code:
    #   envM <- matrix(0, imageH, imageW)
    #   if(sqrt((i-imageH/2)^2+(j-imageH/2)^2)>imageH/8 &
    #      sqrt((i-imageH/2)^2+(j-imageH/2)^2)<imageH/6) envM[i,j]=5
    #
    # NOTE: R uses imageH/2 for BOTH x and y center of the ring,
    # even though imageW != imageH. This is intentional — it makes
    # the ring offset to the left on a landscape canvas.
    trail_map = np.zeros((height, width), dtype=np.float32)
    Y, X = np.ogrid[:height, :width]

    # Use height/2 for both axes, exactly like R
    ring_cx = height / 2.0
    ring_cy = height / 2.0
    dist = np.sqrt((X - ring_cx) ** 2 + (Y - ring_cy) ** 2).astype(np.float32)

    r_inner = height / 8.0
    r_outer = height / 6.0
    mask = (dist > r_inner) & (dist < r_outer)
    trail_map[mask] = 5.0

    # ================================================================
    # 2. Agents — 1000, tiny circle
    # ================================================================
    # R code:
    #   agents <- 1000
    #   x = (imageH/20)*cos(h)+imageH/2
    #   y = (imageH/20)*sin(h)+imageH/2
    n_agents = 1000

    h = np.linspace(0, 2 * math.pi, n_agents, endpoint=False).astype(np.float32)
    agent_r = height / 20.0

    # Again, R uses imageH/2 for BOTH x and y center
    ax = (agent_r * np.cos(h) + ring_cx).astype(np.float32)
    ay = (agent_r * np.sin(h) + ring_cy).astype(np.float32)

    ax = np.clip(ax, 0, width - 1).astype(np.float32)
    ay = np.clip(ay, 0, height - 1).astype(np.float32)

    # ================================================================
    # 3. Heading — h + pi, zero jitter
    # ================================================================
    # R code: h = jitter(h+pi, amount = 0)
    heading = (h + math.pi).astype(np.float32)

    return (trail_map, ax, ay, heading)
