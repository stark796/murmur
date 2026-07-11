import matplotlib.pyplot as plt
import numpy as np
import time
from murmur.audio.features import SongDNA
from murmur.v2.palette import build_song_palette
import matplotlib.colors as mcolors
import math

print("Running exact Fronkonstin buggy simulation...")

W, H = 800, 600
n_agents = 1000
iters = 2000

# 1D array to represent the corrupted memory space
# It is accessed as envM[x + y * 600]
memory_size = W * H
envM = np.zeros(memory_size, dtype=np.float32)

# Magnetic ring
for i in range(H):
    for j in range(W):
        dist = math.sqrt((i - H / 2) ** 2 + (j - H / 2) ** 2)
        if H / 8 < dist < H / 6:
            envM[i + j * H] = (
                5.0  # He initialized it with envM(i,j), so i is row (0..599), j is col (0..799)
            )

# Agents
h = np.linspace(0, 2 * math.pi, n_agents, endpoint=False)
agent_x = (H / 20) * np.cos(h) + H / 2
agent_y = (H / 20) * np.sin(h) + H / 2
agent_h = h + math.pi

# Parameters
decayT = 0.1
depT = 15.0
SO = 6.0
SS = 1.0
FL_angle = 22.5 * math.pi / 180
FR_angle = -22.5 * math.pi / 180
RA = 45 * math.pi / 180


def get_index(M, val):
    val = int(round(val))
    if val < 0:
        return (M + val % M) % M
    if val >= M:
        return val % M
    return val


# Simulation loop
rng = np.random.default_rng(123)
t0 = time.time()
for step in range(iters):
    # Sensor stage
    for i in range(n_agents):
        # Buggy sensor reads
        Fx = get_index(H, agent_x[i] + SO * math.cos(agent_h[i]))
        Fy = get_index(W, agent_y[i] + SO * math.sin(agent_h[i]))

        FLx = get_index(H, agent_x[i] + SO * math.cos(agent_h[i] + FL_angle))
        FLy = get_index(W, agent_y[i] + SO * math.sin(agent_h[i] + FL_angle))

        FRx = get_index(H, agent_x[i] + SO * math.cos(agent_h[i] + FR_angle))
        FRy = get_index(W, agent_y[i] + SO * math.sin(agent_h[i] + FR_angle))

        # Read from memory space: envM(Fx, Fy) -> envM[Fx + Fy * 600]
        F = envM[Fx + Fy * H]
        FL = envM[FLx + FLy * H]
        FR = envM[FRx + FRy * H]

        if F > FL and F > FR:
            pass
        elif F < FL and F < FR:
            if rng.random() < 0.5:
                agent_h[i] += RA
            else:
                agent_h[i] -= RA
        elif FL < FR:
            agent_h[i] -= RA
        elif FR < FL:
            agent_h[i] += RA

    # Motor stage
    for i in range(n_agents):
        agent_x[i] = get_index(W, agent_x[i] + SS * math.cos(agent_h[i]))
        agent_y[i] = get_index(H, agent_y[i] + SS * math.sin(agent_h[i]))

    # Deposition stage
    for i in range(n_agents):
        # Write to memory space: envM(x, y) -> envM[x + y * 600]
        idx = int(agent_x[i]) + int(agent_y[i]) * H
        if idx < memory_size:
            envM[idx] += depT

    # Evaporation
    envM *= 1.0 - decayT

print(f"Simulation took {time.time()-t0:.1f}s")

# Reconstruct 2D image for plotting
# R's melt(envM) maps envM(i, j) to x=i, y=j
trail_map = np.zeros(
    (W, H), dtype=np.float32
)  # Note dimensions: x is 0..599, y is 0..799?
# Wait, envM was 600x800. melt makes Var1=1..600, Var2=1..800.
# So plot x (horizontal) is 1..600, y (vertical) is 1..800.
for i in range(H):
    for j in range(W):
        trail_map[j, i] = envM[i + j * H]

# We need to rotate it back so it's landscape? Or just plot as is to see what it looks like.
# Let's just plot trail_map as it is (800x600 matrix -> imshow will show 800 rows, 600 cols)
# But we want X to be horizontal, so let's transpose.
trail_map = trail_map.T  # Now 600x800

mask = trail_map > 0
log_map = np.zeros_like(trail_map)
log_map[mask] = np.log(trail_map[mask])

if np.any(mask):
    v_min, v_max = log_map[mask].min(), log_map[mask].max()
    log_map[mask] = (log_map[mask] - v_min) / (v_max - v_min)

dna = SongDNA(
    rhythm=0.5,
    density=0.5,
    brightness=0.5,
    complexity=0.5,
    harmonic=0.5,
    dominant_chroma=3,
    bpm=120,
    seed=123,
)
palette = build_song_palette(dna)
cmap = mcolors.ListedColormap(palette / 255.0)

plt.figure(figsize=(10, 10), facecolor="black")
plt.imshow(log_map, cmap=cmap, interpolation="bilinear")
plt.axis("off")
plt.tight_layout(pad=0)
plt.savefig("mpl_buggy.png", dpi=300, bbox_inches="tight", facecolor="black")
print("Saved mpl_buggy.png")
