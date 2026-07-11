import matplotlib.pyplot as plt
import numpy as np
import time
from murmur.audio.features import SongDNA
from murmur.v2.engine import MandalaEngine
from murmur.v2.palette import build_song_palette
import matplotlib.colors as mcolors

print("Running simulation with 100,000 agents...")
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


# Patch the engine to force 100k agents
class HighDensityEngine(MandalaEngine):
    def __init__(self, dna, width, height):
        from murmur.v2.init_geometry import build_init

        self.dna = dna
        self.W = width
        self.H = height

        trail_map, ax, ay, angles = build_init(dna, width, height)
        # Override agents to 100,000
        n_agents = 100000
        import math

        h = np.linspace(0, 2 * math.pi, n_agents, endpoint=False).astype(np.float32)
        agent_r = height / 20.0

        ring_cx = height / 2.0
        ring_cy = height / 2.0
        ax = (agent_r * np.cos(h) + ring_cx).astype(np.float32)
        ay = (agent_r * np.sin(h) + ring_cy).astype(np.float32)
        ax = np.clip(ax, 0, width - 1).astype(np.float32)
        ay = np.clip(ay, 0, height - 1).astype(np.float32)
        angles = (h + math.pi).astype(np.float32)

        self.trail_map = trail_map
        self.ax = ax
        self.ay = ay
        self.angles = angles
        self.rng = np.random.default_rng(dna.seed)
        self.p = self._params(len(ax))


t0 = time.time()
engine = HighDensityEngine(dna, 800, 600)
trail_map = engine.run()
print(f"Simulation took {time.time()-t0:.1f}s")

mask = trail_map > 0
log_map = np.zeros_like(trail_map)
log_map[mask] = np.log(trail_map[mask])

if np.any(mask):
    v_min, v_max = log_map[mask].min(), log_map[mask].max()
    log_map[mask] = (log_map[mask] - v_min) / (v_max - v_min)

palette = build_song_palette(dna)
cmap = mcolors.ListedColormap(palette / 255.0)

plt.figure(figsize=(10, 10), facecolor="black")
plt.imshow(log_map, cmap=cmap, interpolation="bilinear")
plt.axis("off")
plt.tight_layout(pad=0)
plt.savefig("mpl_test_100k.png", dpi=300, bbox_inches="tight", facecolor="black")
print("Saved mpl_test_100k.png")
