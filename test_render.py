import matplotlib.pyplot as plt
import numpy as np
from murmur.audio.features import SongDNA
from murmur.v2.engine import MandalaEngine
from murmur.v2.palette import build_song_palette
import matplotlib.colors as mcolors

print("Running pure simulation...")
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
engine = MandalaEngine(dna, 800, 600)
trail_map = engine.run()

print("Rendering with matplotlib (like ggplot geom_raster)...")
# Filter v > 0
mask = trail_map > 0

# Create an RGBA image filled with white (or transparent)
output = np.zeros((600, 800, 4), dtype=np.float32)
output[:, :, 3] = 1.0  # alpha

# Apply log
log_map = np.zeros_like(trail_map)
log_map[mask] = np.log(trail_map[mask])

# Normalize log_map for the masked region
if np.any(mask):
    v_min, v_max = log_map[mask].min(), log_map[mask].max()
    log_map[mask] = (log_map[mask] - v_min) / (v_max - v_min)

# Build a colormap from the palette
palette = build_song_palette(dna)
cmap = mcolors.ListedColormap(palette / 255.0)

plt.figure(figsize=(10, 10), facecolor="black")
# Plot using imshow with bilinear interpolation, just like R
plt.imshow(log_map, cmap=cmap, interpolation="bilinear")
plt.axis("off")
plt.tight_layout(pad=0)
plt.savefig("mpl_test.png", dpi=300, bbox_inches="tight", facecolor="black")
print("Saved mpl_test.png")
