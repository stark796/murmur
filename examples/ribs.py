"""
Example: Generate abstract art from Ribs by Lorde.

Place 'ribs.mp3' (or ribs.wav) in the same directory as this script,
then run:
    python examples/ribs.py

Or adjust the path and parameters below.
"""

import os
import sys

# Allow running from examples/ directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from murmur.core.config import MurmurConfig
from murmur.core.pipeline import generate

# --- Configure ---
config = MurmurConfig(
    audio_path="examples/Ribs.mp3",  # path to Ribs by Lorde
    vibe=0.0,  # serene vibe for smooth, undisturbed glassy trails
    fps=24,
    resolution=512,  # Will render at 1024x1024 (2x) internally
    output_path="ribs_art_full.png",
)

if __name__ == "__main__":
    out = generate(config)
    print(f"Output saved to: {out}")
