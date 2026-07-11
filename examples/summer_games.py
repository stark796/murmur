import os
import sys

# Allow running from examples/ directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from murmur.core.config import MurmurConfig
from murmur.core.pipeline import generate

# --- Configure ---
config = MurmurConfig(
    audio_path="examples/Summer Games.mp3",
    color="#808080",  # grey
    vibe=0.5,  # balanced vibe
    fps=24,
    resolution=512,  # Will render at 1024x1024 (2x) internally
    output_path="summer_games_art.png",
)

if __name__ == "__main__":
    generate(config)
