"""
GIF and MP4 writer for murmur.

Writes rendered frames to disk as a looping GIF (primary)
or MP4 (optional, requires imageio-ffmpeg).
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
from PIL import Image


def write_gif(
    frames: List[np.ndarray],
    output_path: str,
    fps: int = 24,
    loop: int = 0,
) -> None:
    """
    Write a list of (H, W, 3) uint8 frames to a looping GIF.

    Parameters
    ----------
    frames      : list of RGB uint8 arrays
    output_path : output file path (should end in .gif)
    fps         : frames per second (controls frame duration)
    loop        : 0 = loop forever, n = loop n times
    """
    if not frames:
        raise ValueError("No frames to write")

    duration_ms = int(1000 / fps)  # ms per frame

    pil_frames = [Image.fromarray(f, mode="RGB") for f in frames]

    print(f"  Writing GIF: {output_path} ({len(frames)} frames @ {fps} fps)")

    pil_frames[0].save(
        output_path,
        format="GIF",
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=loop,
        optimize=False,
    )

    size_kb = Path(output_path).stat().st_size / 1024
    print(f"  Done. File size: {size_kb:.1f} KB")


def write_mp4(
    frames: List[np.ndarray],
    output_path: str,
    fps: int = 24,
) -> None:
    """
    Write frames to MP4 (requires imageio[ffmpeg]).

    Parameters
    ----------
    frames      : list of (H, W, 3) uint8 arrays
    output_path : output file path (should end in .mp4)
    fps         : frames per second
    """
    try:
        import imageio
    except ImportError:
        raise ImportError(
            "imageio is required for MP4 output. Install with: pip install imageio[ffmpeg]"
        )

    print(f"  Writing MP4: {output_path} ({len(frames)} frames @ {fps} fps)")

    writer = imageio.get_writer(
        output_path,
        fps=fps,
        codec="libx264",
        quality=8,
        pixelformat="yuv420p",
    )
    for frame in frames:
        writer.append_data(frame)
    writer.close()

    size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"  Done. File size: {size_mb:.2f} MB")
