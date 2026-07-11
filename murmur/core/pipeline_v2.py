"""
murmur v2 — Top-level pipeline.

The single function: paint(audio_path, output_path, resolution)

  Song → DNA → Palette → Simulation → Render → PNG

No vibe. No engine selection. No user color input.
The song decides everything.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from PIL import Image

from murmur.audio.analyzer import AudioAnalyzer
from murmur.v2.engine import MandalaEngine
from murmur.v2.palette import build_song_palette
from murmur.v2.render import render_trail


def paint(
    audio_path: str,
    output_path: str,
    resolution: int = 1024,
    sim_resolution: int | None = None,
) -> str:
    """
    Generate a song-specific abstract mandala image.

    Parameters
    ----------
    audio_path     : path to .mp3 or .wav
    output_path    : where to save the PNG (will be PNG regardless of extension)
    resolution     : final output resolution (square), e.g. 4096 or 9192
    sim_resolution : simulation canvas size (default: min(resolution, 2048))
                     Increase for more filament detail at the cost of time.

    Returns
    -------
    Path to saved file.
    """
    t0 = time.perf_counter()

    # Ensure output is PNG
    out_path = Path(output_path).with_suffix(".png")

    # Simulation resolution: cap at 2048 by default for speed,
    # but user can pass sim_resolution=4096 etc for more detail.
    if sim_resolution is None:
        sim_resolution = min(resolution, 2048)

    print("\n🎨 murmur paint")
    print(f"  Audio     : {audio_path}")
    print(f"  Simulate  : 800×600 (Fronkonstin native)")
    print(f"  Output    : {resolution}×{resolution} → {out_path}")
    print()

    # ------------------------------------------------------------------
    # Step 1: Extract Song DNA
    # ------------------------------------------------------------------
    print("① Extracting Song DNA...")
    analyzer = AudioAnalyzer(
        audio_path=audio_path,
        start_time=0.0,
        duration=None,
        fps=24,
    )
    dna = analyzer.analyze_dna()
    print()

    # ------------------------------------------------------------------
    # Step 2: Build song palette
    # ------------------------------------------------------------------
    print("② Building song palette...")
    palette = build_song_palette(dna)
    print(f"  Base hue: {dna.dominant_chroma}/12 = {dna.dominant_chroma/12.0:.3f}")
    print(f"  Saturation: {0.30 + dna.harmonic * 0.65:.3f}")
    print()

    # ------------------------------------------------------------------
    # Step 3: Grow the Physarum at FRONKONSTIN'S EXACT RESOLUTION
    # ------------------------------------------------------------------
    # Fronkonstin uses imageW=800, imageH=600. Simulating at any other
    # resolution changes the dynamics because SO, SS, agent radius, and
    # ring radius are all in absolute pixels. We simulate at his exact
    # resolution and upscale afterward.
    SIM_W, SIM_H = 800, 600
    print(f"③ Growing Physarum (seed={dna.seed})...")
    engine = MandalaEngine(dna, SIM_W, SIM_H)
    print()

    try:
        from tqdm import tqdm

        with tqdm(total=engine.p.n_steps, desc="  Simulation", unit="step") as pbar:
            trail_map = engine.run(progress_cb=lambda s, t: pbar.update(1))
    except ImportError:
        trail_map = engine.run()

    print()

    # ------------------------------------------------------------------
    # Step 4: Render to RGB
    # ------------------------------------------------------------------
    print("④ Rendering trail map...")
    rgb = render_trail(trail_map, palette)

    # ------------------------------------------------------------------
    # Step 5: Upscale to output resolution
    # ------------------------------------------------------------------
    print(f"⑤ Upscaling {SIM_W}×{SIM_H} → {resolution}×{resolution} (LANCZOS)...")
    img = Image.fromarray(rgb)
    img = img.resize((resolution, resolution), Image.Resampling.LANCZOS)
    rgb = np.array(img)

    # ------------------------------------------------------------------
    # Step 6: Save
    # ------------------------------------------------------------------
    print(f"⑥ Saving → {out_path}")
    Image.fromarray(rgb).save(out_path, dpi=(300, 300))

    elapsed = time.perf_counter() - t0
    print(f"\n✓ Done in {elapsed:.1f}s")
    print(f"  {out_path}")
    return str(out_path)
