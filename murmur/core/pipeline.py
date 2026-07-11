"""
Main pipeline orchestrator for murmur.

Ties together: audio analysis → Physarum simulation → rendering → output.
"""

from __future__ import annotations

import time
import numpy as np

from murmur.core.config import MurmurConfig
from murmur.audio.analyzer import AudioAnalyzer
from murmur.audio.features import audio_to_hex
from murmur.rendering.generator import (
    generate_field_image,
    generate_static_image,
    generate_vein_image,
)
from murmur.simulation.physarum import PhysarumSimulation
from murmur.simulation.veins import generate_vein_map
from murmur.simulation.spheres import build_spheres_image
from murmur.simulation.attractor import AttractorSimulation
from murmur.simulation.lenia import LeniaSimulation, LeniaParams
from murmur.simulation.abstractions import build_abstractions_setup
from murmur.rendering.palette import build_colormap
from murmur.rendering.effects import apply_all_effects
from PIL import Image


def generate(config: MurmurConfig) -> str:
    """
    Run the full murmur pipeline and write the output file.

    Parameters
    ----------
    config : MurmurConfig with all settings

    Returns
    -------
    Path to the output file.
    """
    t_start = time.perf_counter()

    print("\n🌊 murmur — starting render")
    print(
        f"  Audio: {config.audio_path}  [{config.start_time:.1f}s → {config.start_time + config.duration:.1f}s]"
        if config.duration
        else f"  Audio: {config.audio_path}  [{config.start_time:.1f}s → End]"
    )
    print(f"  Engine: {config.engine}")
    print(f"  Vibe: {config.vibe:.2f}")
    print(
        f"  Resolution: {config.resolution}×{config.resolution}  |  {config.fps} fps  |  {config.duration or 'Full'}s"
    )
    print()

    # ------------------------------------------------------------------
    # Step 1: Analyze audio globally
    # ------------------------------------------------------------------
    print("① Analyzing global audio features...")
    analyzer = AudioAnalyzer(
        audio_path=config.audio_path,
        start_time=config.start_time,
        duration=config.duration,
        fps=config.fps,
    )
    features = analyzer.analyze_global()

    print(
        f"  Features extracted: BPM={features.bpm:.1f}, RMS={features.mean_rms:.3f}, Chroma={features.dominant_chroma}"
    )

    # Auto-resolve color
    final_color = config.color
    if not final_color:
        final_color = audio_to_hex(features)
        print(f"  Auto-resolved Color: {final_color}")
    else:
        print(f"  Using manual Color: {final_color}")

    width = config.resolution * 2
    height = config.resolution * 2

    if config.engine == "field":
        print("\n② Solving song-specific abstract field...")
        print("\n③ Rendering high-resolution abstract image...")
        rgb_final = generate_field_image(
            features=features,
            base_hex=final_color,
            vibe=config.vibe,
            width=width,
            height=height,
        )
    elif config.engine == "veins":
        print("\n② Growing song-specific vein structure...")
        vein_map = generate_vein_map(
            features=features,
            vibe=config.vibe,
            width=width,
            height=height,
        )

        print("\n③ Rendering high-resolution abstract image...")
        rgb_final = generate_vein_image(
            features=features,
            base_hex=final_color,
            vibe=config.vibe,
            vein_map=vein_map,
        )
    elif config.engine == "physarum":
        print("\n② Generating stochastic displacement map...")
        unique_seed = int(
            (features.bpm * features.mean_rms * 100) + features.dominant_chroma
        )

        physarum = PhysarumSimulation(
            width=width, height=height, n_agents=100_000, seed=unique_seed
        )

        norm_bpm = max(0.0, min(1.0, (features.bpm - 60.0) / 120.0))
        energy = features.mean_rms
        timbre = features.mean_bandwidth

        class DummyParams:
            sensor_distance = 15.0 + (energy * 60.0)
            step_size = 1.0 + (norm_bpm * 8.0)
            sensor_angle = 0.2 + (timbre * 1.5)
            rotation_angle = 0.2 + (timbre * 1.5)
            decay_factor = 0.85 + (energy * 0.14)
            speed = 2.0
            deposit_amount = 0.1
            diffusion_sigma = 1.5
            advection_strength = 0.0
            noise_amount = 0.0

        d_params = DummyParams()

        for _ in range(200):
            physarum.step(d_params)

        print("\n③ Rendering high-resolution abstract image...")
        rgb_final = generate_static_image(
            features=features,
            base_hex=final_color,
            vibe=config.vibe,
            width=width,
            height=height,
            stochastic_map=physarum.trail_map,
        )
    elif config.engine == "spheres":
        print("\n② Generating True Lombardi mathematical spheres graph...")
        # 1. Build the mathematical image directly
        rgb = build_spheres_image(
            features=features,
            base_hex=final_color,
            vibe=config.vibe,
            width=width,
            height=height,
        )

        print("\n③ Rendering high-resolution abstract image...")
        # Apply standard cinematic effects (bloom, vignette)
        rgb_final = apply_all_effects(rgb, vibe=config.vibe, rms=features.mean_rms)
    elif config.engine == "attractor":
        print("\n② Calculating Strange Attractor (Clifford)...")
        # Map audio features to chaos parameters [-2, 2]
        # We use a mix of features to ensure every song is completely distinct
        a = (features.dominant_chroma / 6.0) - 1.0  # -1 to 1 based on key
        b = (features.bpm / 90.0) - 1.0  # approx -0.3 to 1.0 based on tempo
        c = (features.mean_rms * 2.0) - 1.0  # -1 to 1 based on loudness
        d = (features.mean_bandwidth * 2.0) - 1.0  # -1 to 1 based on texture

        # Scale to [-2.5, 2.5] for more chaotic folding
        a, b, c, d = a * 2.5, b * 2.5, c * 2.5, d * 2.5

        attractor = AttractorSimulation(width=width, height=height, n_iters=15_000_000)

        class DummyParams:
            pass

        dp = DummyParams()
        dp.a, dp.b, dp.c, dp.d = a, b, c, d

        attractor.step(dp)
        field = attractor.trail

        print("\n③ Rendering high-resolution abstract image...")
        hue_shift = (features.dominant_chroma / 12.0) * 0.2
        cmap = build_colormap(final_color, hue_shift=hue_shift)
        indices = (field * 255).astype(np.int32)
        rgb = cmap[indices]
        rgb_final = apply_all_effects(rgb, vibe=config.vibe, rms=features.mean_rms)

    elif config.engine == "lenia":
        print("\n② Growing Continuous Cellular Automata (Lenia)...")

        # Map audio features to biology parameters
        energy = np.clip(features.mean_rms, 0.0, 1.0)
        texture = np.clip(features.mean_bandwidth, 0.0, 1.0)

        # Standard Orbium is mu=0.15, sigma=0.017
        # We vary it slightly based on song
        mu = 0.12 + (energy * 0.08)  # 0.12 to 0.20
        sigma = 0.01 + (texture * 0.015)  # 0.01 to 0.025
        dt = 0.1

        lenia = LeniaSimulation(
            width=width,
            height=height,
            R=15.0 + (config.vibe * 5.0),
            seed=int(features.bpm * 100),
        )
        params = LeniaParams(mu=mu, sigma=sigma, dt=dt)

        # Run 200 steps to let it bloom
        for _ in range(200):
            lenia.step(params)

        field = lenia.trail

        # Smooth the field slightly for a silkier look
        import scipy.ndimage

        field = scipy.ndimage.gaussian_filter(field, sigma=1.0)

        print("\n③ Rendering high-resolution abstract image...")
        hue_shift = (features.dominant_chroma / 12.0) * 0.2
        cmap = build_colormap(final_color, hue_shift=hue_shift)
        indices = (field * 255).astype(np.int32)
        rgb = cmap[indices]
        rgb_final = apply_all_effects(rgb, vibe=config.vibe, rms=features.mean_rms)

    elif config.engine == "abstractions":
        print("\n② Generating True Fronkonstin Abstractions (Physarum)...")
        # 1. Generate the magnetic disc and circular agents
        initial_map, ax, ay, angles = build_abstractions_setup(
            features=features, width=width, height=height, n_agents=150_000
        )

        # 2. Init simulation with custom exact geometry
        unique_seed = int(
            (features.bpm * features.mean_rms * 100) + features.dominant_chroma
        )
        physarum = PhysarumSimulation(
            width=width,
            height=height,
            n_agents=150_000,
            seed=unique_seed,
            initial_map=initial_map,
            ax=ax,
            ay=ay,
            angles=angles,
        )

        # 3. Exact Fronkonstin parameters (with NO diffusion)
        class DummyParams:
            sensor_distance = 6.0
            sensor_angle = 22.5 * 3.14159 / 180.0
            rotation_angle = 45.0 * 3.14159 / 180.0
            step_size = 1.0
            deposit_amount = 15.0
            decay_factor = 0.90
            diffusion_sigma = 0.0  # ZERO DIFFUSION makes the crisp mandalas!
            advection_strength = 0.0
            noise_amount = 0.0

        d_params = DummyParams()

        print("  Running geometric simulation (400 steps)...")
        # Run for many steps to let the intricate geometric mandala form
        for _ in range(400):
            physarum.step(d_params)

        # 4. Logarithmic rendering (ggplot log(v) style)
        raw_field = physarum.trail_map
        # Take the log and normalize
        log_field = np.log1p(raw_field)
        p99 = np.percentile(log_field, 99.9)
        if p99 > 0:
            field = np.clip(log_field / p99, 0.0, 1.0)
        else:
            field = np.clip(log_field / log_field.max(), 0.0, 1.0)

        print("\n③ Rendering high-resolution abstract image...")
        hue_shift = (features.dominant_chroma / 12.0) * 0.2
        cmap = build_colormap(final_color, hue_shift=hue_shift)
        indices = (field * 255).astype(np.int32)
        rgb = cmap[indices]
        rgb_final = apply_all_effects(rgb, vibe=config.vibe, rms=features.mean_rms)

    else:
        raise ValueError(
            f"Unknown engine '{config.engine}'. Choose 'field', 'veins', 'physarum', 'spheres', 'attractor', or 'lenia'."
        )

    # ------------------------------------------------------------------
    # Step 4: Write output
    # ------------------------------------------------------------------
    print(f"\n④ Writing output...")
    out_path = config.output_path

    if out_path.endswith((".gif", ".mp4")):
        out_path = out_path.rsplit(".", 1)[0] + ".png"

    img = Image.fromarray(rgb_final)
    img.save(out_path)

    elapsed = time.perf_counter() - t_start
    print(f"\n✓ Done in {elapsed:.1f}s → {out_path}\n")
    return out_path
