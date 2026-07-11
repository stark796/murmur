"""
CLI entry point for murmur.

Usage (legacy):
    murmur generate --audio ribs.mp3 --color "#4a1942" --vibe 0.6

Usage (v2 — no flags needed):
    murmur paint --audio ribs.mp3 --output ribs_mandala.png
"""

from __future__ import annotations

import click

from murmur.core.config import MurmurConfig, resolve_vibe, VIBE_PRESETS
from murmur.core.pipeline import generate


@click.group()
def main():
    """murmur — grow organic, abstract art from a song."""
    pass


# ---------------------------------------------------------------------------
# Legacy command: murmur generate (untouched)
# ---------------------------------------------------------------------------


@main.command()
@click.option(
    "--audio",
    "-a",
    required=True,
    type=click.Path(exists=True, readable=True),
    help="Path to audio file (.mp3 or .wav)",
)
@click.option(
    "--color",
    "-c",
    required=False,
    help="Base hex color, e.g. '#4a1942'",
)
@click.option(
    "--engine",
    "-e",
    default="field",
    show_default=True,
    type=click.Choice(
        ["field", "veins", "physarum", "spheres", "attractor", "lenia", "abstractions"],
        case_sensitive=False,
    ),
    help="Visual engine to use.",
)
@click.option(
    "--vibe",
    "-v",
    default="0.5",
    show_default=True,
    help=(
        "Vibe: float 0.0–1.0 or named preset. "
        f"Presets: {', '.join(VIBE_PRESETS.keys())}"
    ),
)
@click.option(
    "--start",
    "-s",
    default=0.0,
    show_default=True,
    type=float,
    help="Start time in seconds",
)
@click.option(
    "--duration",
    "-d",
    default=10.0,
    show_default=True,
    type=float,
    help="Duration of the output in seconds",
)
@click.option(
    "--fps",
    default=24,
    show_default=True,
    type=int,
    help="Frames per second",
)
@click.option(
    "--resolution",
    "-r",
    default=512,
    show_default=True,
    type=int,
    help="Output resolution (square, in pixels)",
)
@click.option(
    "--output",
    "-o",
    default="output.gif",
    show_default=True,
    help="Output file path (.gif or .mp4)",
)
def generate_cmd(audio, color, engine, vibe, start, duration, fps, resolution, output):
    """Generate abstract song art from an audio file (legacy engines)."""
    # Resolve vibe (named preset or float)
    try:
        vibe_float = resolve_vibe(vibe)
    except ValueError as e:
        raise click.BadParameter(str(e), param_hint="'--vibe'")

    config = MurmurConfig(
        audio_path=audio,
        color=color,
        engine=engine.lower(),
        vibe=vibe_float,
        start_time=start,
        duration=duration,
        fps=fps,
        resolution=resolution,
        output_path=output,
    )

    generate(config)


main.add_command(generate_cmd, name="generate")


# ---------------------------------------------------------------------------
# v2 command: murmur paint — no flags, song decides everything
# ---------------------------------------------------------------------------


@main.command()
@click.option(
    "--audio",
    "-a",
    required=True,
    type=click.Path(exists=True, readable=True),
    help="Path to audio file (.mp3 or .wav)",
)
@click.option(
    "--output",
    "-o",
    default="mandala.png",
    show_default=True,
    help="Output PNG path",
)
@click.option(
    "--resolution",
    "-r",
    default=1024,
    show_default=True,
    type=int,
    help="Final output resolution in pixels (square). e.g. 1024, 4096, 9192",
)
@click.option(
    "--sim-resolution",
    "-s",
    default=None,
    type=int,
    help=(
        "Simulation canvas size (default: min(resolution, 2048)). "
        "Set higher for more filament detail — e.g. --sim-resolution 4096. "
        "Time scales with sim_resolution²."
    ),
)
def paint(audio, output, resolution, sim_resolution):
    """
    Paint a song-specific abstract mandala. No flags needed — the song decides everything.

    \b
    Examples:
      murmur paint --audio ribs.mp3
      murmur paint --audio ribs.mp3 --output ribs_mandala.png --resolution 4096
      murmur paint --audio song.mp3 --resolution 9192 --sim-resolution 4096
    """
    from murmur.core.pipeline_v2 import paint as paint_fn

    paint_fn(
        audio_path=audio,
        output_path=output,
        resolution=resolution,
        sim_resolution=sim_resolution,
    )


if __name__ == "__main__":
    main()
