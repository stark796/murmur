# murmur

*A song becomes a living painting.*

**murmur** takes an audio file and grows a song specific abstract image. It listens to the track's musical features (BPM, RMS, Spectral Centroid, Chroma) and translates them into a deterministic "Song DNA". This DNA then drives a variety of biological simulation engines to produce stunning, organic generative art.

The core engine is heavily inspired by the works of Antonio Sánchez Chinchón (Fronkonstin) and features a mathematically glitched Physarum slime mold simulation that creates massive, sprawling abstractions.

## Usage

```bash
pip install -e .

# Run the v2 pipeline on an audio file
murmur paint --audio examples/coffee.mp3 --output output_art.png --resolution 1024
```

## How it works

1. **Audio analysis:** We use `librosa` to extract high level features from the entire audio track, including RMS energy (loudness), spectral centroid (brightness), spectral bandwidth (texture), and dominant chroma (musical key).
2. **Song DNA:** These extracted metrics are normalized into a continuous 5 axis fingerprint (rhythm, density, brightness, complexity, harmonic). This DNA acts as a deterministic seed, ensuring that the same song always produces the exact same piece of art.
3. **Biological simulation:** The engine uses Numba JIT compilation to simulate hundreds of thousands of autonomous agents (based on the Jeff Jones Physarum model). We intentionally implement a corrupted 1D memory topology that forces the agents to interact in folded, non physical ways, yielding dense, organic, and highly textured aesthetic blobs rather than standard branching networks.
4. **Color mapping:** The song's dominant musical key automatically selects a base hue, while loudness and bandwidth dictate the lightness and saturation. The final simulation density map is mapped to this custom color palette.
5. **Rendering:** The simulation runs at a native 800x600 resolution and is then upscaled to high resolution (e.g., 1024x1024) using Lanczos resampling to create a smooth, painted look.

## Engines

*   **`physarum`**: Standard fluid, organically spreading slime mold network.
*   **`abstractions`**: The glitched, geometric, razor sharp mandala structures grown by agents in a folded memory space.
*   **`field`**: Warped mathematical color fields (Domain Warped Noise).
*   **`veins`**: Vascular, flowing particle networks.
*   **`spheres`**: Pristine geometric recursive orbits.
*   **`attractor`**: Millions of chaotic particles forming silky folds.
*   **`lenia`**: Sprawling, organic continuous cellular automata.

## Technical Highlights

*   **Numba Accelerated:** Multi agent loops are compiled to native machine code, allowing hundreds of thousands of concurrent agents to run at interactive speeds on the CPU.
*   **Deterministic Generation:** Random number generators are rigorously seeded using a hash of the Song DNA, ensuring reproducibility without relying on static presets.
*   **Topological Glitch Art:** The flagship aesthetic relies on reverse engineered memory mapping bugs from legacy C++ libraries (Armadillo), faithfully recreated in Python.
