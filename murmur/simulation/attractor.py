import numpy as np
from numba import njit
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from murmur.core.config import SimParams


@njit(fastmath=True)
def _compute_clifford(
    a: float, b: float, c: float, d: float, n_iters: int, width: int, height: int
):
    # Histogram array
    grid = np.zeros((height, width), dtype=np.float32)

    x, y = 0.0, 0.0

    # Scale coordinates to fit nicely in the window.
    # Clifford bounds are generally [- (1 + |c|), 1 + |c|]
    bound_x = 1.0 + abs(c)
    bound_y = 1.0 + abs(d)
    bounds = max(bound_x, bound_y) * 1.2  # 20% margin

    # To prevent blowing up at the very first few points (transient), we could skip them,
    # but for a millions-long iteration, it doesn't matter.
    for _ in range(n_iters):
        x_new = np.sin(a * y) + c * np.cos(a * x)
        y_new = np.sin(b * x) + d * np.cos(b * y)
        x, y = x_new, y_new

        # Map to pixel coordinates
        px = int(((x / bounds) + 1.0) * 0.5 * width)
        py = int(((y / bounds) + 1.0) * 0.5 * height)

        if 0 <= px < width and 0 <= py < height:
            grid[py, px] += 1.0

    return grid


class AttractorSimulation:
    """
    Chaotic Strange Attractor engine (Clifford Attractor).
    Uses Numba to quickly plot millions of points into a density map.
    Since it calculates the entire structure instantly based on parameters,
    it is stateless across frames.
    """

    def __init__(self, width: int, height: int, n_iters: int = 5_000_000):
        self.W = width
        self.H = height
        self.n_iters = n_iters
        self.grid = np.zeros((height, width), dtype=np.float32)

        # Compile Numba function on init (warmup)
        _compute_clifford(1.0, 1.0, 1.0, 1.0, 100, width, height)

    def step(self, params: "SimParams"):
        """
        Compute the attractor for the current frame's parameters.
        """
        raw_grid = _compute_clifford(
            a=params.a,
            b=params.b,
            c=params.c,
            d=params.d,
            n_iters=self.n_iters,
            width=self.W,
            height=self.H,
        )

        # Normalize the grid so it maps perfectly into our [0, 1] colormap pipeline.
        # We use a log scale so faint folds are highly visible (very important for attractors)
        grid = np.log1p(raw_grid)
        max_val = grid.max()

        if max_val > 0:
            # We scale it so the 99th percentile hits 1.0, letting the densest cores bloom
            p99 = np.percentile(grid, 99.9)
            if p99 > 0:
                grid = np.clip(grid / p99, 0.0, 1.0)
            else:
                grid = np.clip(grid / max_val, 0.0, 1.0)

        self.grid = grid

    @property
    def trail(self) -> np.ndarray:
        """Alias for compatibility with the rendering pipeline."""
        return self.grid
