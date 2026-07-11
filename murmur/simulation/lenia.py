import numpy as np
from scipy.fft import fft2, ifft2
from dataclasses import dataclass


@dataclass
class LeniaParams:
    """Parameters for one Lenia update."""

    mu: float
    sigma: float
    dt: float


class LeniaSimulation:
    """
    Continuous Cellular Automata (Lenia) engine.
    Uses FFT-based convolutions to compute spatial neighborhoods.
    """

    def __init__(self, width: int, height: int, R: float = 13.0, seed: int = 42):
        self.W = width
        self.H = height
        self.R = R

        # Initialize grid with a dense blob of noise in the center
        self.A = np.zeros((height, width), dtype=np.float32)
        rng = np.random.default_rng(seed)

        cx, cy = width // 2, height // 2
        radius = min(width, height) // 4
        y, x = np.ogrid[-cy : height - cy, -cx : width - cx]
        mask = x**2 + y**2 <= radius**2
        self.A[mask] = rng.random(np.sum(mask)).astype(np.float32)

        # Build static convolution kernel
        self._build_kernel(R)

        # Pre-warm to let it form creatures instead of raw noise
        self._prewarm()

    def _build_kernel(self, R: float):
        """Build a periodic Gaussian ring kernel and precompute its FFT."""
        omega = R / 4.0

        y = np.arange(self.H)
        x = np.arange(self.W)
        yy, xx = np.meshgrid(y, x, indexing="ij")

        # Center coordinates for periodic convolution
        dy = np.minimum(yy, self.H - yy)
        dx = np.minimum(xx, self.W - xx)
        dist = np.sqrt(dx**2 + dy**2)

        # Gaussian ring
        K = np.exp(-((dist - R) ** 2) / (2 * omega**2))

        # Normalize
        K_sum = K.sum()
        if K_sum > 0:
            K /= K_sum

        # Compute FFT and store
        self.K_fft = fft2(K).astype(np.complex64)

    def _prewarm(self, steps: int = 50):
        """Run standard Orbium-like parameters to let creatures form."""
        params = LeniaParams(mu=0.15, sigma=0.017, dt=0.1)
        for _ in range(steps):
            self.step(params)

    def step(self, params: LeniaParams):
        """
        Compute one step of Lenia evolution.
        U = K * A
        G = 2 * exp(-(U - mu)^2 / (2 * sigma^2)) - 1
        A = clip(A + dt * G, 0, 1)
        """
        # 1. Convolution via FFT
        A_fft = fft2(self.A)
        # We need the real part; small imaginary parts are floating point errors
        U = np.real(ifft2(self.K_fft * A_fft))

        # 2. Growth function
        G = 2.0 * np.exp(-((U - params.mu) ** 2) / (2 * params.sigma**2)) - 1.0

        # 3. Update field
        self.A = np.clip(self.A + params.dt * G, 0.0, 1.0)

    @property
    def trail(self) -> np.ndarray:
        """Alias for compatibility with the rendering pipeline."""
        return self.A
