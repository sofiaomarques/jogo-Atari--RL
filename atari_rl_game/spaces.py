"""Tiny fallbacks for Gymnasium spaces.

The project is meant to train with Gymnasium, but these classes keep the core
game runnable in a fresh Python environment where only NumPy is installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


class Discrete:
    def __init__(self, n: int) -> None:
        if n <= 0:
            raise ValueError("n must be positive")
        self.n = int(n)
        self.shape = ()

    def sample(self) -> int:
        return int(np.random.randint(self.n))

    def contains(self, x: Any) -> bool:
        try:
            value = int(x)
        except (TypeError, ValueError):
            return False
        return 0 <= value < self.n


@dataclass(frozen=True)
class Box:
    low: float | int
    high: float | int
    shape: tuple[int, ...]
    dtype: type | np.dtype

    def sample(self) -> np.ndarray:
        dtype = np.dtype(self.dtype)
        if np.issubdtype(dtype, np.integer):
            return np.random.randint(self.low, self.high + 1, self.shape, dtype=dtype)
        return np.random.uniform(self.low, self.high, self.shape).astype(dtype)

    def contains(self, x: Any) -> bool:
        arr = np.asarray(x)
        return arr.shape == self.shape and arr.dtype == np.dtype(self.dtype)
