"""Domain models for scientifically processed astronomical data."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np


@dataclass(frozen=True)
class ImageStatistics:
    """Scientifically meaningful basic statistics computed over valid pixels.

    Computed excluding NaN or explicitly masked invalid pixels to ensure
    mathematical soundness.
    """
    shape: Tuple[int, ...]
    finite_pixels: int
    nan_pixels: int
    min_val: float
    max_val: float
    mean_val: float
    median_val: float
    std_val: float


@dataclass
class ScienceImage:
    """In-memory representation of an astronomical science image.

    Holds the primary scientific pixel data (SCI extension) along with
    essential metadata to maintain scientific provenance without modifying
    the original raw file.
    """
    data: np.ndarray
    filepath: Path
    header: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the scientific data payload."""
        if not isinstance(self.data, np.ndarray):
            raise TypeError("ScienceImage data must be a numpy.ndarray.")
        if self.data.ndim != 2:
            raise ValueError(f"ScienceImage requires 2D data, got {self.data.ndim}D.")
        if not np.issubdtype(self.data.dtype, np.floating):
            # Enforce floating point data for scientific accuracy (NaN handling)
            self.data = self.data.astype(np.float32)

@dataclass(frozen=True)
class Source:
    """Measured properties of a detected astronomical source."""

    source_id: int
    x_centroid: float
    y_centroid: float
    pixel_count: int
    peak_signal: float
    total_signal: float
    peak_snr: float
    background_subtracted_peak: float
    background_subtracted_flux: float