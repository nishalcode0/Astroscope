"""Visualization utilities for astronomical science images."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


class VisualizationError(Exception):
    """Raised when an astronomical image cannot be visualized."""


def normalize_for_display(
    image: np.ndarray,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.0,
) -> np.ndarray:
    """Normalize a science image for visual display.

    The original scientific data is never modified.

    Parameters
    ----------
    image:
        Two-dimensional astronomical science image.

    lower_percentile:
        Lower percentile used as the display minimum.

    upper_percentile:
        Upper percentile used as the display maximum.

    Returns
    -------
    numpy.ndarray
        Floating-point display image scaled to [0, 1].

    Raises
    ------
    VisualizationError
        If the input is invalid or contains no finite pixels.
    """
    if not isinstance(image, np.ndarray):
        raise VisualizationError(
            "Image must be a numpy.ndarray."
        )

    if image.ndim != 2:
        raise VisualizationError(
            f"Visualization requires a 2D image, got {image.ndim}D."
        )

    if not 0.0 <= lower_percentile < upper_percentile <= 100.0:
        raise VisualizationError(
            "Percentiles must satisfy "
            "0 <= lower < upper <= 100."
        )

    finite_mask = np.isfinite(image)

    if not np.any(finite_mask):
        raise VisualizationError(
            "Image contains no finite pixels."
        )

    finite_pixels = image[finite_mask]

    display_min = float(
        np.percentile(
            finite_pixels,
            lower_percentile,
        )
    )
    display_max = float(
        np.percentile(
            finite_pixels,
            upper_percentile,
        )
    )

    if not np.isfinite(display_min) or not np.isfinite(display_max):
        raise VisualizationError(
            "Display limits are not finite."
        )

    if display_max <= display_min:
        return np.zeros(
            image.shape,
            dtype=np.float32,
        )

    normalized = (
        (image.astype(np.float32) - display_min)
        / (display_max - display_min)
    )

    normalized = np.clip(
        normalized,
        0.0,
        1.0,
    )

    # Keep invalid pixels invalid for the visualization layer.
    normalized[~finite_mask] = np.nan

    return normalized


def save_science_image_png(
    image: np.ndarray,
    output_path: Path | str,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.0,
    cmap: str = "gray",
) -> Path:
    """Save an astronomical science image as a normalized PNG.

    This function creates a visualization only. It does not modify the
    scientific input array or the original FITS file.

    Parameters
    ----------
    image:
        Two-dimensional astronomical science image.

    output_path:
        Destination PNG path.

    lower_percentile:
        Lower display percentile.

    upper_percentile:
        Upper display percentile.

    cmap:
        Matplotlib colormap used for display.

    Returns
    -------
    pathlib.Path
        Path to the generated PNG.
    """
    output_path = Path(output_path)

    normalized = normalize_for_display(
        image,
        lower_percentile=lower_percentile,
        upper_percentile=upper_percentile,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure, axis = plt.subplots(
        figsize=(8, 8),
    )

    try:
        axis.imshow(
            normalized,
            origin="lower",
            cmap=cmap,
            interpolation="nearest",
        )

        axis.set_xlabel("X pixel")
        axis.set_ylabel("Y pixel")
        axis.set_title("Astroscope Science Image")

        figure.tight_layout()

        figure.savefig(
            output_path,
            dpi=150,
            bbox_inches="tight",
        )

    finally:
        plt.close(figure)

    return output_path