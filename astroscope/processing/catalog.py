from __future__ import annotations

import numpy as np

from astroscope.processing.models import Source


def measure_source(
    image: np.ndarray,
    snr_map: np.ndarray,
    labels: np.ndarray,
    source_id: int,
) -> Source:
    """
    Measure the properties of one detected astronomical source.
    """
    pixels = labels == source_id

    if not np.any(pixels):
        raise ValueError(f"Source {source_id} does not exist.")

    y, x = np.where(pixels)

    values = image[pixels]
    snr_values = snr_map[pixels]

    finite = np.isfinite(values) & np.isfinite(snr_values)

    if not np.any(finite):
        raise ValueError(
            f"Source {source_id} contains no finite pixels."
        )

    x = x[finite]
    y = y[finite]
    values = values[finite]
    snr_values = snr_values[finite]

    return Source(
        source_id=source_id,
        x_centroid=float(np.mean(x)),
        y_centroid=float(np.mean(y)),
        pixel_count=int(len(values)),
        peak_signal=float(np.max(values)),
        total_signal=float(np.sum(values)),
        peak_snr=float(np.max(snr_values)),
    )


def measure_sources(
    image: np.ndarray,
    snr_map: np.ndarray,
    labels: np.ndarray,
    source_count: int,
) -> list[Source]:
    """
    Measure all detected astronomical sources.
    """
    sources = []

    for source_id in range(1, source_count + 1):
        source = measure_source(
            image,
            snr_map,
            labels,
            source_id,
        )
        sources.append(source)

    return sources