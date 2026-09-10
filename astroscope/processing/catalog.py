from __future__ import annotations

import numpy as np

from astroscope.processing.models import Source


def measure_source(
    image: np.ndarray,
    snr_map: np.ndarray,
    labels: np.ndarray,
    source_id: int,
    background: float = 0.0,
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

    corrected_values = values - background
    total_signal = float(np.sum(values))
    corrected_flux = float(np.sum(corrected_values))

    if corrected_flux <= 0:
        raise ValueError(
            f"Source {source_id} has non-positive background-subtracted flux."
        )

    x_centroid = float(
        np.sum(x * corrected_values) / corrected_flux
    )
    y_centroid = float(
        np.sum(y * corrected_values) / corrected_flux
    )

    return Source(
        source_id=source_id,
        x_centroid=x_centroid,
        y_centroid=y_centroid,
        pixel_count=int(len(values)),
        peak_signal=float(np.max(values)),
        total_signal=total_signal,
        peak_snr=float(np.max(snr_values)),
        background_subtracted_peak=float(np.max(corrected_values)),
        background_subtracted_flux=corrected_flux,
    )


def measure_sources(
    image: np.ndarray,
    snr_map: np.ndarray,
    labels: np.ndarray,
    source_count: int,
    background: float = 0.0,
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
            background=background,
        )
        sources.append(source)

    return sources