from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from astroscope.processing.algorithms import (
    build_snr_map,
    estimate_background,
    estimate_noise,
)
from astroscope.processing.catalog import (
    measure_aperture_fluxes,
    measure_sources,
)
from astroscope.processing.detection import (
    create_detection_mask,
    filter_sources,
    label_sources,
)
from astroscope.processing.models import Source


@dataclass(frozen=True)
class ProcessingResult:
    """Complete result of processing a science image."""

    background: float
    noise: float
    snr_map: np.ndarray
    detection_mask: np.ndarray
    labels: np.ndarray
    sources: list[Source]


def process_image(
    image: np.ndarray,
    threshold: float = 5.0,
    min_pixels: int = 3,
    aperture_radius: float | None = None,
) -> ProcessingResult:
    """
    Run the complete source-detection pipeline on a science image.

    Pipeline:

        image
        -> background/noise estimation
        -> SNR map
        -> detection mask
        -> connected-component labeling
        -> source-size filtering
        -> source measurement
        -> optional aperture photometry
    """
    if image.ndim != 2:
        raise ValueError(
            f"Processing pipeline requires a 2D image, got {image.ndim}D."
        )

    background = estimate_background(image)
    noise = estimate_noise(image)

    snr_map = build_snr_map(image)

    detection_mask = create_detection_mask(
        snr_map,
        threshold=threshold,
    )

    labels, source_count = label_sources(detection_mask)

    labels, filtered_count = filter_sources(
        labels,
        source_count,
        min_pixels=min_pixels,
    )

    sources = measure_sources(
        image,
        snr_map,
        labels,
        filtered_count,
        background=background,
    )

    if aperture_radius is not None:
        aperture_fluxes = measure_aperture_fluxes(
            image,
            sources,
            radius=aperture_radius,
            background=background,
        )

        sources = [
            replace(
                source,
                aperture_flux=aperture_fluxes[source.source_id],
            )
            for source in sources
        ]

    return ProcessingResult(
        background=background,
        noise=noise,
        snr_map=snr_map,
        detection_mask=detection_mask,
        labels=labels,
        sources=sources,
    )