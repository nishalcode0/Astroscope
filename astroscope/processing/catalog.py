from __future__ import annotations

import numpy as np

from astroscope.processing.models import Source
from astroscope.processing.photometry import aperture_flux


def _measure_shape_properties(
    x: np.ndarray,
    y: np.ndarray,
    image_shape: tuple[int, int],
) -> tuple[int, int, float, float, float]:
    """Measure geometric properties of a detected source."""

    x_min = int(np.min(x))
    x_max = int(np.max(x))
    y_min = int(np.min(y))
    y_max = int(np.max(y))

    bbox_width = x_max - x_min + 1
    bbox_height = y_max - y_min + 1

    bbox_area = bbox_width * bbox_height
    pixel_count = len(x)

    compactness = (
        float(pixel_count / bbox_area)
        if bbox_area > 0
        else 0.0
    )

    major_axis = max(bbox_width, bbox_height)
    minor_axis = min(bbox_width, bbox_height)

    elongation = (
        float(major_axis / minor_axis)
        if minor_axis > 0
        else 0.0
    )

    height, width = image_shape

    distances = (
        x,
        width - 1 - x,
        y,
        height - 1 - y,
    )

    edge_distance = float(
        np.min(np.concatenate(distances))
    )

    return (
        bbox_width,
        bbox_height,
        compactness,
        elongation,
        edge_distance,
    )


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
            f"Source {source_id} has non-positive "
            "background-subtracted flux."
        )

    x_centroid = float(
        np.sum(x * corrected_values) / corrected_flux
    )

    y_centroid = float(
        np.sum(y * corrected_values) / corrected_flux
    )

    (
        bbox_width,
        bbox_height,
        compactness,
        elongation,
        edge_distance,
    ) = _measure_shape_properties(
        x,
        y,
        image.shape,
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
        bbox_width=bbox_width,
        bbox_height=bbox_height,
        compactness=compactness,
        elongation=elongation,
        edge_distance=edge_distance,
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


def measure_aperture_fluxes(
    image: np.ndarray,
    sources: list[Source],
    radius: float,
    background: float = 0.0,
) -> dict[int, float]:
    """
    Measure circular-aperture flux for every detected source.

    Parameters
    ----------
    image : np.ndarray
        2D science image.

    sources : list[Source]
        Detected and measured astronomical sources.

    radius : float
        Aperture radius in pixels.

    background : float, optional
        Background signal per pixel.

    Returns
    -------
    dict[int, float]
        Mapping from source ID to background-subtracted aperture flux.
    """
    if radius <= 0:
        raise ValueError(
            "Aperture radius must be greater than zero."
        )

    aperture_fluxes: dict[int, float] = {}

    for source in sources:
        flux = aperture_flux(
            image=image,
            x_center=source.x_centroid,
            y_center=source.y_centroid,
            radius=radius,
            background=background,
        )

        aperture_fluxes[source.source_id] = flux

    return aperture_fluxes