from __future__ import annotations

import numpy as np


def aperture_mask(
    image_shape: tuple[int, int],
    x_center: float,
    y_center: float,
    radius: float,
) -> np.ndarray:
    """
    Create a circular aperture mask centered on a source.

    Parameters
    ----------
    image_shape : tuple[int, int]
        Shape of the 2D science image.

    x_center : float
        Source x-coordinate in pixel coordinates.

    y_center : float
        Source y-coordinate in pixel coordinates.

    radius : float
        Aperture radius in pixels.

    Returns
    -------
    np.ndarray
        Boolean mask selecting pixels inside the aperture.
    """
    if len(image_shape) != 2:
        raise ValueError("Image shape must be 2D.")

    if radius <= 0:
        raise ValueError("Aperture radius must be greater than zero.")

    height, width = image_shape

    y, x = np.ogrid[:height, :width]

    distance_squared = (
        (x - x_center) ** 2
        + (y - y_center) ** 2
    )

    return distance_squared <= radius**2


def aperture_flux(
    image: np.ndarray,
    x_center: float,
    y_center: float,
    radius: float,
    background: float = 0.0,
) -> float:
    """
    Calculate background-subtracted flux inside a circular aperture.

    Parameters
    ----------
    image : np.ndarray
        2D science image.

    x_center : float
        Source x-coordinate in pixel coordinates.

    y_center : float
        Source y-coordinate in pixel coordinates.

    radius : float
        Aperture radius in pixels.

    background : float, optional
        Background signal per pixel.

    Returns
    -------
    float
        Background-subtracted aperture flux.
    """
    if image.ndim != 2:
        raise ValueError(
            f"Aperture photometry requires a 2D image, got {image.ndim}D."
        )

    mask = aperture_mask(
        image.shape,
        x_center,
        y_center,
        radius,
    )

    values = image[mask]

    finite = np.isfinite(values)

    if not np.any(finite):
        raise ValueError(
            "Aperture contains no finite pixels."
        )

    corrected_values = values[finite] - background

    return float(np.sum(corrected_values))


def aperture_pixel_count(
    image_shape: tuple[int, int],
    x_center: float,
    y_center: float,
    radius: float,
) -> int:
    """
    Count the number of pixels inside a circular aperture.
    """
    mask = aperture_mask(
        image_shape,
        x_center,
        y_center,
        radius,
    )

    return int(np.sum(mask))