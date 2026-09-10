from __future__ import annotations

import numpy as np


def estimate_background(image: np.ndarray) -> float:
    """
    Estimate the background signal using the median of finite pixels.
    """
    finite_pixels = image[np.isfinite(image)]

    if finite_pixels.size == 0:
        raise ValueError("Image contains no finite pixels.")

    return float(np.median(finite_pixels))


def subtract_background(
    image: np.ndarray,
    background: float,
) -> np.ndarray:
    """
    Subtract a scalar background estimate from an image.
    """
    return image - background


def estimate_noise(image: np.ndarray) -> float:
    """
    Estimate image noise using the median absolute deviation (MAD).

    The factor 1.4826 converts MAD to an estimate of the standard
    deviation for approximately normally distributed noise.
    """
    finite_pixels = image[np.isfinite(image)]

    if finite_pixels.size == 0:
        raise ValueError("Image contains no finite pixels.")

    median = np.median(finite_pixels)
    mad = np.median(np.abs(finite_pixels - median))

    noise = 1.4826 * mad

    if noise <= 0:
        raise ValueError("Estimated image noise must be greater than zero.")

    return float(noise)


def calculate_snr(signal: float, noise: float) -> float:
    """
    Calculate signal-to-noise ratio.
    """
    if noise <= 0:
        raise ValueError("Noise must be greater than zero.")

    return float(signal / noise)


def calculate_snr_map(
    image: np.ndarray,
    background: float,
    noise: float,
) -> np.ndarray:
    """
    Convert an image into a signal-to-noise ratio map.

    Each pixel is background-subtracted and divided by the
    estimated noise level.
    """
    if noise <= 0:
        raise ValueError("Noise must be greater than zero.")

    signal = image - background
    return signal / noise


def build_snr_map(image: np.ndarray) -> np.ndarray:
    """
    Build a signal-to-noise ratio map from a science image.

    Background and noise are estimated directly from the input image.
    """
    background = estimate_background(image)
    noise = estimate_noise(image)

    return calculate_snr_map(
        image,
        background,
        noise,
    )