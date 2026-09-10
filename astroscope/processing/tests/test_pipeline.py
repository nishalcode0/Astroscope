from __future__ import annotations

import numpy as np
import pytest

from astroscope.processing.pipeline import ProcessingResult, process_image


def make_synthetic_image(
    shape: tuple[int, int] = (30, 30),
    background: float = 10.0,
    noise: float = 1.0,
) -> np.ndarray:
    """Create a synthetic astronomical image with Gaussian background noise."""
    rng = np.random.default_rng(42)

    image = rng.normal(
        loc=background,
        scale=noise,
        size=shape,
    )

    return image


def test_process_image_returns_result() -> None:
    """Verify that the processing pipeline returns a complete result."""
    image = make_synthetic_image((20, 20))

    image[10, 10] += 100.0
    image[10, 11] += 80.0
    image[11, 10] += 80.0
    image[11, 11] += 60.0

    result = process_image(
        image,
        threshold=5.0,
        min_pixels=2,
    )

    assert isinstance(result, ProcessingResult)
    assert result.snr_map.shape == image.shape
    assert result.detection_mask.shape == image.shape
    assert result.labels.shape == image.shape
    assert isinstance(result.sources, list)
    assert result.noise > 0


def test_process_image_detects_synthetic_source() -> None:
    """Verify that a strong synthetic source is detected."""
    image = make_synthetic_image((30, 30))

    image[15, 15] += 100.0
    image[15, 16] += 90.0
    image[16, 15] += 90.0
    image[16, 16] += 80.0

    result = process_image(
        image,
        threshold=5.0,
        min_pixels=3,
    )

    assert len(result.sources) >= 1

    source = max(
        result.sources,
        key=lambda item: item.peak_signal,
    )

    assert source.pixel_count >= 3
    assert source.peak_signal > 80.0
    assert source.x_centroid == pytest.approx(15.5, abs=1.0)
    assert source.y_centroid == pytest.approx(15.5, abs=1.0)


def test_process_image_uses_background_subtracted_measurements() -> None:
    """Verify that source measurements are background-subtracted."""
    image = make_synthetic_image(
        (20, 20),
        background=10.0,
        noise=1.0,
    )

    image[10, 10] += 100.0
    image[10, 11] += 90.0
    image[11, 10] += 90.0
    image[11, 11] += 80.0

    result = process_image(
        image,
        threshold=5.0,
        min_pixels=3,
    )

    assert result.background == pytest.approx(10.0, abs=0.5)

    source = max(
        result.sources,
        key=lambda item: item.background_subtracted_peak,
    )

    assert source.background_subtracted_peak > 70.0
    assert source.background_subtracted_flux > 200.0


def test_process_image_rejects_image_with_no_finite_pixels() -> None:
    """Verify that an image containing only NaN values is rejected."""
    image = np.full((10, 10), np.nan)

    with pytest.raises(ValueError, match="finite pixels"):
        process_image(image)


def test_process_image_respects_min_pixels() -> None:
    """Verify that detections smaller than min_pixels are removed."""
    image = make_synthetic_image((20, 20))

    image[5, 5] += 100.0

    result = process_image(
        image,
        threshold=5.0,
        min_pixels=3,
    )

    assert len(result.sources) == 0


def test_process_image_rejects_invalid_threshold() -> None:
    """Verify that non-positive detection thresholds are rejected."""
    image = make_synthetic_image((10, 10))

    with pytest.raises(ValueError, match="Threshold"):
        process_image(
            image,
            threshold=0.0,
        )


def test_process_image_rejects_invalid_min_pixels() -> None:
    """Verify that non-positive minimum source sizes are rejected."""
    image = make_synthetic_image((10, 10))

    with pytest.raises(ValueError, match="min_pixels"):
        process_image(
            image,
            min_pixels=0,
        )


def test_process_image_preserves_image_shape() -> None:
    """Verify that all generated maps preserve the input image dimensions."""
    image = make_synthetic_image((25, 40))

    result = process_image(
        image,
        threshold=5.0,
        min_pixels=2,
    )

    assert result.snr_map.shape == (25, 40)
    assert result.detection_mask.shape == (25, 40)
    assert result.labels.shape == (25, 40)


def test_process_image_rejects_non_2d_image() -> None:
    """Verify that only 2D astronomical images are accepted."""
    image = np.zeros((5, 5, 3), dtype=float)

    with pytest.raises(ValueError, match="2D"):
        process_image(image)


def test_process_image_rejects_non_positive_threshold() -> None:
    """Verify that negative detection thresholds are rejected."""
    image = make_synthetic_image((10, 10))

    with pytest.raises(ValueError, match="Threshold"):
        process_image(
            image,
            threshold=-5.0,
        )