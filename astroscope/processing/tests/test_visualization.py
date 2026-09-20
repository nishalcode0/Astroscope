"""Tests for astronomical image visualization."""

from pathlib import Path

import numpy as np
import pytest

from astroscope.processing.visualization import (
    VisualizationError,
    normalize_for_display,
    save_science_image_png,
)


def test_normalize_for_display_returns_unit_range():
    image = np.arange(
        100,
        dtype=np.float32,
    ).reshape(10, 10)

    normalized = normalize_for_display(
        image,
        lower_percentile=0.0,
        upper_percentile=100.0,
    )

    assert normalized.shape == image.shape
    assert normalized.dtype == np.float32
    assert np.nanmin(normalized) == pytest.approx(0.0)
    assert np.nanmax(normalized) == pytest.approx(1.0)


def test_normalize_does_not_modify_input():
    image = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ],
        dtype=np.float32,
    )

    original = image.copy()

    normalize_for_display(
        image,
        lower_percentile=0.0,
        upper_percentile=100.0,
    )

    np.testing.assert_array_equal(
        image,
        original,
    )


def test_normalize_preserves_invalid_pixels():
    image = np.array(
        [
            [1.0, 2.0],
            [np.nan, 4.0],
        ],
        dtype=np.float32,
    )

    normalized = normalize_for_display(
        image,
        lower_percentile=0.0,
        upper_percentile=100.0,
    )

    assert np.isnan(normalized[1, 0])


def test_normalize_rejects_non_2d_image():
    image = np.ones(
        (2, 2, 2),
        dtype=np.float32,
    )

    with pytest.raises(VisualizationError):
        normalize_for_display(image)


def test_normalize_rejects_all_invalid_image():
    image = np.full(
        (10, 10),
        np.nan,
        dtype=np.float32,
    )

    with pytest.raises(VisualizationError):
        normalize_for_display(image)


def test_normalize_rejects_invalid_percentiles():
    image = np.ones(
        (10, 10),
        dtype=np.float32,
    )

    with pytest.raises(VisualizationError):
        normalize_for_display(
            image,
            lower_percentile=99.0,
            upper_percentile=1.0,
        )


def test_save_science_image_png(tmp_path: Path):
    image = np.arange(
        100,
        dtype=np.float32,
    ).reshape(10, 10)

    output_path = (
        tmp_path / "science_image.png"
    )

    result = save_science_image_png(
        image,
        output_path,
        lower_percentile=0.0,
        upper_percentile=100.0,
    )

    assert result == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0