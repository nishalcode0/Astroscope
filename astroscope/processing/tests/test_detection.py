
from __future__ import annotations

import numpy as np
import pytest

from astroscope.processing.detection import (
    create_detection_mask,
    label_sources,
)


def test_create_detection_mask() -> None:
    """Pixels at or above the threshold are detected."""
    snr_map = np.array([
        [2.0, 5.0, 7.0],
        [1.0, 3.0, 10.0],
    ])

    mask = create_detection_mask(snr_map, threshold=5.0)

    expected = np.array([
        [False, True, True],
        [False, False, True],
    ])

    np.testing.assert_array_equal(mask, expected)


def test_create_detection_mask_default_threshold() -> None:
    """The default detection threshold is 5 sigma."""
    snr_map = np.array([
        [4.9, 5.0],
        [5.1, 10.0],
    ])

    mask = create_detection_mask(snr_map)

    expected = np.array([
        [False, True],
        [True, True],
    ])

    np.testing.assert_array_equal(mask, expected)


def test_detection_ignores_nan() -> None:
    """NaN pixels must never be marked as detections."""
    snr_map = np.array([
        [np.nan, 6.0],
        [2.0, np.nan],
    ])

    mask = create_detection_mask(snr_map, threshold=5.0)

    expected = np.array([
        [False, True],
        [False, False],
    ])

    np.testing.assert_array_equal(mask, expected)


def test_detection_ignores_infinity() -> None:
    """Infinite pixels must never be marked as detections."""
    snr_map = np.array([
        [np.inf, 6.0],
        [-np.inf, 7.0],
    ])

    mask = create_detection_mask(snr_map, threshold=5.0)

    expected = np.array([
        [False, True],
        [False, True],
    ])

    np.testing.assert_array_equal(mask, expected)


def test_detection_rejects_zero_threshold() -> None:
    """A zero threshold is invalid."""
    snr_map = np.ones((3, 3))

    with pytest.raises(ValueError):
        create_detection_mask(snr_map, threshold=0.0)


def test_detection_rejects_negative_threshold() -> None:
    """A negative threshold is invalid."""
    snr_map = np.ones((3, 3))

    with pytest.raises(ValueError):
        create_detection_mask(snr_map, threshold=-5.0)


def test_label_sources_finds_two_sources() -> None:
    """Two separated groups of detected pixels become two sources."""
    mask = np.array([
        [False, True, True, False, False],
        [False, True, True, False, False],
        [False, False, False, False, True],
        [False, False, False, False, True],
    ])

    labels, count = label_sources(mask)

    assert count == 2

    assert labels[0, 1] == labels[1, 2]
    assert labels[2, 4] == labels[3, 4]

    assert labels[0, 1] != labels[2, 4]


def test_label_sources_diagonal_pixels_are_connected() -> None:
    """Diagonal pixels belong to the same source using 8-connectivity."""
    mask = np.array([
        [True, False],
        [False, True],
    ])

    labels, count = label_sources(mask)

    assert count == 1
    assert labels[0, 0] == labels[1, 1]


def test_label_sources_separates_sources() -> None:
    """Separated detected regions receive different labels."""
    mask = np.array([
        [True, False, False, True],
        [False, False, False, False],
        [False, True, True, False],
        [False, False, False, False],
    ])

    labels, count = label_sources(mask)

    assert count == 3

    assert labels[0, 0] != labels[0, 3]
    assert labels[0, 0] != labels[2, 1]
    assert labels[0, 3] != labels[2, 1]


def test_label_sources_empty_mask() -> None:
    """An empty detection mask contains zero sources."""
    mask = np.zeros((5, 5), dtype=bool)

    labels, count = label_sources(mask)

    assert count == 0
    assert np.all(labels == 0)


def test_label_sources_rejects_non_boolean_mask() -> None:
    """Source labeling requires a boolean detection mask."""
    mask = np.array([
        [0, 1],
        [1, 0],
    ])

    with pytest.raises(ValueError):
        label_sources(mask)

