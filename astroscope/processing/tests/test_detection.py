from __future__ import annotations

import numpy as np
import pytest

from astroscope.processing.detection import (
    create_detection_mask,
    filter_sources,
    label_sources,
)


def test_create_detection_mask() -> None:
    snr_map = np.array(
        [
            [2.0, 5.0, 7.0],
            [np.nan, 4.0, 10.0],
        ]
    )

    mask = create_detection_mask(snr_map, threshold=5.0)

    expected = np.array(
        [
            [False, True, True],
            [False, False, True],
        ]
    )

    np.testing.assert_array_equal(mask, expected)


def test_create_detection_mask_rejects_non_positive_threshold() -> None:
    snr_map = np.ones((3, 3))

    with pytest.raises(ValueError, match="Threshold"):
        create_detection_mask(snr_map, threshold=0.0)

    with pytest.raises(ValueError, match="Threshold"):
        create_detection_mask(snr_map, threshold=-1.0)


def test_create_detection_mask_rejects_nan_pixels() -> None:
    snr_map = np.array([[5.0, np.nan]])

    mask = create_detection_mask(snr_map, threshold=5.0)

    assert mask[0, 0]
    assert not mask[0, 1]


def test_label_sources_single_source() -> None:
    mask = np.array(
        [
            [False, True, False],
            [False, True, False],
            [False, False, False],
        ]
    )

    labels, count = label_sources(mask)

    assert count == 1
    assert labels[0, 1] == 1
    assert labels[1, 1] == 1
    assert np.count_nonzero(labels) == 2


def test_label_sources_multiple_sources() -> None:
    mask = np.array(
        [
            [True, False, False, False],
            [False, False, True, True],
            [False, False, False, False],
            [True, True, False, False],
        ]
    )

    labels, count = label_sources(mask)

    assert count == 3
    assert labels[0, 0] == 1
    assert labels[1, 2] == 2
    assert labels[1, 3] == 2
    assert labels[3, 0] == 3
    assert labels[3, 1] == 3


def test_label_sources_diagonal_pixels_are_connected() -> None:
    mask = np.array(
        [
            [True, False],
            [False, True],
        ]
    )

    labels, count = label_sources(mask)

    assert count == 1
    assert labels[0, 0] == 1
    assert labels[1, 1] == 1


def test_label_sources_background_is_zero() -> None:
    mask = np.array(
        [
            [True, False],
            [False, True],
        ]
    )

    labels, _ = label_sources(mask)

    assert labels[0, 1] == 0
    assert labels[1, 0] == 0


def test_label_sources_requires_boolean_mask() -> None:
    mask = np.array(
        [
            [1, 0],
            [0, 1],
        ],
        dtype=np.int32,
    )

    with pytest.raises(ValueError, match="boolean"):
        label_sources(mask)


def test_filter_sources_removes_small_sources() -> None:
    labels = np.array(
        [
            [1, 1, 0, 2],
            [1, 0, 0, 2],
            [0, 0, 0, 2],
        ],
        dtype=np.int32,
    )

    filtered, count = filter_sources(
        labels,
        source_count=2,
        min_pixels=3,
    )

    expected = np.array(
        [
            [1, 1, 0, 2],
            [1, 0, 0, 2],
            [0, 0, 0, 2],
        ],
        dtype=np.int32,
    )

    np.testing.assert_array_equal(filtered, expected)
    assert count == 2


def test_filter_sources_removes_one_pixel_source() -> None:
    labels = np.array(
        [
            [1, 0, 2],
            [0, 0, 0],
            [3, 3, 0],
        ],
        dtype=np.int32,
    )

    filtered, count = filter_sources(
        labels,
        source_count=3,
        min_pixels=2,
    )

    expected = np.array(
        [
            [0, 0, 0],
            [0, 0, 0],
            [1, 1, 0],
        ],
        dtype=np.int32,
    )

    np.testing.assert_array_equal(filtered, expected)
    assert count == 1


def test_filter_sources_keeps_sources_at_minimum_size() -> None:
    labels = np.array(
        [
            [1, 1, 0],
            [0, 0, 2],
            [0, 0, 2],
        ],
        dtype=np.int32,
    )

    filtered, count = filter_sources(
        labels,
        source_count=2,
        min_pixels=2,
    )

    assert count == 2
    assert filtered[0, 0] == 1
    assert filtered[0, 1] == 1
    assert filtered[1, 2] == 2
    assert filtered[2, 2] == 2


def test_filter_sources_renumbers_labels() -> None:
    labels = np.array(
        [
            [1, 1, 0, 2],
            [1, 0, 0, 2],
        ],
        dtype=np.int32,
    )

    filtered, count = filter_sources(
        labels,
        source_count=2,
        min_pixels=3,
    )

    assert count == 1
    assert np.all(filtered[labels == 1] == 1)
    assert np.all(filtered[labels == 2] == 0)


def test_filter_sources_rejects_non_positive_min_pixels() -> None:
    labels = np.zeros((3, 3), dtype=np.int32)

    with pytest.raises(ValueError, match="min_pixels"):
        filter_sources(labels, source_count=0, min_pixels=0)

    with pytest.raises(ValueError, match="min_pixels"):
        filter_sources(labels, source_count=0, min_pixels=-1)


def test_filter_sources_rejects_non_2d_labels() -> None:
    labels = np.zeros((3, 3, 1), dtype=np.int32)

    with pytest.raises(ValueError, match="2D"):
        filter_sources(labels, source_count=0)


def test_filter_sources_with_no_sources() -> None:
    labels = np.zeros((3, 3), dtype=np.int32)

    filtered, count = filter_sources(
        labels,
        source_count=0,
        min_pixels=3,
    )

    np.testing.assert_array_equal(filtered, labels)
    assert count == 0