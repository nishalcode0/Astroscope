
from pathlib import Path

import numpy as np

from astroscope.processing.models import Source
from astroscope.processing.visualization import (
    save_detection_overlay,
)


def make_source(
    source_id: int,
    x: float,
    y: float,
) -> Source:
    return Source(
        source_id=source_id,
        x_centroid=x,
        y_centroid=y,
        pixel_count=5,
        peak_signal=20.0,
        total_signal=50.0,
        peak_snr=10.0,
        background_subtracted_peak=18.0,
        background_subtracted_flux=45.0,
    )


def test_detection_overlay_creates_png(tmp_path: Path):
    image = np.zeros((50, 50), dtype=np.float32)

    sources = [
        make_source(1, 10.5, 20.5),
        make_source(2, 30.0, 40.0),
    ]

    output = tmp_path / "detections.png"

    result = save_detection_overlay(
        image,
        sources,
        output,
    )

    assert result == output
    assert output.exists()
    assert output.stat().st_size > 0


def test_detection_overlay_accepts_empty_source_list(
    tmp_path: Path,
):
    image = np.ones((30, 30), dtype=np.float32)

    output = tmp_path / "empty.png"

    result = save_detection_overlay(
        image,
        [],
        output,
    )

    assert result == output
    assert output.exists()


def test_detection_overlay_does_not_modify_image(
    tmp_path: Path,
):
    image = np.random.default_rng(42).normal(
        size=(40, 40),
    ).astype(np.float32)

    original = image.copy()

    sources = [
        make_source(1, 15.0, 25.0),
    ]

    save_detection_overlay(
        image,
        sources,
        tmp_path / "overlay.png",
    )

    np.testing.assert_array_equal(
        image,
        original,
    )

