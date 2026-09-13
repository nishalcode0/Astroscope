from __future__ import annotations

import csv

import pytest

from astroscope.processing.export import export_sources_csv
from astroscope.processing.models import Source


def make_source(
    source_id: int,
    x: float = 10.0,
    y: float = 20.0,
    aperture_flux: float = 0.0,
) -> Source:
    """Create a representative astronomical source for testing."""
    return Source(
        source_id=source_id,
        x_centroid=x,
        y_centroid=y,
        pixel_count=5,
        peak_signal=25.0,
        total_signal=100.0,
        peak_snr=6.25,
        background_subtracted_peak=24.0,
        background_subtracted_flux=95.0,
        aperture_flux=aperture_flux,
    )


def test_export_sources_csv_creates_file(tmp_path) -> None:
    """Verify that a CSV catalog file is created."""
    output_path = tmp_path / "catalog.csv"
    sources = [make_source(1)]

    export_sources_csv(sources, output_path)

    assert output_path.exists()
    assert output_path.is_file()


def test_export_sources_csv_writes_header(tmp_path) -> None:
    """Verify that the CSV contains the expected catalog columns."""
    output_path = tmp_path / "catalog.csv"

    export_sources_csv([make_source(1)], output_path)

    with output_path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:
        reader = csv.DictReader(file)

        assert reader.fieldnames == [
            "source_id",
            "x_centroid",
            "y_centroid",
            "pixel_count",
            "peak_signal",
            "total_signal",
            "peak_snr",
            "background_subtracted_peak",
            "background_subtracted_flux",
            "aperture_flux",
        ]


def test_export_sources_csv_writes_source_values(tmp_path) -> None:
    """Verify that measured source properties are exported correctly."""
    output_path = tmp_path / "catalog.csv"
    source = make_source(
        7,
        x=12.5,
        y=34.5,
        aperture_flux=1234.5,
    )

    export_sources_csv([source], output_path)

    with output_path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:
        rows = list(csv.DictReader(file))

    assert len(rows) == 1

    row = rows[0]

    assert row["source_id"] == "7"
    assert float(row["x_centroid"]) == pytest.approx(12.5)
    assert float(row["y_centroid"]) == pytest.approx(34.5)
    assert row["pixel_count"] == "5"
    assert float(row["peak_signal"]) == pytest.approx(25.0)
    assert float(row["total_signal"]) == pytest.approx(100.0)
    assert float(row["peak_snr"]) == pytest.approx(6.25)
    assert float(row["background_subtracted_peak"]) == pytest.approx(24.0)
    assert float(row["background_subtracted_flux"]) == pytest.approx(95.0)
    assert float(row["aperture_flux"]) == pytest.approx(1234.5)


def test_export_sources_csv_writes_multiple_sources(tmp_path) -> None:
    """Verify that multiple detected sources are exported."""
    output_path = tmp_path / "catalog.csv"

    sources = [
        make_source(1, x=10.0, y=20.0, aperture_flux=100.0),
        make_source(2, x=30.0, y=40.0, aperture_flux=200.0),
        make_source(3, x=50.0, y=60.0, aperture_flux=300.0),
    ]

    export_sources_csv(sources, output_path)

    with output_path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:
        rows = list(csv.DictReader(file))

    assert len(rows) == 3
    assert [row["source_id"] for row in rows] == ["1", "2", "3"]
    assert float(rows[0]["aperture_flux"]) == pytest.approx(100.0)
    assert float(rows[1]["aperture_flux"]) == pytest.approx(200.0)
    assert float(rows[2]["aperture_flux"]) == pytest.approx(300.0)


def test_export_sources_csv_supports_empty_catalog(tmp_path) -> None:
    """Verify that an empty source list still produces a valid CSV."""
    output_path = tmp_path / "empty_catalog.csv"

    export_sources_csv([], output_path)

    assert output_path.exists()

    with output_path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:
        reader = csv.DictReader(file)
        rows = list(reader)

        assert reader.fieldnames == [
            "source_id",
            "x_centroid",
            "y_centroid",
            "pixel_count",
            "peak_signal",
            "total_signal",
            "peak_snr",
            "background_subtracted_peak",
            "background_subtracted_flux",
            "aperture_flux",
        ]

        assert rows == []


def test_export_sources_csv_creates_parent_directory(tmp_path) -> None:
    """Verify that missing parent directories are created."""
    output_path = tmp_path / "results" / "catalog.csv"

    export_sources_csv([make_source(1)], output_path)

    assert output_path.exists()
    assert output_path.parent.exists()