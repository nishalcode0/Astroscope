"""Test command-line interface behavior."""

import pytest
from unittest.mock import patch, MagicMock

from astroscope.cli.main import main
from astroscope.observation import Observation
from astroscope.product import Product


def test_cli_help(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify CLI --help option outputs header and exits cleanly."""
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])

    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "Astroscope" in captured.out
    assert "Open-source computational astronomical observatory" in captured.out


def test_cli_subcommands(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify CLI subcommands output clear not-implemented messaging."""
    subcommands = ["discover", "analyze", "candidates"]
    for cmd in subcommands:
        exit_code = main([cmd])
        assert exit_code == 0
        captured = capsys.readouterr()
        assert f"Command '{cmd}' is not implemented yet in Session 1." in captured.out


def test_cli_ingest_missing_args(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify ingest command requires --obs-id."""
    with pytest.raises(SystemExit) as exc_info:
        main(["ingest"])
    assert exc_info.value.code == 2


def test_cli_ingest_missing_product_id(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify missing --product-id produces a clear error."""
    exit_code = main(["ingest", "--obs-id", "n4eya1020"])
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "Error: --product-id is required for ingestion." in captured.out


@patch("astroscope.cli.main.HubbleAdapter")
def test_cli_ingest_product_not_found(mock_adapter_class, capsys: pytest.CaptureFixture[str]) -> None:
    """Verify error when the requested product does not exist."""
    mock_adapter = MagicMock()
    mock_adapter_class.return_value = mock_adapter

    mock_obs = Observation(mission="HST", observation_id="n4eya1020", target_name="M31", right_ascension=0.0, declination=0.0, instrument="NICMOS", filter_band="F160W", mast_obsid=123)
    mock_adapter.get_observation_metadata.return_value = mock_obs
    mock_adapter.get_products.return_value = []

    exit_code = main(["ingest", "--obs-id", "n4eya1020", "--product-id", "invalid.fits"])
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "Error: Product invalid.fits not found in observation n4eya1020." in captured.out


@patch("astroscope.cli.main.HubbleAdapter")
@patch("astroscope.cli.main.ingest_product")
def test_cli_ingest_success(mock_ingest, mock_adapter_class, capsys: pytest.CaptureFixture[str]) -> None:
    """Verify exactly one product is ingested successfully."""
    mock_adapter = MagicMock()
    mock_adapter_class.return_value = mock_adapter

    mock_obs = Observation(mission="HST", observation_id="n4eya1020", target_name="M31", right_ascension=0.0, declination=0.0, instrument="NICMOS", filter_band="F160W", mast_obsid=123)
    mock_product = Product(product_id="test.fits", parent_obsid=123, collection="HST", product_type="SCIENCE", data_product_type="image", calib_level=3, filters="F160W", filename="test.fits", data_uri="uri")

    mock_adapter.get_observation_metadata.return_value = mock_obs
    mock_adapter.get_products.return_value = [mock_product]

    mock_ingest.return_value = "/tmp/fake/path.fits"

    exit_code = main(["ingest", "--obs-id", "n4eya1020", "--product-id", "test.fits"])

    assert exit_code == 0
    mock_ingest.assert_called_once_with(mock_product, mock_obs)
    captured = capsys.readouterr()
    assert "Saved to /tmp/fake/path.fits" in captured.out


def test_cli_process_missing_input(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify process command requires --input."""
    with pytest.raises(SystemExit) as exc_info:
        main(["process"])
    assert exc_info.value.code == 2


@patch("astroscope.cli.main.compute_statistics")
@patch("astroscope.cli.main.load_science_image")
def test_cli_process_success(mock_load, mock_compute, capsys: pytest.CaptureFixture[str]) -> None:
    """Verify successful processing outputs correct stats."""
    mock_image = MagicMock()
    mock_image.header = {'TELESCOP': 'HST', 'INSTRUME': 'NICMOS', 'FILTER': 'F160W'}
    mock_load.return_value = mock_image

    mock_stats = MagicMock()
    mock_stats.shape = (10, 10)
    mock_stats.finite_pixels = 100
    mock_stats.nan_pixels = 0
    mock_stats.min_val = 1.0
    mock_stats.max_val = 10.0
    mock_stats.mean_val = 5.0
    mock_stats.median_val = 5.0
    mock_stats.std_val = 2.0
    mock_compute.return_value = mock_stats

    exit_code = main(["process", "--input", "fake.fits"])

    assert exit_code == 0
    mock_load.assert_called_once()
    mock_compute.assert_called_once_with(mock_image)

    captured = capsys.readouterr()
    assert "Loading science image from fake.fits..." in captured.out
    assert "Mission: HST" in captured.out
    assert "Instrument: NICMOS" in captured.out
    assert "Filter: F160W" in captured.out
    assert "Shape: (10, 10)" in captured.out
    assert "Finite pixels: 100" in captured.out


@patch("astroscope.cli.main.load_science_image")
def test_cli_process_error(mock_load, capsys: pytest.CaptureFixture[str]) -> None:
    """Verify processing errors are handled cleanly."""
    from astroscope.processing.core import ProcessingError
    mock_load.side_effect = ProcessingError("Invalid file")

    exit_code = main(["process", "--input", "fake.fits"])

    assert exit_code == 1
    captured = capsys.readouterr()
    assert "Error: Invalid file" in captured.out
