"""Tests for the ingestion layer."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from astroscope.observation import Observation
from astroscope.product import Product
from astroscope.ingestion.core import ingest_product, get_local_path, IngestionError
from astroscope.archive.mast import MastQueryError

@pytest.fixture
def mock_observation():
    return Observation(
        mission="HST",
        observation_id="n4eya1020",
        target_name="M31",
        right_ascension=10.0,
        declination=41.0,
        instrument="NICMOS",
        filter_band="F160W",
        mast_obsid=12345
    )

@pytest.fixture
def mock_product():
    return Product(
        product_id="n4eya1020_mos.fits",
        parent_obsid=12345,
        collection="HST",
        product_type="SCIENCE",
        data_product_type="image",
        calib_level=3,
        filters="F160W",
        filename="n4eya1020_mos.fits",
        data_uri="mast:HST/product/n4eya1020_mos.fits",
        file_size=1024,
    )

def test_get_local_path():
    path = get_local_path("/tmp/data", "HST", "n4eya1020", "file.fits")
    assert path == Path("/tmp/data/raw/HST/n4eya1020/file.fits")

@patch("astroscope.ingestion.core.download_file")
def test_ingest_success(mock_download, mock_observation, mock_product, tmp_path):
    # Simulate a successful download by creating the expected .part file
    def fake_download(uri, local_path):
        Path(local_path).write_text("fake data")
        
    mock_download.side_effect = fake_download
    
    result = ingest_product(mock_product, mock_observation, base_dir=str(tmp_path))
    
    expected_path = tmp_path / "raw" / "HST" / "n4eya1020" / "n4eya1020_mos.fits"
    assert result == expected_path
    assert expected_path.exists()
    assert expected_path.read_text() == "fake data"
    mock_download.assert_called_once_with(mock_product.data_uri, str(expected_path.with_suffix(".fits.part")))

@patch("astroscope.ingestion.core.download_file")
def test_ingest_duplicate_skips_download(mock_download, mock_observation, mock_product, tmp_path):
    expected_path = tmp_path / "raw" / "HST" / "n4eya1020" / "n4eya1020_mos.fits"
    expected_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create the file in advance with the correct size (1024 bytes)
    expected_path.write_bytes(b"0" * 1024)
    
    result = ingest_product(mock_product, mock_observation, base_dir=str(tmp_path))
    
    assert result == expected_path
    mock_download.assert_not_called()

@patch("astroscope.ingestion.core.download_file")
def test_ingest_duplicate_size_mismatch_redownloads(mock_download, mock_observation, mock_product, tmp_path):
    expected_path = tmp_path / "raw" / "HST" / "n4eya1020" / "n4eya1020_mos.fits"
    expected_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create the file in advance with incorrect size (500 bytes vs expected 1024)
    expected_path.write_bytes(b"0" * 500)
    
    def fake_download(uri, local_path):
        Path(local_path).write_bytes(b"0" * 1024)
        
    mock_download.side_effect = fake_download
    
    result = ingest_product(mock_product, mock_observation, base_dir=str(tmp_path))
    
    assert result == expected_path
    mock_download.assert_called_once()
    assert expected_path.stat().st_size == 1024

@patch("astroscope.ingestion.core.download_file")
def test_ingest_failed_download_cleans_up(mock_download, mock_observation, mock_product, tmp_path):
    # Simulate a failed download that creates a partial file but then crashes
    def fake_fail(uri, local_path):
        Path(local_path).write_text("partial data")
        raise MastQueryError("Network failure")
        
    mock_download.side_effect = fake_fail
    
    with pytest.raises(IngestionError, match="Failed to download"):
        ingest_product(mock_product, mock_observation, base_dir=str(tmp_path))
        
    expected_path = tmp_path / "raw" / "HST" / "n4eya1020" / "n4eya1020_mos.fits"
    part_path = expected_path.with_suffix(".fits.part")
    
    assert not expected_path.exists()
    assert not part_path.exists()

def test_ingest_missing_uri(mock_observation):
    product = Product(
        product_id="test",
        parent_obsid=1,
        collection="HST",
        product_type="SCIENCE",
        data_product_type="image",
        calib_level=1,
        filters="F",
        filename="test.fits",
        data_uri=""  # Empty URI
    )
    
    with pytest.raises(IngestionError, match="has no data URI"):
        ingest_product(product, mock_observation, base_dir="/tmp/data")
