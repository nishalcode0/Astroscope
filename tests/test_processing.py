"""Tests for the Astroscope scientific processing layer."""

import numpy as np
import pytest
from astropy.io import fits

from astroscope.processing.core import load_science_image, compute_statistics, ProcessingError
from astroscope.processing.models import ScienceImage

def create_synthetic_fits(path, include_sci=True, data_shape=(10, 10), data_values=None):
    """Helper to create a synthetic FITS file for testing."""
    primary_hdu = fits.PrimaryHDU()
    primary_hdu.header['TELESCOP'] = 'SYNTHETIC'
    primary_hdu.header['INSTRUME'] = 'TEST_INST'
    primary_hdu.header['FILTER'] = 'TEST_FILT'

    hdul = fits.HDUList([primary_hdu])

    if data_values is None:
        data = np.ones(data_shape, dtype=np.float32)
    else:
        data = np.array(data_values, dtype=np.float32)

    if include_sci:
        sci_hdu = fits.ImageHDU(data=data, name='SCI')
        sci_hdu.header['BUNIT'] = 'ELECTRONS/S'
        hdul.append(sci_hdu)
    else:
        # Put data in primary
        hdul[0].data = data

    hdul.writeto(path, overwrite=True)

def test_load_science_image_with_sci(tmp_path):
    """Test loading a FITS file with a designated SCI extension."""
    fits_path = tmp_path / "test_sci.fits"
    create_synthetic_fits(fits_path, include_sci=True)

    image = load_science_image(fits_path)

    assert isinstance(image, ScienceImage)
    assert image.data.shape == (10, 10)
    assert image.data.dtype == np.float32
    assert image.filepath == fits_path

    # Metadata should be injected from PRIMARY if missing in SCI, or preserved if in SCI
    assert image.header['TELESCOP'] == 'SYNTHETIC'
    assert image.header['BUNIT'] == 'ELECTRONS/S'

def test_load_science_image_primary_fallback(tmp_path):
    """Test loading a FITS file where science data is in the PRIMARY extension."""
    fits_path = tmp_path / "test_primary.fits"
    create_synthetic_fits(fits_path, include_sci=False)

    image = load_science_image(fits_path)

    assert isinstance(image, ScienceImage)
    assert image.data.shape == (10, 10)
    assert image.header['TELESCOP'] == 'SYNTHETIC'

def test_load_science_image_invalid_no_data(tmp_path):
    """Test loading a FITS file with no valid 2D image data."""
    fits_path = tmp_path / "empty.fits"
    primary_hdu = fits.PrimaryHDU()
    hdul = fits.HDUList([primary_hdu])
    hdul.writeto(fits_path, overwrite=True)

    with pytest.raises(ProcessingError, match="No SCI extension or valid 2D PRIMARY data"):
        load_science_image(fits_path)

def test_load_science_image_not_found(tmp_path):
    """Test loading a missing FITS file."""
    with pytest.raises(ProcessingError, match="FITS file not found"):
        load_science_image(tmp_path / "missing.fits")

def test_compute_statistics_normal():
    """Test statistics on a clean array without NaNs."""
    data = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0]
    ], dtype=np.float32)

    image = ScienceImage(data=data, filepath="dummy.fits")
    stats = compute_statistics(image)

    assert stats.shape == (3, 3)
    assert stats.finite_pixels == 9
    assert stats.nan_pixels == 0
    assert stats.min_val == 1.0
    assert stats.max_val == 9.0
    assert stats.mean_val == 5.0
    assert stats.median_val == 5.0
    assert pytest.approx(stats.std_val, 0.01) == 2.581988

def test_compute_statistics_with_nans():
    """Test statistics correctly ignore NaN values (e.g. dead pixels, masked regions)."""
    data = np.array([
        [1.0, 2.0, np.nan],
        [4.0, 5.0, 6.0],
        [np.nan, 8.0, 9.0]
    ], dtype=np.float32)

    image = ScienceImage(data=data, filepath="dummy.fits")
    stats = compute_statistics(image)

    assert stats.finite_pixels == 7
    assert stats.nan_pixels == 2
    assert stats.min_val == 1.0
    assert stats.max_val == 9.0

    # Expected mean: (1+2+4+5+6+8+9) / 7 = 35 / 7 = 5.0
    assert stats.mean_val == 5.0

    # Sorted valid: 1, 2, 4, 5, 6, 8, 9. Median is 5.
    assert stats.median_val == 5.0

def test_compute_statistics_all_nans():
    """Test statistics handle all-NaN arrays gracefully without crashing."""
    data = np.full((3, 3), np.nan, dtype=np.float32)

    image = ScienceImage(data=data, filepath="dummy.fits")
    stats = compute_statistics(image)

    assert stats.finite_pixels == 0
    assert stats.nan_pixels == 9
    assert np.isnan(stats.min_val)
    assert np.isnan(stats.max_val)
    assert np.isnan(stats.mean_val)
    assert np.isnan(stats.median_val)
    assert np.isnan(stats.std_val)
