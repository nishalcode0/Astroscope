"""Core processing functions for astronomical image data."""

import logging
from pathlib import Path

import numpy as np
from astropy.io import fits

from astroscope.processing.models import ImageStatistics, ScienceImage

log = logging.getLogger(__name__)

class ProcessingError(Exception):
    """Raised when processing fails due to invalid data or unsupported formats."""

def load_science_image(filepath: Path) -> ScienceImage:
    """Load the scientific data array (SCI) from a raw FITS file.

    This function handles the extraction of the actual scientific signal,
    ignoring auxiliary extensions unless necessary, and converting to a
    floating-point representation to allow for correct NaN handling.

    Parameters
    ----------
    filepath:
        Path to the raw FITS file on disk.

    Returns
    -------
    ScienceImage
        The in-memory representation of the scientific data.

    Raises
    ------
    ProcessingError
        If the file cannot be opened, or no valid SCI extension is found.
    """
    if not filepath.exists():
        raise ProcessingError(f"FITS file not found: {filepath}")

    try:
        # Open with memmap=True to avoid loading all extensions into memory
        with fits.open(filepath, memmap=True) as hdul:
            sci_hdu = None

            # 1. Look for explicit SCI extension (standard HST/JWST pipeline format)
            for hdu in hdul:
                if hdu.name == 'SCI':
                    sci_hdu = hdu
                    break

            # 2. Fallback to PRIMARY if it contains actual image data (simple FITS)
            if sci_hdu is None and hdul[0].data is not None and hdul[0].data.ndim >= 2:
                sci_hdu = hdul[0]

            if sci_hdu is None:
                raise ProcessingError("No SCI extension or valid 2D PRIMARY data found in FITS.")

            # Copy data into memory to safely close the file and avoid memmap issues later
            # Convert to float32 to ensure standard scientific representation
            raw_data = np.array(sci_hdu.data, dtype=np.float32)

            # Extract header dict (basic key-value pairs)
            header_dict = dict(sci_hdu.header)

            # Also capture primary header metadata if we used SCI
            if sci_hdu.name == 'SCI':
                primary_header = dict(hdul[0].header)
                # Inject basic mission/instrument metadata into the science header if missing
                for key in ['TELESCOP', 'INSTRUME', 'FILTER']:
                    if key in primary_header and key not in header_dict:
                        header_dict[key] = primary_header[key]

            return ScienceImage(data=raw_data, filepath=filepath, header=header_dict)

    except OSError as exc:
        raise ProcessingError(f"Failed to open FITS file {filepath}: {exc}") from exc
    except Exception as exc:
        if isinstance(exc, ProcessingError):
            raise
        raise ProcessingError(f"Unexpected error loading {filepath}: {exc}") from exc


def compute_statistics(image: ScienceImage) -> ImageStatistics:
    """Compute mathematically sound basic statistics for a science image.

    Properly masks NaN and infinity values which are common in uncalibrated
    or masked astronomical data.
    """
    data = image.data

    # Identify finite pixels (not NaN, not Inf)
    finite_mask = np.isfinite(data)
    finite_pixels = int(np.count_nonzero(finite_mask))
    nan_pixels = int(data.size - finite_pixels)

    if finite_pixels == 0:
        # Handle all-NaN case gracefully
        return ImageStatistics(
            shape=data.shape,
            finite_pixels=finite_pixels,
            nan_pixels=nan_pixels,
            min_val=float('nan'),
            max_val=float('nan'),
            mean_val=float('nan'),
            median_val=float('nan'),
            std_val=float('nan')
        )

    valid_data = data[finite_mask]

    return ImageStatistics(
        shape=data.shape,
        finite_pixels=finite_pixels,
        nan_pixels=nan_pixels,
        min_val=float(np.min(valid_data)),
        max_val=float(np.max(valid_data)),
        mean_val=float(np.mean(valid_data)),
        median_val=float(np.median(valid_data)),
        std_val=float(np.std(valid_data))
    )
