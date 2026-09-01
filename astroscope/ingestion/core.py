"""Core ingestion logic for the Astroscope computational observatory."""

import logging
import os
from pathlib import Path

from astroscope.observation import Observation
from astroscope.product import Product
from astroscope.archive.mast import download_file, MastQueryError

log = logging.getLogger(__name__)

class IngestionError(Exception):
    """Raised when ingestion fails for any reason."""

def get_local_path(base_dir: str, mission: str, observation_id: str, filename: str) -> Path:
    """Determine the deterministic local path for a downloaded product."""
    return Path(base_dir) / "raw" / mission / observation_id / filename

def ingest_product(
    product: Product,
    observation: Observation,
    base_dir: str = "data",
) -> Path:
    """Ingest a single scientific data product into local storage.

    Parameters
    ----------
    product:
        The data product metadata containing the MAST URI.
    observation:
        The parent observation context (used for deterministic path generation).
    base_dir:
        The root directory for the local data store (default: "data").

    Returns
    -------
    Path
        The local filesystem path to the ingested file.

    Raises
    ------
    IngestionError
        If the download fails or the URI is invalid.
    """
    if not product.data_uri:
        raise IngestionError(f"Product {product.product_id} has no data URI.")

    # 1. Determine local path
    final_path = get_local_path(
        base_dir, observation.mission, observation.observation_id, product.filename
    )
    
    # 2. Duplicate detection
    if final_path.exists():
        if final_path.stat().st_size > 0:
            if product.file_size is not None and final_path.stat().st_size != product.file_size:
                log.warning(
                    "File exists but size mismatch for %s. Expected %s, got %s. Re-downloading.",
                    final_path, product.file_size, final_path.stat().st_size
                )
            else:
                log.info("Product %s already exists at %s, skipping download.", product.product_id, final_path)
                return final_path
        else:
            log.warning("File exists but is empty at %s, re-downloading.", final_path)

    # 3. Prepare for download
    final_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = final_path.with_suffix(final_path.suffix + ".part")

    # 4. Download safely
    log.info("Downloading %s to %s", product.data_uri, temp_path)
    try:
        download_file(product.data_uri, str(temp_path))
    except MastQueryError as exc:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass
        raise IngestionError(f"Failed to download {product.product_id}: {exc}") from exc
    except Exception as exc:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass
        raise IngestionError(f"Unexpected error downloading {product.product_id}: {exc}") from exc

    # 5. Atomic rename on success
    try:
        os.replace(temp_path, final_path)
    except OSError as exc:
        raise IngestionError(f"Failed to move downloaded file to {final_path}: {exc}") from exc

    log.info("Successfully ingested %s to %s", product.product_id, final_path)
    return final_path
