"""Core ingestion logic for the Astroscope computational observatory."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from astroscope.archive.mast import MastQueryError, download_file
from astroscope.inventory.core import LocalInventory, LocalProduct
from astroscope.observation import Observation
from astroscope.product import Product

log = logging.getLogger(__name__)


class IngestionError(Exception):
    """Raised when ingestion fails for any reason."""


def get_local_path(
    base_dir: str,
    mission: str,
    observation_id: str,
    filename: str,
) -> Path:
    """Determine the deterministic local path for a downloaded product."""
    return (
        Path(base_dir)
        / "raw"
        / mission
        / observation_id
        / filename
    )


def _register_local_product(
    product: Product,
    observation: Observation,
    final_path: Path,
    base_dir: str,
) -> None:
    """Register a successfully available product in the local inventory."""
    base_path = Path(base_dir)

    try:
        relative_path = final_path.relative_to(base_path)
    except ValueError as exc:
        raise IngestionError(
            f"Downloaded file {final_path} is outside data root "
            f"{base_path}."
        ) from exc

    local_product = LocalProduct(
        product_id=product.product_id,
        mission=observation.mission,
        observation_id=observation.observation_id,
        filename=product.filename,
        local_path=relative_path.as_posix(),
        file_size=final_path.stat().st_size,
    )

    inventory = LocalInventory(
        base_path / "inventory.json"
    )

    inventory.load()
    inventory.register(local_product)
    inventory.save()

    log.info(
        "Registered product %s in local inventory.",
        product.product_id,
    )


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
        The parent observation context used for deterministic
        path generation and local provenance.

    base_dir:
        Root directory for the local data store.

    Returns
    -------
    Path
        Local filesystem path to the ingested file.

    Raises
    ------
    IngestionError
        If the download or inventory registration fails.
    """
    if not product.data_uri:
        raise IngestionError(
            f"Product {product.product_id} has no data URI."
        )

    # 1. Determine local path.
    final_path = get_local_path(
        base_dir,
        observation.mission,
        observation.observation_id,
        product.filename,
    )

    # 2. Duplicate detection.
    if final_path.exists():
        if final_path.stat().st_size > 0:
            if (
                product.file_size is not None
                and final_path.stat().st_size != product.file_size
            ):
                log.warning(
                    "File exists but size mismatch for %s. "
                    "Expected %s, got %s. Re-downloading.",
                    final_path,
                    product.file_size,
                    final_path.stat().st_size,
                )
            else:
                log.info(
                    "Product %s already exists at %s, "
                    "skipping download.",
                    product.product_id,
                    final_path,
                )

                # The file may predate the inventory system.
                # Register it so the local inventory reflects reality.
                try:
                    _register_local_product(
                        product,
                        observation,
                        final_path,
                        base_dir,
                    )
                except Exception as exc:
                    raise IngestionError(
                        f"Product exists but inventory registration "
                        f"failed for {product.product_id}: {exc}"
                    ) from exc

                return final_path
        else:
            log.warning(
                "File exists but is empty at %s, re-downloading.",
                final_path,
            )

    # 3. Prepare for download.
    final_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_path = final_path.with_suffix(
        final_path.suffix + ".part"
    )

    # Remove a stale temporary file before starting.
    if temp_path.exists():
        try:
            temp_path.unlink()
        except OSError as exc:
            raise IngestionError(
                f"Could not remove stale temporary file "
                f"{temp_path}: {exc}"
            ) from exc

    # 4. Download safely.
    log.info(
        "Downloading %s to %s",
        product.data_uri,
        temp_path,
    )

    try:
        download_file(
            product.data_uri,
            str(temp_path),
        )

    except MastQueryError as exc:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass

        raise IngestionError(
            f"Failed to download {product.product_id}: {exc}"
        ) from exc

    except Exception as exc:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass

        raise IngestionError(
            f"Unexpected error downloading "
            f"{product.product_id}: {exc}"
        ) from exc

    # 5. Verify that the temporary download actually exists.
    if not temp_path.exists():
        raise IngestionError(
            f"Download completed but temporary file was not created: "
            f"{temp_path}"
        )

    if temp_path.stat().st_size == 0:
        try:
            temp_path.unlink()
        except OSError:
            pass

        raise IngestionError(
            f"Download produced an empty file for "
            f"{product.product_id}."
        )

    # 6. Atomic rename on success.
    try:
        os.replace(
            temp_path,
            final_path,
        )

    except OSError as exc:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass

        raise IngestionError(
            f"Failed to move downloaded file to "
            f"{final_path}: {exc}"
        ) from exc

    # 7. Register only after the final file exists.
    try:
        _register_local_product(
            product,
            observation,
            final_path,
            base_dir,
        )

    except Exception as exc:
        raise IngestionError(
            f"File downloaded successfully, but local inventory "
            f"registration failed for {product.product_id}: {exc}"
        ) from exc

    log.info(
        "Successfully ingested %s to %s",
        product.product_id,
        final_path,
    )

    return final_path