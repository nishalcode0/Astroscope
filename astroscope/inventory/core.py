"""Persistent local inventory for Astroscope astronomical data products."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional


class InventoryError(Exception):
    """Base exception for local inventory errors."""


class InventoryFormatError(InventoryError):
    """Raised when the inventory file contains invalid JSON or records."""


@dataclass(frozen=True)
class LocalProduct:
    """Metadata describing an astronomical product stored locally."""

    product_id: str
    mission: str
    observation_id: str
    filename: str
    local_path: str
    file_size: int

    def __post_init__(self) -> None:
        """Validate the local product record."""
        if not self.product_id.strip():
            raise ValueError("product_id must not be empty.")

        if not self.mission.strip():
            raise ValueError("mission must not be empty.")

        if not self.observation_id.strip():
            raise ValueError("observation_id must not be empty.")

        if not self.filename.strip():
            raise ValueError("filename must not be empty.")

        if not self.local_path.strip():
            raise ValueError("local_path must not be empty.")

        if self.file_size < 0:
            raise ValueError("file_size must be non-negative.")


class LocalInventory:
    """Persistent inventory of locally available astronomical products."""

    def __init__(
        self,
        inventory_path: Path | str = "data/inventory.json",
    ) -> None:
        self.inventory_path = Path(inventory_path)
        self._products: dict[str, LocalProduct] = {}

    def load(self) -> None:
        """Load the inventory from disk.

        A missing inventory file is treated as an empty inventory.

        Raises
        ------
        InventoryFormatError
            If the JSON file or any record is malformed.
        """
        if not self.inventory_path.exists():
            self._products = {}
            return

        try:
            with self.inventory_path.open(
                "r",
                encoding="utf-8",
            ) as handle:
                data = json.load(handle)

        except json.JSONDecodeError as exc:
            raise InventoryFormatError(
                f"Invalid inventory JSON: {self.inventory_path}"
            ) from exc

        except OSError as exc:
            raise InventoryError(
                f"Could not read inventory: {self.inventory_path}"
            ) from exc

        if not isinstance(data, list):
            raise InventoryFormatError(
                "Inventory root must be a JSON list."
            )

        products: dict[str, LocalProduct] = {}

        for index, record in enumerate(data):
            if not isinstance(record, dict):
                raise InventoryFormatError(
                    f"Inventory record {index} must be a JSON object."
                )

            try:
                product = LocalProduct(**record)

            except (TypeError, ValueError) as exc:
                raise InventoryFormatError(
                    f"Invalid inventory record {index}: {exc}"
                ) from exc

            if product.product_id in products:
                raise InventoryFormatError(
                    f"Duplicate product_id: {product.product_id}"
                )

            products[product.product_id] = product

        self._products = products

    def save(self) -> None:
        """Persist the inventory using an atomic file replacement."""
        self.inventory_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        temporary_path = self.inventory_path.with_suffix(
            self.inventory_path.suffix + ".part"
        )

        data = [
            asdict(product)
            for product in self._products.values()
        ]

        try:
            with temporary_path.open(
                "w",
                encoding="utf-8",
            ) as handle:
                json.dump(
                    data,
                    handle,
                    indent=2,
                    sort_keys=True,
                )
                handle.write("\n")

            os.replace(
                temporary_path,
                self.inventory_path,
            )

        except OSError as exc:
            if temporary_path.exists():
                try:
                    temporary_path.unlink()
                except OSError:
                    pass

            raise InventoryError(
                f"Could not save inventory: {self.inventory_path}"
            ) from exc

    def register(self, product: LocalProduct) -> None:
        """Register or update a locally available product."""
        self._products[product.product_id] = product

    def get(self, product_id: str) -> Optional[LocalProduct]:
        """Return a product by ID, or None if it is not registered."""
        return self._products.get(product_id)

    def list_products(self) -> List[LocalProduct]:
        """Return all registered products in deterministic order."""
        return sorted(
            self._products.values(),
            key=lambda product: product.product_id,
        )

    def remove(self, product_id: str) -> bool:
        """Remove a product from the inventory.

        This does NOT delete the corresponding scientific file.

        Returns
        -------
        bool
            True if an entry was removed, otherwise False.
        """
        if product_id not in self._products:
            return False

        del self._products[product_id]
        return True

    def scan_local_files(
        self,
        raw_root: Path | str = "data/raw",
    ) -> List[Path]:
        """Find FITS files present under the local raw-data directory.

        This only discovers files. It does not modify the inventory and
        does not attempt to infer archive metadata.
        """
        root = Path(raw_root)

        if not root.exists():
            return []

        if not root.is_dir():
            raise InventoryError(
                f"Raw data path is not a directory: {root}"
            )

        files = [
            path
            for path in root.rglob("*")
            if path.is_file()
            and path.suffix.lower() in {
                ".fits",
                ".fit",
                ".fts",
            }
        ]

        return sorted(files)