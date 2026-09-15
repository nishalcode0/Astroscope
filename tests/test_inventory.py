"""Tests for the Astroscope local inventory."""

import json

import pytest

from astroscope.inventory.core import (
    InventoryFormatError,
    LocalInventory,
    LocalProduct,
)


def make_product() -> LocalProduct:
    """Create a representative local product for testing."""
    return LocalProduct(
        product_id="n4eya1020_mos.fits",
        mission="HST",
        observation_id="n4eya1020",
        filename="n4eya1020_mos.fits",
        local_path="raw/HST/n4eya1020/n4eya1020_mos.fits",
        file_size=1234567,
    )


def test_missing_inventory_is_empty(tmp_path):
    """A missing inventory file should behave as an empty inventory."""
    inventory = LocalInventory(
        tmp_path / "inventory.json"
    )

    inventory.load()

    assert inventory.list_products() == []


def test_register_and_get_product(tmp_path):
    """A registered product should be retrievable by ID."""
    inventory = LocalInventory(
        tmp_path / "inventory.json"
    )

    product = make_product()

    inventory.register(product)

    assert inventory.get(product.product_id) == product


def test_save_and_reload_inventory(tmp_path):
    """Registered products should survive persistence and reload."""
    inventory_path = tmp_path / "inventory.json"

    inventory = LocalInventory(inventory_path)

    product = make_product()

    inventory.register(product)
    inventory.save()

    reloaded = LocalInventory(inventory_path)
    reloaded.load()

    assert reloaded.get(product.product_id) == product


def test_list_products_is_deterministic(tmp_path):
    """Products should be returned in product-ID order."""
    inventory = LocalInventory(
        tmp_path / "inventory.json"
    )

    product_b = LocalProduct(
        product_id="b.fits",
        mission="HST",
        observation_id="obs2",
        filename="b.fits",
        local_path="raw/HST/obs2/b.fits",
        file_size=20,
    )

    product_a = LocalProduct(
        product_id="a.fits",
        mission="HST",
        observation_id="obs1",
        filename="a.fits",
        local_path="raw/HST/obs1/a.fits",
        file_size=10,
    )

    inventory.register(product_b)
    inventory.register(product_a)

    assert [
        product.product_id
        for product in inventory.list_products()
    ] == [
        "a.fits",
        "b.fits",
    ]


def test_remove_does_not_delete_file(tmp_path):
    """Removing an entry should only change inventory state."""
    inventory = LocalInventory(
        tmp_path / "inventory.json"
    )

    product = make_product()

    inventory.register(product)

    assert inventory.remove(product.product_id) is True
    assert inventory.get(product.product_id) is None
    assert inventory.remove(product.product_id) is False


def test_invalid_product_is_rejected():
    """Invalid LocalProduct values should raise ValueError."""
    with pytest.raises(ValueError):
        LocalProduct(
            product_id="",
            mission="HST",
            observation_id="obs",
            filename="file.fits",
            local_path="raw/file.fits",
            file_size=10,
        )

    with pytest.raises(ValueError):
        LocalProduct(
            product_id="file.fits",
            mission="HST",
            observation_id="obs",
            filename="file.fits",
            local_path="raw/file.fits",
            file_size=-1,
        )


def test_malformed_json_is_rejected(tmp_path):
    """Malformed JSON should raise an inventory-specific error."""
    inventory_path = tmp_path / "inventory.json"

    inventory_path.write_text(
        "{not valid json",
        encoding="utf-8",
    )

    inventory = LocalInventory(inventory_path)

    with pytest.raises(InventoryFormatError):
        inventory.load()


def test_non_list_json_is_rejected(tmp_path):
    """The inventory root must be a JSON list."""
    inventory_path = tmp_path / "inventory.json"

    inventory_path.write_text(
        json.dumps({"product_id": "bad"}),
        encoding="utf-8",
    )

    inventory = LocalInventory(inventory_path)

    with pytest.raises(InventoryFormatError):
        inventory.load()


def test_scan_local_fits_files(tmp_path):
    """Scanner should discover FITS files without modifying inventory."""
    raw_root = tmp_path / "raw"

    first = (
        raw_root
        / "HST"
        / "obs1"
        / "image.fits"
    )

    second = (
        raw_root
        / "HST"
        / "obs2"
        / "image.FITS"
    )

    ignored = (
        raw_root
        / "HST"
        / "obs3"
        / "notes.txt"
    )

    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    ignored.parent.mkdir(parents=True)

    first.write_bytes(b"fake fits")
    second.write_bytes(b"fake fits")
    ignored.write_text(
        "not a FITS file",
        encoding="utf-8",
    )

    inventory = LocalInventory(
        tmp_path / "inventory.json"
    )

    found = inventory.scan_local_files(raw_root)

    assert found == sorted(
        [
            first,
            second,
        ]
    )


def test_scan_missing_directory_returns_empty(tmp_path):
    """A missing raw-data directory should return no files."""
    inventory = LocalInventory(
        tmp_path / "inventory.json"
    )

    assert (
        inventory.scan_local_files(
            tmp_path / "does_not_exist"
        )
        == []
    )