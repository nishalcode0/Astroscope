"""Unit tests for the Product dataclass.

All tests are offline and deterministic — no network calls are made.
"""

import pytest
from dataclasses import FrozenInstanceError

from astroscope.product import Product


def _make_product(**kwargs) -> Product:
    """Construct a minimal valid Product, overriding any fields via kwargs."""
    defaults = dict(
        product_id="n4eya1020_mos.fits",
        parent_obsid=24854354,
        collection="HST",
        product_type="SCIENCE",
        data_product_type="image",
        calib_level=3,
        filters="F160W",
        filename="n4eya1020_mos.fits",
        data_uri="mast:HST/product/n4eya1020_mos.fits",
    )
    defaults.update(kwargs)
    return Product(**defaults)


# ---------------------------------------------------------------------------
# Instantiation
# ---------------------------------------------------------------------------

def test_product_instantiation_minimal() -> None:
    """Verify Product can be created with required fields only."""
    prod = _make_product()
    assert prod.product_id == "n4eya1020_mos.fits"
    assert prod.parent_obsid == 24854354
    assert prod.collection == "HST"
    assert prod.product_type == "SCIENCE"
    assert prod.data_product_type == "image"
    assert prod.calib_level == 3
    assert prod.filters == "F160W"
    assert prod.filename == "n4eya1020_mos.fits"
    assert prod.data_uri == "mast:HST/product/n4eya1020_mos.fits"


def test_product_optional_defaults() -> None:
    """Verify optional fields default to None / empty string / PUBLIC."""
    prod = _make_product()
    assert prod.file_size is None
    assert prod.data_rights == "PUBLIC"
    assert prod.description == ""
    assert prod.product_group_description == ""
    assert prod.product_sub_group_description == ""
    assert prod.extra_metadata == {}


def test_product_with_all_fields() -> None:
    """Verify Product can be created with all optional fields populated."""
    prod = _make_product(
        file_size=1048576,
        data_rights="PUBLIC",
        description="Mosaic drizzle image",
        product_group_description="Minimum Recommended Products",
        product_sub_group_description="SCIENCE",
        extra_metadata={"provenance": "CALNIC"},
    )
    assert prod.file_size == 1048576
    assert prod.description == "Mosaic drizzle image"
    assert prod.product_group_description == "Minimum Recommended Products"
    assert prod.product_sub_group_description == "SCIENCE"
    assert prod.extra_metadata == {"provenance": "CALNIC"}


# ---------------------------------------------------------------------------
# Immutability
# ---------------------------------------------------------------------------

def test_product_immutability() -> None:
    """Verify Product fields cannot be mutated after creation (frozen dataclass)."""
    prod = _make_product()
    with pytest.raises(FrozenInstanceError):
        prod.collection = "HLA"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Product types
# ---------------------------------------------------------------------------

def test_product_science_type() -> None:
    """Verify SCIENCE product_type is preserved correctly."""
    prod = _make_product(product_type="SCIENCE")
    assert prod.product_type == "SCIENCE"


def test_product_preview_type() -> None:
    """Verify PREVIEW product_type is preserved correctly."""
    prod = _make_product(
        product_type="PREVIEW",
        filename="n4eya1020_mos.jpg",
        data_uri="mast:HST/product/n4eya1020_mos.jpg",
        product_id="n4eya1020_mos.jpg",
    )
    assert prod.product_type == "PREVIEW"


def test_product_auxiliary_type() -> None:
    """Verify AUXILIARY product_type is preserved correctly."""
    prod = _make_product(
        product_type="AUXILIARY",
        filename="n4eya1020_jif.fits",
        data_uri="mast:HST/product/n4eya1020_jif.fits",
        product_id="n4eya1020_jif.fits",
    )
    assert prod.product_type == "AUXILIARY"


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------

def test_product_hst_collection() -> None:
    """Verify HST collection label is preserved."""
    prod = _make_product(collection="HST")
    assert prod.collection == "HST"


def test_product_hla_collection() -> None:
    """Verify HLA collection label is preserved (distinct from HST)."""
    prod = _make_product(
        collection="HLA",
        product_id="HST_7171_a1_NIC_NIC2_F160W_drz.fits",
        filename="HST_7171_a1_NIC_NIC2_F160W_drz.fits",
        data_uri="mast:HLA/product/HST_7171_a1_NIC_NIC2_F160W_drz.fits",
        calib_level=4,
    )
    assert prod.collection == "HLA"
    assert prod.calib_level == 4


# ---------------------------------------------------------------------------
# Calibration levels
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("level", [0, 1, 2, 3, 4])
def test_product_calib_level_range(level: int) -> None:
    """Verify calibration levels 0–4 are all preserved."""
    prod = _make_product(calib_level=level)
    assert prod.calib_level == level


# ---------------------------------------------------------------------------
# Data URI
# ---------------------------------------------------------------------------

def test_product_data_uri_preserved() -> None:
    """Verify data_uri is stored exactly as provided."""
    uri = "mast:HST/product/n4eya1pzq_raw.fits"
    prod = _make_product(data_uri=uri, product_id="n4eya1pzq_raw.fits", filename="n4eya1pzq_raw.fits")
    assert prod.data_uri == uri


# ---------------------------------------------------------------------------
# Extra metadata
# ---------------------------------------------------------------------------

def test_product_extra_metadata_preserved() -> None:
    """Verify extra_metadata dict is stored without modification."""
    meta = {"provenance_name": "CALNIC", "wave_region": "Infrared", "em_min": 1408.4}
    prod = _make_product(extra_metadata=meta)
    assert prod.extra_metadata == meta
