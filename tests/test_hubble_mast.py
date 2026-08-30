"""Comprehensive offline unit tests for HubbleAdapter and MAST normalisation.

All tests use mocked MAST responses (synthetic Astropy Tables).
No network calls are made.  No FITS files are opened.

Test coverage:
1.  HubbleAdapter mission name.
2.  Hubble search result normalisation → Observation.
3.  Both obs_id and MAST obsid preservation.
4.  RA/Dec normalisation.
5.  Masked / NaN metadata handling (RA, Dec, instrument).
6.  Observation time conversion from MJD (t_min).
7.  Product table → Product normalisation.
8.  SCIENCE product type preservation.
9.  PREVIEW product type preservation.
10. AUXILIARY product type preservation.
11. Calibration level preservation.
12. HLA vs HST collection label preservation.
13. Product data_uri preservation.
14. get_data_product_uris compat method returns SCIENCE URIs only.
15. MastQueryError raised when no criteria provided.
16. MastQueryError raised on partial coordinate input.
17. MAST obsid missing → MastQueryError from get_data_product_uris.
18. Integer mast_obsid is converted to str before calling get_product_list (regression).
"""

import math
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from astropy.table import MaskedColumn, Table

from astroscope.archive.hubble import HubbleAdapter, _row_to_observation, _row_to_product
from astroscope.archive.mast import MastQueryError, query_hst_observations
from astroscope.observation import Observation
from astroscope.product import Product


# ---------------------------------------------------------------------------
# Helpers: synthetic Astropy Tables matching real MAST schema
# ---------------------------------------------------------------------------

def _make_obs_table(rows: List[Dict[str, Any]]) -> Table:
    """Build a minimal MAST observation table from a list of row dicts."""
    if not rows:
        return Table()

    # Collect all keys across all rows
    all_keys = list(rows[0].keys())

    data: Dict[str, list] = {k: [] for k in all_keys}
    for row in rows:
        for k in all_keys:
            data[k].append(row.get(k, ""))

    return Table(data)


def _make_product_table(rows: List[Dict[str, Any]]) -> Table:
    """Build a minimal MAST product table from a list of row dicts."""
    if not rows:
        return Table()
    all_keys = list(rows[0].keys())
    data: Dict[str, list] = {k: [] for k in all_keys}
    for row in rows:
        for k in all_keys:
            data[k].append(row.get(k, ""))
    return Table(data)


def _sample_obs_row(**overrides) -> Dict[str, Any]:
    """Return a synthetic MAST observation row matching real field names."""
    row = {
        "obs_id": "n4eya1020",
        "obsid": 24854354,
        "target_name": "M31",
        "s_ra": 10.68483284454,
        "s_dec": 41.2697105005,
        "instrument_name": "NICMOS/NIC2",
        "filters": "F160W",
        "t_min": 50784.88284722,
        "t_max": 50784.88576389,
        "t_exptime": 64.0,
        "t_obs_release": 51154.75641203,
        "calib_level": 3,
        "dataRights": "PUBLIC",
        "obs_title": "Stellar Distributions in Nearby Galaxy Nuclei",
        "proposal_id": "7171",
        "target_classification": "GALAXY",
        "wave_region": "Infrared",
        "provenance_name": "CALNIC",
        "dataproduct_type": "image",
        "obs_collection": "HST",
    }
    row.update(overrides)
    return row


def _sample_science_product_row(**overrides) -> Dict[str, Any]:
    row = {
        "productFilename": "n4eya1020_mos.fits",
        "obsID": 24854354,
        "obs_collection": "HST",
        "productType": "SCIENCE",
        "dataproduct_type": "image",
        "calib_level": 3,
        "filters": "F160W",
        "description": "Mosaic drizzle image",
        "dataURI": "mast:HST/product/n4eya1020_mos.fits",
        "size": 10485760,
        "dataRights": "PUBLIC",
        "productGroupDescription": "Minimum Recommended Products",
        "productSubGroupDescription": "SCIENCE",
    }
    row.update(overrides)
    return row


# ---------------------------------------------------------------------------
# 1. Mission name
# ---------------------------------------------------------------------------

def test_hubble_mission_name() -> None:
    """HubbleAdapter.mission_name must return the canonical HST name."""
    adapter = HubbleAdapter()
    assert adapter.mission_name == "Hubble Space Telescope"


# ---------------------------------------------------------------------------
# 2. Search result normalisation
# ---------------------------------------------------------------------------

def test_search_normalises_observation() -> None:
    """search_observations must normalise a MAST table row into an Observation."""
    raw = _make_obs_table([_sample_obs_row()])
    with patch("astroscope.archive.mast.Observations") as mock_obs:
        mock_obs.query_criteria.return_value = raw
        adapter = HubbleAdapter()
        results = adapter.search_observations(target_name="M31")

    assert len(results) == 1
    obs = results[0]
    assert isinstance(obs, Observation)
    assert obs.mission == "HST"
    assert obs.target_name == "M31"
    assert obs.instrument == "NICMOS/NIC2"
    assert obs.filter_band == "F160W"


# ---------------------------------------------------------------------------
# 3. obs_id AND mast_obsid preservation
# ---------------------------------------------------------------------------

def test_search_preserves_obs_id_and_mast_obsid() -> None:
    """Both mission obs_id and MAST integer obsid must be preserved."""
    raw = _make_obs_table([_sample_obs_row(obs_id="n4eya1020", obsid=24854354)])
    with patch("astroscope.archive.mast.Observations") as mock_obs:
        mock_obs.query_criteria.return_value = raw
        adapter = HubbleAdapter()
        results = adapter.search_observations(target_name="M31")

    obs = results[0]
    assert obs.observation_id == "n4eya1020"
    assert obs.mast_obsid == 24854354


# ---------------------------------------------------------------------------
# 4. RA / Dec normalisation
# ---------------------------------------------------------------------------

def test_search_normalises_ra_dec() -> None:
    """RA and Dec must be stored as floats within valid astronomical ranges."""
    raw = _make_obs_table([_sample_obs_row(s_ra=10.6848, s_dec=41.2697)])
    with patch("astroscope.archive.mast.Observations") as mock_obs:
        mock_obs.query_criteria.return_value = raw
        adapter = HubbleAdapter()
        results = adapter.search_observations(target_name="M31")

    obs = results[0]
    assert abs(obs.right_ascension - 10.6848) < 1e-6
    assert abs(obs.declination - 41.2697) < 1e-6


# ---------------------------------------------------------------------------
# 5. Masked / NaN metadata handling
# ---------------------------------------------------------------------------

def test_search_skips_row_with_masked_ra() -> None:
    """Rows with masked RA must be skipped gracefully (logged, not raised)."""
    row_dict = _sample_obs_row()
    raw = _make_obs_table([row_dict])
    # Replace s_ra column with a masked value
    raw["s_ra"] = MaskedColumn([np.ma.masked], name="s_ra")

    with patch("astroscope.archive.mast.Observations") as mock_obs:
        mock_obs.query_criteria.return_value = raw
        adapter = HubbleAdapter()
        results = adapter.search_observations(target_name="M31")

    # Row should be skipped; no exception raised
    assert results == []


def test_search_skips_row_with_nan_dec() -> None:
    """Rows with NaN Dec must be skipped gracefully."""
    raw = _make_obs_table([_sample_obs_row(s_dec=float("nan"))])
    with patch("astroscope.archive.mast.Observations") as mock_obs:
        mock_obs.query_criteria.return_value = raw
        adapter = HubbleAdapter()
        results = adapter.search_observations(target_name="M31")

    assert results == []


def test_search_handles_masked_instrument() -> None:
    """Masked instrument_name should produce a fallback string, not a crash."""
    raw = _make_obs_table([_sample_obs_row()])
    raw["instrument_name"] = MaskedColumn(["--"], name="instrument_name")

    with patch("astroscope.archive.mast.Observations") as mock_obs:
        mock_obs.query_criteria.return_value = raw
        adapter = HubbleAdapter()
        results = adapter.search_observations(target_name="M31")

    assert len(results) == 1
    # Should fall back to "UNKNOWN" for masked instrument
    assert results[0].instrument == "UNKNOWN"


# ---------------------------------------------------------------------------
# 6. Observation time from t_min MJD
# ---------------------------------------------------------------------------

def test_observation_t_min_mjd_stored() -> None:
    """t_min MJD value must be stored in the Observation.t_min field."""
    raw = _make_obs_table([_sample_obs_row(t_min=50784.88284722)])
    with patch("astroscope.archive.mast.Observations") as mock_obs:
        mock_obs.query_criteria.return_value = raw
        adapter = HubbleAdapter()
        results = adapter.search_observations(target_name="M31")

    obs = results[0]
    assert obs.t_min is not None
    assert abs(obs.t_min - 50784.88284722) < 1e-6


def test_observation_start_datetime_from_mjd() -> None:
    """observation_start_datetime() must convert t_min MJD to a UTC datetime."""
    raw = _make_obs_table([_sample_obs_row(t_min=50784.88284722)])
    with patch("astroscope.archive.mast.Observations") as mock_obs:
        mock_obs.query_criteria.return_value = raw
        adapter = HubbleAdapter()
        results = adapter.search_observations(target_name="M31")

    obs = results[0]
    dt = obs.observation_start_datetime()
    assert dt is not None
    # MJD 50784 is around 1997-12 — sanity check the year
    assert dt.year == 1997


def test_t_exptime_stored() -> None:
    """t_exptime must be stored as a float in seconds."""
    raw = _make_obs_table([_sample_obs_row(t_exptime=64.0)])
    with patch("astroscope.archive.mast.Observations") as mock_obs:
        mock_obs.query_criteria.return_value = raw
        adapter = HubbleAdapter()
        results = adapter.search_observations(target_name="M31")

    obs = results[0]
    assert obs.t_exptime == 64.0


def test_calib_level_stored() -> None:
    """calib_level must be stored as an int."""
    raw = _make_obs_table([_sample_obs_row(calib_level=3)])
    with patch("astroscope.archive.mast.Observations") as mock_obs:
        mock_obs.query_criteria.return_value = raw
        adapter = HubbleAdapter()
        results = adapter.search_observations(target_name="M31")

    obs = results[0]
    assert obs.calib_level == 3


# ---------------------------------------------------------------------------
# 7–10. Product table → Product normalisation
# ---------------------------------------------------------------------------

def test_product_normalisation_science() -> None:
    """A SCIENCE product row must produce a Product with correct fields."""
    raw = _make_product_table([_sample_science_product_row()])
    with patch("astroscope.archive.mast.Observations") as mock_obs:
        mock_obs.get_product_list.return_value = raw
        adapter = HubbleAdapter()
        products = adapter.get_products(24854354)

    assert len(products) == 1
    prod = products[0]
    assert isinstance(prod, Product)
    assert prod.product_type == "SCIENCE"
    assert prod.collection == "HST"
    assert prod.calib_level == 3
    assert prod.data_uri == "mast:HST/product/n4eya1020_mos.fits"
    assert prod.filename == "n4eya1020_mos.fits"
    assert prod.filters == "F160W"
    assert prod.parent_obsid == 24854354


def test_product_normalisation_preview() -> None:
    """A PREVIEW product row must produce a Product with PREVIEW type."""
    row = _sample_science_product_row(
        productFilename="n4eya1020_mos.jpg",
        productType="PREVIEW",
        dataURI="mast:HST/product/n4eya1020_mos.jpg",
        calib_level=2,
    )
    raw = _make_product_table([row])
    with patch("astroscope.archive.mast.Observations") as mock_obs:
        mock_obs.get_product_list.return_value = raw
        adapter = HubbleAdapter()
        products = adapter.get_products(24854354)

    assert len(products) == 1
    assert products[0].product_type == "PREVIEW"


def test_product_normalisation_auxiliary() -> None:
    """An AUXILIARY product row must produce a Product with AUXILIARY type."""
    row = _sample_science_product_row(
        productFilename="n4eya1pzq_jif.fits",
        productType="AUXILIARY",
        dataURI="mast:HST/product/n4eya1pzq_jif.fits",
    )
    raw = _make_product_table([row])
    with patch("astroscope.archive.mast.Observations") as mock_obs:
        mock_obs.get_product_list.return_value = raw
        adapter = HubbleAdapter()
        products = adapter.get_products(24854354)

    assert len(products) == 1
    assert products[0].product_type == "AUXILIARY"


# ---------------------------------------------------------------------------
# 11. Calibration level preservation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("level", [1, 2, 3, 4])
def test_product_calib_level_preserved(level: int) -> None:
    """calib_level values 1–4 must all be preserved correctly."""
    row = _sample_science_product_row(calib_level=level)
    raw = _make_product_table([row])
    with patch("astroscope.archive.mast.Observations") as mock_obs:
        mock_obs.get_product_list.return_value = raw
        adapter = HubbleAdapter()
        products = adapter.get_products(24854354)

    assert products[0].calib_level == level


# ---------------------------------------------------------------------------
# 12. HLA vs HST collection preservation
# ---------------------------------------------------------------------------

def test_product_hst_collection_preserved() -> None:
    """HST collection label must be preserved in the Product."""
    raw = _make_product_table([_sample_science_product_row(obs_collection="HST")])
    with patch("astroscope.archive.mast.Observations") as mock_obs:
        mock_obs.get_product_list.return_value = raw
        adapter = HubbleAdapter()
        products = adapter.get_products(24854354)

    assert products[0].collection == "HST"


def test_product_hla_collection_preserved() -> None:
    """HLA collection label must be preserved separately from HST."""
    row = _sample_science_product_row(
        obs_collection="HLA",
        productFilename="HST_7171_a1_NIC_NIC2_F160W_drz.fits",
        dataURI="mast:HLA/product/HST_7171_a1_NIC_NIC2_F160W_drz.fits",
        calib_level=4,
    )
    raw = _make_product_table([row])
    with patch("astroscope.archive.mast.Observations") as mock_obs:
        mock_obs.get_product_list.return_value = raw
        adapter = HubbleAdapter()
        products = adapter.get_products(24854354)

    assert products[0].collection == "HLA"
    assert products[0].calib_level == 4


# ---------------------------------------------------------------------------
# 13. Product URI preservation
# ---------------------------------------------------------------------------

def test_product_data_uri_preserved() -> None:
    """data_uri must be stored exactly as provided by MAST."""
    uri = "mast:HST/product/n4eya1pzq_raw.fits"
    row = _sample_science_product_row(dataURI=uri, productFilename="n4eya1pzq_raw.fits")
    raw = _make_product_table([row])
    with patch("astroscope.archive.mast.Observations") as mock_obs:
        mock_obs.get_product_list.return_value = raw
        adapter = HubbleAdapter()
        products = adapter.get_products(24854354)

    assert products[0].data_uri == uri


# ---------------------------------------------------------------------------
# 14. get_data_product_uris compat: returns SCIENCE URIs only
# ---------------------------------------------------------------------------

def test_get_data_product_uris_returns_science_uris_only() -> None:
    """get_data_product_uris must return only SCIENCE product URIs."""
    # Build product table with one SCIENCE and one PREVIEW
    science_row = _sample_science_product_row(
        productType="SCIENCE",
        dataURI="mast:HST/product/n4eya1020_mos.fits",
    )
    preview_row = _sample_science_product_row(
        productFilename="n4eya1020_mos.jpg",
        productType="PREVIEW",
        dataURI="mast:HST/product/n4eya1020_mos.jpg",
    )
    products_table = _make_product_table([science_row, preview_row])

    obs_table = _make_obs_table([_sample_obs_row()])

    with patch("astroscope.archive.mast.Observations") as mock_obs:
        mock_obs.query_criteria.return_value = obs_table
        mock_obs.get_product_list.return_value = products_table
        adapter = HubbleAdapter()
        uris = adapter.get_data_product_uris("n4eya1020")

    assert "mast:HST/product/n4eya1020_mos.fits" in uris
    assert "mast:HST/product/n4eya1020_mos.jpg" not in uris


# ---------------------------------------------------------------------------
# 15–16. MastQueryError raised for bad inputs
# ---------------------------------------------------------------------------

def test_mast_query_error_no_criteria() -> None:
    """query_hst_observations must raise MastQueryError when no criteria given."""
    with pytest.raises(MastQueryError, match="At least one"):
        query_hst_observations()


def test_mast_query_error_partial_coordinates() -> None:
    """Providing only RA without Dec must raise ValueError."""
    with pytest.raises(ValueError, match="both be provided"):
        query_hst_observations(right_ascension=10.6848)


def test_mast_query_error_partial_coordinates_dec_only() -> None:
    """Providing only Dec without RA must raise ValueError."""
    with pytest.raises(ValueError, match="both be provided"):
        query_hst_observations(declination=41.2697)


# ---------------------------------------------------------------------------
# 17. get_data_product_uris: missing mast_obsid raises MastQueryError
# ---------------------------------------------------------------------------

def test_get_data_product_uris_missing_mast_obsid() -> None:
    """If the normalised Observation has no mast_obsid, raise MastQueryError."""
    # Build an obs table with obsid as masked/empty
    obs_row = _sample_obs_row()
    raw = _make_obs_table([obs_row])
    raw["obsid"] = MaskedColumn([np.ma.masked], name="obsid")

    with patch("astroscope.archive.mast.Observations") as mock_obs:
        mock_obs.query_criteria.return_value = raw
        adapter = HubbleAdapter()
        with pytest.raises(MastQueryError, match="obsid"):
            adapter.get_data_product_uris("n4eya1020")


# ---------------------------------------------------------------------------
# 18. Mixed product table: all types returned faithfully
# ---------------------------------------------------------------------------

def test_get_products_returns_all_types() -> None:
    """get_products must return ALL product types, not just SCIENCE."""
    rows = [
        _sample_science_product_row(productType="SCIENCE"),
        _sample_science_product_row(
            productFilename="preview.jpg", productType="PREVIEW",
            dataURI="mast:HST/product/preview.jpg"
        ),
        _sample_science_product_row(
            productFilename="aux.fits", productType="AUXILIARY",
            dataURI="mast:HST/product/aux.fits"
        ),
    ]
    raw = _make_product_table(rows)
    with patch("astroscope.archive.mast.Observations") as mock_obs:
        mock_obs.get_product_list.return_value = raw
        adapter = HubbleAdapter()
        products = adapter.get_products(24854354)

    assert len(products) == 3
    types = {p.product_type for p in products}
    assert types == {"SCIENCE", "PREVIEW", "AUXILIARY"}


# ---------------------------------------------------------------------------
# 19. MAST network failure propagates as MastQueryError
# ---------------------------------------------------------------------------

def test_mast_network_failure_raises_mast_query_error() -> None:
    """A simulated network failure in astroquery must raise MastQueryError."""
    with patch("astroscope.archive.mast.Observations") as mock_obs:
        mock_obs.query_criteria.side_effect = ConnectionError("simulated network failure")
        with pytest.raises(MastQueryError, match="MAST HST observation query failed"):
            query_hst_observations(target_name="M31")


# ---------------------------------------------------------------------------
# 20. Limit is applied
# ---------------------------------------------------------------------------

def test_search_limit_applied() -> None:
    """search_observations must respect the limit parameter."""
    rows = [_sample_obs_row(obs_id=f"obs{i:04d}", obsid=i) for i in range(10)]
    raw = _make_obs_table(rows)
    with patch("astroscope.archive.mast.Observations") as mock_obs:
        mock_obs.query_criteria.return_value = raw
        adapter = HubbleAdapter()
        results = adapter.search_observations(target_name="M31", limit=3)

    assert len(results) <= 3


# ---------------------------------------------------------------------------
# 21. Regression: integer mast_obsid converted to str for get_product_list
# ---------------------------------------------------------------------------

def test_get_product_list_receives_string_obsid() -> None:
    """Regression: get_product_list must receive a str, not an int/NumPy int.

    astroquery's Observations.get_product_list() requires the observation ID
    to be passed as a string.  Passing a bare Python int or NumPy integer
    causes a live MAST query to fail.  This test asserts that
    ``get_product_list()`` in mast.py calls the astroquery method with
    ``str(mast_obsid)`` rather than the raw integer.
    """
    raw = _make_product_table([_sample_science_product_row()])
    with patch("astroscope.archive.mast.Observations") as mock_obs:
        mock_obs.get_product_list.return_value = raw
        from astroscope.archive.mast import get_product_list
        get_product_list(24854354)

    # The mock must have been called with a str, not an int
    call_args = mock_obs.get_product_list.call_args
    assert call_args is not None, "get_product_list was never called"
    passed_arg = call_args.args[0] if call_args.args else call_args.kwargs.get("obsids")
    assert isinstance(passed_arg, str), (
        f"Expected str obsid, got {type(passed_arg).__name__!r}: {passed_arg!r}"
    )
    assert passed_arg == "24854354"
