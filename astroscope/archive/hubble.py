"""Hubble Space Telescope (HST) archive adapter — MAST implementation.

This module provides the real HST archive integration via MAST/astroquery.
It covers the **archive layer** only:

- Discovering observations (``search_observations``)
- Querying metadata for a known observation (``get_observation_metadata``)
- Listing the full product inventory (``get_products``)
- Providing backwards-compatible data-URI access (``get_data_product_uris``)

No data is downloaded here.  No files are opened.  No FITS headers are parsed.
These responsibilities belong to the ingestion and processing layers.

MAST Identifiers
----------------
MAST uses two distinct observation identifiers:

- **obs_id** (``Observation.observation_id``): The mission/instrument identifier
  assigned by the telescope team, e.g. ``"n4eya1020"``.
- **obsid** (``Observation.mast_obsid``): The MAST internal integer key used
  to retrieve the product list via ``get_product_list(obsid)``.

These must not be conflated.  Both are preserved in every :class:`Observation`
returned by this adapter.

Product Types
-------------
MAST product tables include ``SCIENCE``, ``PREVIEW``, and ``AUXILIARY`` entries.
``get_products()`` returns all of them faithfully; it does not filter or
automatically select a preferred product.  Selection is deferred to the
ingestion layer.
"""

import logging
import math
from typing import Any, Dict, List, Optional

from astropy.table import MaskedColumn, Row, Table

from astroscope.archive.base import ArchiveAdapter
from astroscope.archive.mast import (
    MastQueryError,
    get_product_list,
    query_hst_observation_by_id,
    query_hst_observations,
)
from astroscope.observation import Observation
from astroscope.product import Product

log = logging.getLogger(__name__)

# MAST column names we care about for Observation normalisation
_OBS_COLUMNS = {
    "obs_id",
    "obsid",
    "target_name",
    "s_ra",
    "s_dec",
    "instrument_name",
    "filters",
    "t_min",
    "t_max",
    "t_exptime",
    "t_obs_release",
    "calib_level",
    "dataRights",
    "obs_title",
    "proposal_id",
    "target_classification",
    "wave_region",
    "provenance_name",
    "dataproduct_type",
    "obs_collection",
}

# MAST column names for Product normalisation
_PROD_COLUMNS = {
    "productFilename",
    "obsID",
    "obs_collection",
    "productType",
    "dataproduct_type",
    "calib_level",
    "filters",
    "description",
    "dataURI",
    "size",
    "dataRights",
    "productGroupDescription",
    "productSubGroupDescription",
}


# ---------------------------------------------------------------------------
# Private normalisation helpers
# ---------------------------------------------------------------------------

def _safe_str(value: Any, default: str = "") -> str:
    """Convert a potentially masked or NaN value to a plain string.

    Masked column values in Astropy tables appear as ``np.ma.masked``
    (evaluating as truthy but stringifying to ``"--"``).  This helper
    collapses all falsy / masked / empty representations to ``default``.
    """
    if value is None:
        return default
    # Astropy masked scalar: str(masked) == "--"
    str_val = str(value)
    if str_val in ("--", "nan", "None", ""):
        return default
    try:
        import numpy as np
        if np.ma.is_masked(value):
            return default
    except Exception:
        pass
    return str_val.strip() or default


def _safe_float(value: Any) -> Optional[float]:
    """Convert a potentially masked/NaN column value to float or None."""
    if value is None:
        return None
    try:
        import numpy as np
        if np.ma.is_masked(value):
            return None
    except Exception:
        pass
    try:
        f = float(value)
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> Optional[int]:
    """Convert a potentially masked/NaN column value to int or None."""
    f = _safe_float(value)
    return None if f is None else int(f)


def _row_to_observation(row: Any) -> Optional[Observation]:
    """Normalise one MAST table row into an :class:`Observation`.

    Returns ``None`` and logs a warning if the row cannot be normalised
    (e.g. missing required RA/Dec values).

    Parameters
    ----------
    row:
        An Astropy ``Table`` row (or any mapping with the expected column names).

    Returns
    -------
    Observation or None
    """
    obs_id = _safe_str(row.get("obs_id") if hasattr(row, "get") else row["obs_id"] if "obs_id" in row.colnames else None)
    if not obs_id:
        log.warning("Skipping MAST row: obs_id is empty or masked")
        return None

    ra = _safe_float(row["s_ra"] if "s_ra" in row.colnames else None)
    dec = _safe_float(row["s_dec"] if "s_dec" in row.colnames else None)
    if ra is None or dec is None:
        log.warning("Skipping MAST row obs_id=%r: RA or Dec is missing/masked", obs_id)
        return None

    # Clamp to valid ranges to handle floating-point edge cases from MAST
    ra = max(0.0, min(360.0, ra))
    dec = max(-90.0, min(90.0, dec))

    def col(name: str, default: str = "") -> str:
        try:
            return _safe_str(row[name], default)
        except (KeyError, IndexError):
            return default

    def fcol(name: str) -> Optional[float]:
        try:
            return _safe_float(row[name])
        except (KeyError, IndexError):
            return None

    def icol(name: str) -> Optional[int]:
        try:
            return _safe_int(row[name])
        except (KeyError, IndexError):
            return None

    mast_obsid_val = icol("obsid")

    # Build extra_metadata from fields that don't map to core attributes
    extra: Dict[str, Any] = {}
    for key in ("obs_title", "proposal_id", "target_classification", "wave_region",
                "provenance_name", "dataproduct_type", "obs_collection"):
        val = col(key)
        if val:
            extra[key] = val

    try:
        return Observation(
            mission="HST",
            observation_id=obs_id,
            target_name=col("target_name", default="UNKNOWN"),
            right_ascension=ra,
            declination=dec,
            instrument=col("instrument_name", default="UNKNOWN"),
            filter_band=col("filters", default="UNKNOWN"),
            mast_obsid=mast_obsid_val,
            t_min=fcol("t_min"),
            t_max=fcol("t_max"),
            t_exptime=fcol("t_exptime"),
            t_obs_release=fcol("t_obs_release"),
            calib_level=icol("calib_level"),
            data_rights=col("dataRights", default="PUBLIC"),
            extra_metadata=extra,
        )
    except ValueError as exc:
        log.warning("Skipping MAST row obs_id=%r: validation error: %s", obs_id, exc)
        return None


def _row_to_product(row: Any, parent_obsid: int) -> Optional[Product]:
    """Normalise one MAST product table row into a :class:`Product`.

    Returns ``None`` and logs a warning if the row cannot be normalised.

    Parameters
    ----------
    row:
        An Astropy ``Table`` row.
    parent_obsid:
        The MAST obsid of the parent observation (used to populate
        :attr:`Product.parent_obsid`).
    """
    def col(name: str, default: str = "") -> str:
        try:
            return _safe_str(row[name], default)
        except (KeyError, IndexError):
            return default

    def icol(name: str) -> Optional[int]:
        try:
            return _safe_int(row[name])
        except (KeyError, IndexError):
            return None

    filename = col("productFilename")
    data_uri = col("dataURI")
    if not filename and not data_uri:
        log.warning("Skipping MAST product row: both filename and dataURI are empty/masked")
        return None

    product_id = filename or data_uri

    calib_raw = icol("calib_level")
    calib = calib_raw if calib_raw is not None else 0

    file_size = icol("size")

    return Product(
        product_id=product_id,
        parent_obsid=parent_obsid,
        collection=col("obs_collection", default="HST"),
        product_type=col("productType", default="UNKNOWN"),
        data_product_type=col("dataproduct_type", default=""),
        calib_level=calib,
        filters=col("filters", default=""),
        filename=filename,
        data_uri=data_uri,
        file_size=file_size,
        data_rights=col("dataRights", default="PUBLIC"),
        description=col("description", default=""),
        product_group_description=col("productGroupDescription", default=""),
        product_sub_group_description=col("productSubGroupDescription", default=""),
    )


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class HubbleAdapter(ArchiveAdapter):
    """Archive adapter for Hubble Space Telescope (HST) observations.

    Uses MAST/astroquery to discover HST observations and their associated
    data products.  All operations are read-only queries against the public
    MAST archive; no data is downloaded.

    Session 2 implements:
    - :meth:`search_observations` — real MAST query → normalised Observations
    - :meth:`get_observation_metadata` — query by ``obs_id`` → single Observation
    - :meth:`get_products` — product inventory → normalised Products
    - :meth:`get_data_product_uris` — backwards-compatible URI list

    JWST implementation is deferred to a future session.
    """

    @property
    def mission_name(self) -> str:
        """Return the canonical mission name for Hubble."""
        return "Hubble Space Telescope"

    def search_observations(
        self,
        target_name: Optional[str] = None,
        right_ascension: Optional[float] = None,
        declination: Optional[float] = None,
        radius_arcmin: float = 1.0,
        limit: int = 50,
    ) -> List[Observation]:
        """Search the HST MAST archive for observations matching the given criteria.

        Parameters
        ----------
        target_name:
            Celestial target identifier, e.g. ``"M31"`` or ``"NGC 1068"``.
            MAST resolves the name through Simbad/NED internally.
        right_ascension:
            Right Ascension of the search cone centre in degrees (0 – 360).
            Must be supplied together with ``declination``.
        declination:
            Declination of the search cone centre in degrees (−90 – +90).
            Must be supplied together with ``right_ascension``.
        radius_arcmin:
            Cone search radius in arcminutes.  Only used for coordinate searches.
        limit:
            Maximum number of observations to return.

        Returns
        -------
        List[Observation]
            Normalised HST observation metadata objects.  Rows that cannot be
            normalised (masked RA/Dec, missing obs_id) are skipped with a
            logged warning rather than raising an exception.

        Raises
        ------
        MastQueryError
            If the MAST service is unavailable or the query itself fails.
        ValueError
            If only one of ``right_ascension`` / ``declination`` is provided.
        """
        raw_table: Table = query_hst_observations(
            target_name=target_name,
            right_ascension=right_ascension,
            declination=declination,
            radius_arcmin=radius_arcmin,
            limit=limit,
        )

        observations: List[Observation] = []
        for row in raw_table:
            obs = _row_to_observation(row)
            if obs is not None:
                observations.append(obs)

        log.info(
            "HubbleAdapter: normalised %d/%d MAST rows into Observation objects",
            len(observations),
            len(raw_table),
        )
        return observations

    def get_observation_metadata(self, observation_id: str) -> Observation:
        """Retrieve metadata for a specific HST observation by its mission obs_id.

        Parameters
        ----------
        observation_id:
            The mission-assigned observation identifier, e.g. ``"n4eya1020"``.
            This is the ``obs_id`` column value, *not* the MAST integer ``obsid``.

        Returns
        -------
        Observation
            Normalised observation metadata.

        Raises
        ------
        MastQueryError
            If the MAST query fails.
        LookupError
            If no HST observation with the given ``obs_id`` is found in MAST.
        """
        # MAST does not expose a direct obs_id lookup endpoint in astroquery's
        # query_criteria, but we can filter by obs_id after a target-name-free
        # criteria query.  The most reliable approach is to search by obs_id
        # directly as a criteria value.
        raw_table: Table = query_hst_observation_by_id(observation_id)

        if raw_table is None or len(raw_table) == 0:
            raise LookupError(
                f"No HST observation found in MAST for obs_id={observation_id!r}"
            )

        obs = _row_to_observation(raw_table[0])
        if obs is None:
            raise MastQueryError(
                f"Could not normalise MAST row for obs_id={observation_id!r}: "
                "RA/Dec or obs_id is missing or masked in the archive response."
            )
        return obs

    def get_products(self, mast_obsid: int) -> List[Product]:
        """Retrieve the full product inventory for one HST MAST observation.

        Returns all products faithfully as reported by MAST, including SCIENCE,
        PREVIEW, AUXILIARY, and CATALOG entries.  No filtering is applied.

        Parameters
        ----------
        mast_obsid:
            The MAST internal integer observation key (``Observation.mast_obsid``).
            This is *not* the mission ``obs_id`` string.

        Returns
        -------
        List[Product]
            All products associated with the observation.

        Raises
        ------
        MastQueryError
            If the MAST product query fails.

        Notes
        -----
        **Do not assume any particular product is present or preferred.**
        Product selection — choosing between HST vs. HLA, calibration level,
        or file type — is the responsibility of the ingestion layer, which is
        implemented in a future session.
        """
        raw_table: Table = get_product_list(mast_obsid)

        products: List[Product] = []
        for row in raw_table:
            prod = _row_to_product(row, parent_obsid=mast_obsid)
            if prod is not None:
                products.append(prod)

        log.info(
            "HubbleAdapter: normalised %d/%d MAST product rows for obsid=%d",
            len(products),
            len(raw_table),
            mast_obsid,
        )
        return products

    def get_data_product_uris(self, observation_id: str) -> List[str]:
        """Return MAST URIs for SCIENCE products of an HST observation.

        This method is retained for backwards compatibility with the
        :class:`~astroscope.archive.base.ArchiveAdapter` interface.  New code
        should call :meth:`get_products` directly to obtain structured
        :class:`~astroscope.product.Product` objects with full metadata.

        The implementation first resolves the MAST ``obsid`` by querying for
        ``observation_id``, then retrieves all products and returns the
        ``data_uri`` of ``SCIENCE``-type entries only.

        Parameters
        ----------
        observation_id:
            Mission ``obs_id`` string, e.g. ``"n4eya1020"``.

        Returns
        -------
        List[str]
            MAST URI strings for SCIENCE products only.

        Raises
        ------
        MastQueryError
            If the MAST query fails.
        LookupError
            If no HST observation with the given ``obs_id`` is found.
        """
        obs = self.get_observation_metadata(observation_id)
        if obs.mast_obsid is None:
            raise MastQueryError(
                f"Could not determine MAST obsid for obs_id={observation_id!r}: "
                "the archive returned no obsid for this observation."
            )

        products = self.get_products(obs.mast_obsid)
        uris = [p.data_uri for p in products if p.product_type == "SCIENCE" and p.data_uri]
        log.info(
            "get_data_product_uris: %d SCIENCE URIs found for obs_id=%r (obsid=%d)",
            len(uris),
            observation_id,
            obs.mast_obsid,
        )
        return uris
