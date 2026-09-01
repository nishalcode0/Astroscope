"""Shared MAST/astroquery wrapper used by mission-specific archive adapters.

This module encapsulates all calls to ``astroquery.mast.Observations`` and
provides a single point of failure for network-related issues.  Mission
adapters (Hubble, JWST) import from here rather than calling astroquery
directly, keeping MAST mechanics out of mission-specific normalisation logic.

Architecture
------------
This module sits between astroquery and the mission adapters::

    astroquery.mast.Observations
           ↓
    astroscope.archive.mast  ← this module
           ↓
    HubbleAdapter / JWSTAdapter

Import note
-----------
``astroquery.mast.Observations`` is imported at module level so that tests can
patch it at ``astroscope.archive.mast.Observations`` without needing to reach
into the astroquery package namespace.

Only the archive *query* is performed here.  Normalisation into Astroscope
model objects is the responsibility of each mission adapter.

Exceptions
----------
All network and API failures are raised as :class:`MastQueryError`, which
wraps the underlying exception as a cause.  Callers should not catch the
raw astroquery or urllib exceptions.
"""

import logging
from typing import Any, Dict, Optional

from astropy.table import Table

try:
    from astroquery.mast import Observations
except ImportError as _astroquery_import_error:  # pragma: no cover
    raise ImportError(
        "astroquery is required for MAST archive access. "
        "Install it with: pip install astroquery"
    ) from _astroquery_import_error

log = logging.getLogger(__name__)


class MastQueryError(Exception):
    """Raised when a MAST archive query fails.

    Wraps lower-level exceptions (network errors, astroquery errors, malformed
    responses) with a descriptive message.  Inspect ``__cause__`` for the
    original exception.

    Examples
    --------
    >>> raise MastQueryError("MAST unreachable") from original_exc
    """


def query_hst_observations(
    target_name: Optional[str] = None,
    right_ascension: Optional[float] = None,
    declination: Optional[float] = None,
    radius_arcmin: float = 1.0,
    limit: int = 50,
) -> Table:
    """Query MAST for HST observations matching the given criteria.

    Parameters
    ----------
    target_name:
        Celestial target identifier (e.g. ``"M31"``, ``"NGC 1068"``).
        When provided alone, MAST resolves the name to coordinates via
        Simbad/NED and searches within a default cone radius.
    right_ascension:
        Right Ascension of the search cone centre in degrees.
        Must be provided together with ``declination`` and ``radius_arcmin``.
    declination:
        Declination of the search cone centre in degrees.
        Must be provided together with ``right_ascension`` and ``radius_arcmin``.
    radius_arcmin:
        Cone search radius in arcminutes.  Only used when ``right_ascension``
        and ``declination`` are both provided.
    limit:
        Maximum number of rows to return.

    Returns
    -------
    astropy.table.Table
        Raw MAST result table.  May have zero rows.  The caller is responsible
        for normalising rows into Astroscope model objects.

    Raises
    ------
    MastQueryError
        If no search criteria are provided, if the MAST service is unavailable,
        or if astroquery raises any exception during the query.
    ValueError
        If ``right_ascension`` and ``declination`` are partially provided
        (i.e. one is set but the other is not).

    Notes
    -----
    The ``radius_arcmin`` parameter is converted to degrees for the MAST
    cone-search API (``s_ra``, ``s_dec`` with ``radius`` keyword).
    When only ``target_name`` is supplied, MAST performs its own name
    resolution and the ``radius_arcmin`` parameter has no effect.
    """
    if target_name is None and right_ascension is None and declination is None:
        raise MastQueryError(
            "At least one of target_name or (right_ascension, declination) must be provided."
        )

    # Validate coordinate pair completeness
    coord_provided = (right_ascension is not None, declination is not None)
    if any(coord_provided) and not all(coord_provided):
        raise ValueError(
            "right_ascension and declination must both be provided for a coordinate search."
        )

    try:
        criteria: Dict[str, Any] = {"obs_collection": "HST"}

        if target_name is not None:
            log.info("MAST HST query: target_name=%r limit=%d", target_name, limit)
            criteria["target_name"] = target_name
            result: Table = Observations.query_criteria(**criteria)
        else:
            # Coordinate cone search
            radius_deg = radius_arcmin / 60.0
            log.info(
                "MAST HST cone search: ra=%.4f dec=%.4f radius_arcmin=%.2f limit=%d",
                right_ascension,
                declination,
                radius_arcmin,
                limit,
            )
            result = Observations.query_criteria(
                s_ra=[right_ascension - radius_deg, right_ascension + radius_deg],
                s_dec=[declination - radius_deg, declination + radius_deg],
                **criteria,
            )

        row_count = len(result) if result is not None else 0
        log.info("MAST HST query returned %d observation(s)", row_count)

        # Apply limit (MAST does not expose a server-side row limit in query_criteria)
        if result is not None and len(result) > limit:
            result = result[:limit]

        return result if result is not None else Table()

    except MastQueryError:
        raise
    except Exception as exc:
        raise MastQueryError(
            f"MAST HST observation query failed: {exc}"
        ) from exc


def get_product_list(mast_obsid: int) -> Table:
    """Retrieve the full product inventory for one MAST observation.

    Parameters
    ----------
    mast_obsid:
        The integer MAST observation key (``Observation.mast_obsid``).
        This is the ``obsid`` column returned by MAST queries, *not* the
        mission ``obs_id`` string.

    Returns
    -------
    astropy.table.Table
        Raw product table from MAST.  May have zero rows if no products are
        associated with the observation.  The caller is responsible for
        normalising rows into :class:`~astroscope.product.Product` objects.

    Raises
    ------
    MastQueryError
        If the product lookup fails for any reason (network, bad obsid, etc.).

    Notes
    -----
    This wraps ``astroquery.mast.Observations.get_product_list()``.  MAST
    returns *all* product types (SCIENCE, PREVIEW, AUXILIARY, CATALOG).
    Filtering by product type is intentionally left to the caller.
    """
    log.info("MAST product query: mast_obsid=%d", mast_obsid)

    try:
        result: Table = Observations.get_product_list(str(mast_obsid))
        product_count = len(result) if result is not None else 0
        log.info("MAST product query returned %d product(s) for obsid=%d", product_count, mast_obsid)
        return result if result is not None else Table()
    except MastQueryError:
        raise
    except Exception as exc:
        raise MastQueryError(
            f"MAST product list query failed for obsid={mast_obsid}: {exc}"
        ) from exc


def query_hst_observation_by_id(obs_id: str) -> Table:
    """Query MAST for a single HST observation by its mission obs_id.

    Parameters
    ----------
    obs_id:
        The mission-assigned observation identifier (e.g. ``"n4eya1020"``).
        This is the ``obs_id`` column value from MAST, *not* the integer
        ``obsid`` used for product lookups.

    Returns
    -------
    astropy.table.Table
        Raw MAST result table.  May have zero rows if the observation is not
        found.

    Raises
    ------
    MastQueryError
        If the MAST query fails for any reason.
    """
    log.info("MAST HST metadata query: obs_id=%r", obs_id)
    try:
        result: Table = Observations.query_criteria(
            obs_collection="HST",
            obs_id=obs_id,
        )
        return result if result is not None else Table()
    except MastQueryError:
        raise
    except Exception as exc:
        raise MastQueryError(
            f"MAST get_observation_metadata query failed for obs_id={obs_id!r}: {exc}"
        ) from exc


def download_file(data_uri: str, local_path: str) -> None:
    """Download a single MAST product by its URI.

    Parameters
    ----------
    data_uri:
        The MAST product URI (e.g. ``"mast:HST/product/n4eya1020_mos.fits"``).
    local_path:
        The local filesystem path where the file should be saved.

    Raises
    ------
    MastQueryError
        If the download fails due to network or MAST issues.
    """
    log.info("MAST download: uri=%r -> %r", data_uri, local_path)
    try:
        Observations.download_file(data_uri, local_path=local_path)
    except MastQueryError:
        raise
    except Exception as exc:
        raise MastQueryError(
            f"MAST download failed for uri={data_uri!r}: {exc}"
        ) from exc
