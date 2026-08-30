"""Scientific representation of astronomical observation metadata."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple


@dataclass(frozen=True)
class Observation:
    """Typed metadata representation of an astronomical observation.

    Represents observational metadata (e.g. target, coordinates, instrument parameters,
    and references to scientific data products) without loading image pixel payloads.

    MAST Time Fields
    ----------------
    MAST expresses times as Modified Julian Days (MJD), not Unix timestamps.
    The fields ``t_min``, ``t_max``, ``t_obs_release`` are MJD floats.
    Use :attr:`observation_time` for a convenience UTC datetime derived from ``t_min``.

    MAST Identifiers
    ----------------
    MAST distinguishes between two observation identifiers:
    - ``observation_id``: The mission/instrument identifier assigned by the telescope
      team (e.g. ``n4eya1020`` for HST, or a JWST programme + visit string).
    - ``mast_obsid``: The internal MAST integer key (e.g. 24854354) used when
      calling ``Observations.get_product_list()``.

    Both are preserved so that downstream code can interact with MAST without
    having to re-query for the numeric key.
    """

    # ------------------------------------------------------------------ #
    # Core identification                                                  #
    # ------------------------------------------------------------------ #
    mission: str
    """Canonical mission label, e.g. ``"HST"`` or ``"JWST"``."""

    observation_id: str
    """Mission-assigned observation identifier, e.g. ``"n4eya1020"``."""

    target_name: str
    """Human-readable target name, e.g. ``"M31"``."""

    right_ascension: float
    """Right Ascension of the pointing centre, degrees (0 – 360)."""

    declination: float
    """Declination of the pointing centre, degrees (−90 – +90)."""

    instrument: str
    """Instrument and detector name, e.g. ``"NICMOS/NIC2"``."""

    filter_band: str
    """Primary filter identifier, e.g. ``"F160W"``."""

    # ------------------------------------------------------------------ #
    # Backwards-compatible time field                                      #
    # ------------------------------------------------------------------ #
    observation_time: Optional[datetime] = None
    """UTC start time of the observation (for backwards compatibility).

    When an ``Observation`` is created from a MAST archive row the value is
    derived from :attr:`t_min` (MJD).  Pre-Session-2 code that passes a
    ``datetime`` directly continues to work unchanged.
    """

    # ------------------------------------------------------------------ #
    # MAST time fields (MJD)                                              #
    # ------------------------------------------------------------------ #
    t_min: Optional[float] = None
    """Observation start time in Modified Julian Days (MJD)."""

    t_max: Optional[float] = None
    """Observation end time in Modified Julian Days (MJD)."""

    t_exptime: Optional[float] = None
    """Total exposure time in seconds."""

    t_obs_release: Optional[float] = None
    """Public data-release date in Modified Julian Days (MJD)."""

    # ------------------------------------------------------------------ #
    # Archive / product metadata                                          #
    # ------------------------------------------------------------------ #
    mast_obsid: Optional[int] = None
    """MAST internal observation key used to retrieve the product list.

    Example: ``24854354``.  Distinct from :attr:`observation_id`.
    """

    calib_level: Optional[int] = None
    """Calibration level of the observation (0 = raw, 1–4 = increasingly processed)."""

    data_rights: str = "PUBLIC"
    """Data access rights: ``"PUBLIC"`` or ``"RESTRICTED"``."""

    # ------------------------------------------------------------------ #
    # Linked data products and extra fields                               #
    # ------------------------------------------------------------------ #
    data_products: Tuple[str, ...] = field(default_factory=tuple)
    """Tuple of MAST URIs referencing the raw or calibrated data products.

    Populated during archive normalisation.  For the full structured product
    inventory use ``HubbleAdapter.get_products(mast_obsid)``.
    """

    extra_metadata: Dict[str, Any] = field(default_factory=dict)
    """Archive-specific fields that do not map to a first-class attribute.

    Typical keys from MAST include ``"obs_title"``, ``"proposal_id"``,
    ``"target_classification"``, ``"wave_region"``, ``"provenance_name"``.
    """

    # ------------------------------------------------------------------ #
    # Validation                                                          #
    # ------------------------------------------------------------------ #
    def __post_init__(self) -> None:
        """Validate metadata boundaries upon initialization."""
        if not self.mission or not self.mission.strip():
            raise ValueError("mission must be a non-empty string")
        if not self.observation_id or not self.observation_id.strip():
            raise ValueError("observation_id must be a non-empty string")
        if not (0.0 <= self.right_ascension <= 360.0):
            raise ValueError(
                f"right_ascension must be between 0 and 360 degrees, got {self.right_ascension}"
            )
        if not (-90.0 <= self.declination <= 90.0):
            raise ValueError(
                f"declination must be between -90 and 90 degrees, got {self.declination}"
            )

    # ------------------------------------------------------------------ #
    # Convenience helpers                                                 #
    # ------------------------------------------------------------------ #
    def observation_start_datetime(self) -> Optional[datetime]:
        """Return the observation start as a UTC ``datetime``, if available.

        Prefers :attr:`observation_time` if set directly.  Falls back to
        converting :attr:`t_min` from MJD using Astropy.  Returns ``None``
        if neither is available.

        Note: MJD values are *not* Unix timestamps.  The MJD epoch is
        1858-11-17 00:00 UTC; Unix epoch is 1970-01-01 00:00 UTC.
        """
        if self.observation_time is not None:
            return self.observation_time
        if self.t_min is not None:
            try:
                from astropy.time import Time as AstropyTime  # local import to avoid hard dep at module level
                return AstropyTime(self.t_min, format="mjd", scale="utc").to_datetime(timezone=timezone.utc)
            except Exception:
                return None
        return None
