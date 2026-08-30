"""Scientific representation of a single MAST archive data product."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class Product:
    """Metadata for one data product entry from the MAST archive.

    A single :class:`~astroscope.observation.Observation` can have many
    associated products (raw, calibrated, preview images, auxiliary files, etc.).
    This class represents one row from the MAST product table, faithfully
    preserving the inventory without selecting or prioritising any product.

    Identifiers
    -----------
    ``parent_obsid`` is the MAST integer observation key (``Observation.mast_obsid``).
    ``product_id`` is constructed from the archive filename and is used as a
    stable local identifier for this product entry.

    Collections
    -----------
    MAST returns products from two distinct collections for HST data:

    - ``"HST"`` — pipeline-calibrated products produced by STScI.
    - ``"HLA"`` — Hubble Legacy Archive higher-level mosaics and drizzle products.

    These are *not* interchangeable: HLA products may cover a larger area or
    use different astrometric solutions.  The ``collection`` field preserves
    which collection a product belongs to.

    Product Types
    -------------
    The ``product_type`` field reflects the MAST ``productType`` column:

    - ``"SCIENCE"`` — a science-grade data product (e.g. ``.fits``).
    - ``"PREVIEW"`` — a quick-look JPEG or small FITS cutout.
    - ``"AUXILIARY"`` — supporting files (jitter, trailer, association tables).
    - ``"CATALOG"`` — source or photometry catalogues.

    Only ``"SCIENCE"`` products are appropriate for photometric or spectroscopic
    analysis.  ``"PREVIEW"`` and ``"AUXILIARY"`` products must not be treated as
    equivalent to science-grade data.

    Calibration Levels
    ------------------
    The ``calib_level`` field reflects the MAST ``calib_level`` column:

    - 0 — raw telemetry
    - 1 — raw uncalibrated (e.g. ``_raw.fits``)
    - 2 — calibrated single exposure (e.g. ``_cal.fits``, ``_flt.fits``)
    - 3 — combined/mosaiced product (e.g. ``_drz.fits``, ``_mos.fits``)
    - 4 — enhanced / HLA-level product

    Downloading
    -----------
    This class does *not* download files.  The ``data_uri`` field holds the
    MAST URI (e.g. ``mast:HST/product/n4eya1020_mos.fits``) that the ingestion
    layer will use in a future session.
    """

    # ------------------------------------------------------------------ #
    # Core identification                                                  #
    # ------------------------------------------------------------------ #
    product_id: str
    """Stable identifier for this product, typically the bare filename."""

    parent_obsid: int
    """MAST obsid of the parent observation (used with get_product_list)."""

    # ------------------------------------------------------------------ #
    # Collection and typing                                               #
    # ------------------------------------------------------------------ #
    collection: str
    """Archive collection: ``"HST"`` or ``"HLA"`` (or ``"JWST"`` for future use)."""

    product_type: str
    """MAST product type: ``"SCIENCE"``, ``"PREVIEW"``, ``"AUXILIARY"``, ``"CATALOG"``."""

    data_product_type: str
    """Data dimensionality type from MAST: ``"image"``, ``"spectrum"``, ``"timeseries"``, etc."""

    calib_level: int
    """Calibration level 0–4 (see class docstring for definitions)."""

    # ------------------------------------------------------------------ #
    # Instrument / filter                                                  #
    # ------------------------------------------------------------------ #
    filters: str
    """Filter name(s) as reported by MAST, e.g. ``"F160W"`` or ``"CLEAR"``."""

    # ------------------------------------------------------------------ #
    # File references                                                      #
    # ------------------------------------------------------------------ #
    filename: str
    """Bare filename of the product, e.g. ``"n4eya1020_mos.fits"``."""

    data_uri: str
    """MAST URI for the product, e.g. ``"mast:HST/product/n4eya1020_mos.fits"``.

    This URI is passed to the MAST download service by the ingestion layer.
    The product is NOT downloaded by this class.
    """

    # ------------------------------------------------------------------ #
    # Size and access                                                      #
    # ------------------------------------------------------------------ #
    file_size: Optional[int] = None
    """File size in bytes as reported by MAST.  May be ``None`` if unavailable."""

    data_rights: str = "PUBLIC"
    """Data access rights: ``"PUBLIC"`` or ``"RESTRICTED"``."""

    # ------------------------------------------------------------------ #
    # Human-readable descriptions                                         #
    # ------------------------------------------------------------------ #
    description: str = ""
    """Product description string from the MAST product table."""

    product_group_description: str = ""
    """MAST product group description (e.g. ``"Minimum Recommended Products"``)."""

    product_sub_group_description: str = ""
    """MAST product sub-group description (e.g. ``"SCIENCE"`` within a group)."""

    # ------------------------------------------------------------------ #
    # Archive-specific overflow                                           #
    # ------------------------------------------------------------------ #
    extra_metadata: Dict[str, Any] = field(default_factory=dict)
    """Additional archive-specific fields not mapped to a first-class attribute."""
