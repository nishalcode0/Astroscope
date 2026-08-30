"""James Webb Space Telescope (JWST) archive adapter.

Session 2 note: JWST networking is intentionally not implemented in this
session.  The adapter retains structural stubs to satisfy the
:class:`~astroscope.archive.base.ArchiveAdapter` interface.  The new
``get_products()`` method is stubbed alongside the existing methods.

JWST archive integration is planned for a future session.
"""

from typing import List, Optional

from astroscope.archive.base import ArchiveAdapter
from astroscope.observation import Observation
from astroscope.product import Product


class JWSTAdapter(ArchiveAdapter):
    """Archive adapter for James Webb Space Telescope (JWST) observations.

    Establishes the structural entry point for MAST queries and JWST-specific
    data product references.  All methods are stubs pending Session 3+.
    """

    @property
    def mission_name(self) -> str:
        """Return the canonical mission name for JWST."""
        return "James Webb Space Telescope"

    def search_observations(
        self,
        target_name: Optional[str] = None,
        right_ascension: Optional[float] = None,
        declination: Optional[float] = None,
        radius_arcmin: float = 1.0,
        limit: int = 50,
    ) -> List[Observation]:
        """Search JWST archive for observations matching search parameters.

        Note: Structural stub. JWST MAST networking is not implemented in Session 2.
        """
        raise NotImplementedError(
            "JWST search_observations is not implemented yet. "
            "JWST archive integration is planned for a future session."
        )

    def get_observation_metadata(self, observation_id: str) -> Observation:
        """Retrieve Observation metadata for a JWST dataset ID.

        Note: Structural stub. JWST MAST networking is not implemented in Session 2.
        """
        raise NotImplementedError(
            "JWST get_observation_metadata is not implemented yet. "
            "JWST archive integration is planned for a future session."
        )

    def get_products(self, mast_obsid: int) -> List[Product]:
        """Retrieve the product inventory for a JWST MAST observation.

        Note: Structural stub. JWST MAST networking is not implemented in Session 2.
        """
        raise NotImplementedError(
            "JWST get_products is not implemented yet. "
            "JWST archive integration is planned for a future session."
        )

    def get_data_product_uris(self, observation_id: str) -> List[str]:
        """Retrieve data product URIs for a JWST observation ID.

        Note: Structural stub. JWST MAST networking is not implemented in Session 2.
        """
        raise NotImplementedError(
            "JWST get_data_product_uris is not implemented yet. "
            "JWST archive integration is planned for a future session."
        )
