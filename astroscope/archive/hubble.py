"""Hubble Space Telescope (HST) archive adapter."""

from typing import List, Optional

from astroscope.archive.base import ArchiveAdapter
from astroscope.observation import Observation


class HubbleAdapter(ArchiveAdapter):
    """Archive adapter for Hubble Space Telescope (HST) observations.

    Establishes the structural entry point for MAST queries and Hubble-specific
    data product references.
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
        """Search HST archive for observations matching search parameters.

        Note: Session 1 structural stub. MAST network queries are intentionally omitted.
        """
        raise NotImplementedError(
            "Hubble search_observations network ingestion is not implemented in Session 1."
        )

    def get_observation_metadata(self, observation_id: str) -> Observation:
        """Retrieve Observation metadata for a Hubble dataset ID.

        Note: Session 1 structural stub. MAST network queries are intentionally omitted.
        """
        raise NotImplementedError(
            "Hubble get_observation_metadata network ingestion is not implemented in Session 1."
        )

    def get_data_product_uris(self, observation_id: str) -> List[str]:
        """Retrieve data product URIs for a Hubble observation ID.

        Note: Session 1 structural stub. MAST network queries are intentionally omitted.
        """
        raise NotImplementedError(
            "Hubble get_data_product_uris network ingestion is not implemented in Session 1."
        )
