"""Base abstract interface for astronomical archive adapters."""

from abc import ABC, abstractmethod
from typing import List, Optional

from astroscope.observation import Observation


class ArchiveAdapter(ABC):
    """Abstract base class for mission-specific astronomical archive adapters.

    Provides a common structural interface for discovering observations, querying
    metadata, and referencing data products across space telescope archives
    (e.g., Hubble, JWST) without exposing mission-specific archive APIs to downstream
    processing pipelines.
    """

    @property
    @abstractmethod
    def mission_name(self) -> str:
        """Return the canonical name of the mission handled by this adapter."""
        pass

    @abstractmethod
    def search_observations(
        self,
        target_name: Optional[str] = None,
        right_ascension: Optional[float] = None,
        declination: Optional[float] = None,
        radius_arcmin: float = 1.0,
        limit: int = 50,
    ) -> List[Observation]:
        """Search archive for observations matching astronomical target criteria.

        Args:
            target_name: Celestial target identifier (e.g. 'M31', 'NGC 1068').
            right_ascension: Center Right Ascension in degrees.
            declination: Center Declination in degrees.
            radius_arcmin: Search radius around coordinates in arcminutes.
            limit: Maximum number of observation records to return.

        Returns:
            List of matching Observation metadata objects.
        """
        pass

    @abstractmethod
    def get_observation_metadata(self, observation_id: str) -> Observation:
        """Retrieve metadata for a specific observation ID.

        Args:
            observation_id: Unique mission/archive observation identifier.

        Returns:
            Observation metadata object.
        """
        pass

    @abstractmethod
    def get_data_product_uris(self, observation_id: str) -> List[str]:
        """Retrieve references/URIs for scientific data products belonging to an observation.

        Args:
            observation_id: Unique mission/archive observation identifier.

        Returns:
            List of URI strings referencing raw or calibrated data products.
        """
        pass
