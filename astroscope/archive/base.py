"""Base abstract interface for astronomical archive adapters."""

from abc import ABC, abstractmethod
from typing import List, Optional

from astroscope.observation import Observation
from astroscope.product import Product


class ArchiveAdapter(ABC):
    """Abstract base class for mission-specific astronomical archive adapters.

    Provides a common structural interface for discovering observations, querying
    metadata, and referencing data products across space telescope archives
    (e.g., Hubble, JWST) without exposing mission-specific archive APIs to downstream
    processing pipelines.

    Pipeline Layers
    ---------------
    This adapter covers the **archive layer** only:

    - ``search_observations`` / ``get_observation_metadata`` — *"What observations exist?"*
    - ``get_products`` — *"What data products exist for an observation?"*
    - ``get_data_product_uris`` — backwards-compatible URI list (wraps ``get_products``).

    Downloading, processing, and analysis are separate concerns handled by
    subsequent pipeline layers.
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

        Parameters
        ----------
        target_name:
            Celestial target identifier (e.g. ``"M31"``, ``"NGC 1068"``).
        right_ascension:
            Centre Right Ascension in degrees (0 – 360).
        declination:
            Centre Declination in degrees (−90 – +90).
        radius_arcmin:
            Search radius around coordinates in arcminutes.
        limit:
            Maximum number of observation records to return.

        Returns
        -------
        List[Observation]
            Normalised observation metadata objects.
        """
        pass

    @abstractmethod
    def get_observation_metadata(self, observation_id: str) -> Observation:
        """Retrieve metadata for a specific observation ID.

        Parameters
        ----------
        observation_id:
            Unique mission/archive observation identifier (``obs_id``).

        Returns
        -------
        Observation
            Normalised observation metadata object.
        """
        pass

    @abstractmethod
    def get_products(self, mast_obsid: int) -> List[Product]:
        """Retrieve the full product inventory for one MAST observation.

        Parameters
        ----------
        mast_obsid:
            The MAST integer observation key (``Observation.mast_obsid``).
            This is *not* the mission ``obs_id`` string.

        Returns
        -------
        List[Product]
            All products associated with the observation, including SCIENCE,
            PREVIEW, and AUXILIARY entries.  The list is returned as-is from
            the archive; no filtering or selection is performed.

        Notes
        -----
        Callers must not assume:

        - That any particular product type is present.
        - That the highest calibration level is the correct choice.
        - That PREVIEW and SCIENCE products are interchangeable.
        - That HST and HLA products within the same list are equivalent.

        Product selection and download are the responsibility of the ingestion
        layer, which is implemented in a future session.
        """
        pass

    @abstractmethod
    def get_data_product_uris(self, observation_id: str) -> List[str]:
        """Retrieve MAST URIs for scientific data products of an observation.

        This method is retained for backwards compatibility.  New code should
        prefer :meth:`get_products`, which returns structured
        :class:`~astroscope.product.Product` objects with full metadata.

        Parameters
        ----------
        observation_id:
            Unique mission/archive observation identifier (``obs_id``).

        Returns
        -------
        List[str]
            MAST URI strings (e.g. ``"mast:HST/product/n4eya1020_mos.fits"``)
            for SCIENCE-type products only.
        """
        pass
