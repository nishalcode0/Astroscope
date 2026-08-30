"""Test ArchiveAdapter interface and mission adapters (Hubble & JWST).

Session 2 notes
---------------
- HubbleAdapter methods now have real MAST implementations.
  The old assertions that they raise NotImplementedError have been removed.
  Comprehensive mocked tests for Hubble live in tests/test_hubble_mast.py.
- JWSTAdapter methods remain structural stubs (JWST networking is not yet
  implemented) and still raise NotImplementedError.
"""

import pytest

from astroscope.archive.base import ArchiveAdapter
from astroscope.archive.hubble import HubbleAdapter
from astroscope.archive.jwst import JWSTAdapter


def test_hubble_adapter_is_archive_adapter() -> None:
    """Verify HubbleAdapter inherits from ArchiveAdapter."""
    adapter = HubbleAdapter()
    assert isinstance(adapter, ArchiveAdapter)


def test_hubble_adapter_mission_name() -> None:
    """Verify HubbleAdapter returns the correct canonical mission name."""
    adapter = HubbleAdapter()
    assert adapter.mission_name == "Hubble Space Telescope"


def test_jwst_adapter_interface() -> None:
    """Verify JWSTAdapter inherits from ArchiveAdapter and exposes required interface."""
    adapter = JWSTAdapter()
    assert isinstance(adapter, ArchiveAdapter)
    assert adapter.mission_name == "James Webb Space Telescope"

    # JWST networking is not yet implemented; stubs must raise NotImplementedError
    with pytest.raises(NotImplementedError):
        adapter.search_observations(target_name="SMACS0723")

    with pytest.raises(NotImplementedError):
        adapter.get_observation_metadata("jwst_01234")

    with pytest.raises(NotImplementedError):
        adapter.get_products(99999)

    with pytest.raises(NotImplementedError):
        adapter.get_data_product_uris("jwst_01234")
