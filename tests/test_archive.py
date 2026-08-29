"""Test ArchiveAdapter interface and mission adapters (Hubble & JWST)."""

import pytest

from astroscope.archive.base import ArchiveAdapter
from astroscope.archive.hubble import HubbleAdapter
from astroscope.archive.jwst import JWSTAdapter


def test_hubble_adapter_interface() -> None:
    """Verify HubbleAdapter inherits from ArchiveAdapter and exposes required interface."""
    adapter = HubbleAdapter()
    assert isinstance(adapter, ArchiveAdapter)
    assert adapter.mission_name == "Hubble Space Telescope"

    # Verify structural stubs raise NotImplementedError offline
    with pytest.raises(NotImplementedError):
        adapter.search_observations(target_name="M31")

    with pytest.raises(NotImplementedError):
        adapter.get_observation_metadata("hst_12345")

    with pytest.raises(NotImplementedError):
        adapter.get_data_product_uris("hst_12345")


def test_jwst_adapter_interface() -> None:
    """Verify JWSTAdapter inherits from ArchiveAdapter and exposes required interface."""
    adapter = JWSTAdapter()
    assert isinstance(adapter, ArchiveAdapter)
    assert adapter.mission_name == "James Webb Space Telescope"

    # Verify structural stubs raise NotImplementedError offline
    with pytest.raises(NotImplementedError):
        adapter.search_observations(target_name="SMACS0723")

    with pytest.raises(NotImplementedError):
        adapter.get_observation_metadata("jwst_01234")

    with pytest.raises(NotImplementedError):
        adapter.get_data_product_uris("jwst_01234")
