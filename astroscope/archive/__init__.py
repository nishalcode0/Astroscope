"""Astronomical archive adapter module."""

from astroscope.archive.base import ArchiveAdapter
from astroscope.archive.hubble import HubbleAdapter
from astroscope.archive.jwst import JWSTAdapter

__all__ = ["ArchiveAdapter", "HubbleAdapter", "JWSTAdapter"]
