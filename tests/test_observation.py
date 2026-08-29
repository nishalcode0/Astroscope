"""Test Observation dataclass behavior and validation."""

from datetime import datetime, timezone
import pytest
from dataclasses import FrozenInstanceError

from astroscope import Observation


def test_observation_instantiation() -> None:
    """Verify Observation instantiation with valid scientific metadata."""
    obs_time = datetime(2026, 8, 29, 12, 0, 0, tzinfo=timezone.utc)
    obs = Observation(
        mission="HST",
        observation_id="hst_12345_f606w",
        target_name="M31",
        right_ascension=10.6847,
        declination=41.2687,
        observation_time=obs_time,
        instrument="ACS/WFC",
        filter_band="F606W",
        data_products=("hst_12345_f606w_drz.fits",),
        extra_metadata={"exposure_time": 1200.0},
    )

    assert obs.mission == "HST"
    assert obs.observation_id == "hst_12345_f606w"
    assert obs.target_name == "M31"
    assert obs.right_ascension == 10.6847
    assert obs.declination == 41.2687
    assert obs.observation_time == obs_time
    assert obs.instrument == "ACS/WFC"
    assert obs.filter_band == "F606W"
    assert obs.data_products == ("hst_12345_f606w_drz.fits",)
    assert obs.extra_metadata == {"exposure_time": 1200.0}


def test_observation_immutability() -> None:
    """Verify Observation fields cannot be mutated post-instantiation."""
    obs = Observation(
        mission="JWST",
        observation_id="jwst_01234_nircam",
        target_name="SMACS0723",
        right_ascension=110.8375,
        declination=-73.4542,
        observation_time=None,
        instrument="NIRCam",
        filter_band="F200W",
    )

    with pytest.raises(FrozenInstanceError):
        obs.mission = "Hubble"  # type: ignore[misc]


def test_observation_coordinate_validation() -> None:
    """Verify validation raises ValueError for invalid coordinate bounds."""
    with pytest.raises(ValueError, match="right_ascension"):
        Observation(
            mission="HST",
            observation_id="obs_bad_ra",
            target_name="Target",
            right_ascension=361.0,
            declination=0.0,
            observation_time=None,
            instrument="WFC3",
            filter_band="F160W",
        )

    with pytest.raises(ValueError, match="declination"):
        Observation(
            mission="HST",
            observation_id="obs_bad_dec",
            target_name="Target",
            right_ascension=180.0,
            declination=-95.0,
            observation_time=None,
            instrument="WFC3",
            filter_band="F160W",
        )
