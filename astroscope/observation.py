"""Scientific representation of astronomical observation metadata."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional, Tuple


@dataclass(frozen=True)
class Observation:
    """Typed metadata representation of an astronomical observation.

    Represents observational metadata (e.g. target, coordinates, instrument parameters,
    and references to scientific data products) without loading image pixel payloads.
    """

    mission: str
    observation_id: str
    target_name: str
    right_ascension: float  # Right Ascension in degrees (0.0 to 360.0)
    declination: float  # Declination in degrees (-90.0 to +90.0)
    observation_time: Optional[datetime]  # UTC observation timestamp
    instrument: str
    filter_band: str
    data_products: Tuple[str, ...] = field(default_factory=tuple)
    extra_metadata: Dict[str, Any] = field(default_factory=dict)

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
