"""Identity models for the future multicamera pipeline.

The key separation is:
- CameraROI: a visual polygon observed in one camera frame.
- PhysicalSpace: the real parking stall that one or more ROIs can refer to.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


CameraId = str
ROIId = str
PhysicalSpaceId = str
LayoutId = str
ZoneId = str
Point = Tuple[float, float]


@dataclass(frozen=True)
class CameraConfig:
    """Configuration for a single video source in a multicamera setup."""

    camera_id: CameraId
    name: str
    source: str
    layout_id: LayoutId
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CameraROI:
    """Visual parking-space ROI in one camera's image coordinates."""

    roi_id: ROIId
    camera_id: CameraId
    layout_id: LayoutId
    physical_space_id: PhysicalSpaceId
    zone_id: ZoneId
    points: List[Point]
    label: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PhysicalSpace:
    """Real parking stall, independent of how many cameras observe it."""

    physical_space_id: PhysicalSpaceId
    zone_id: ZoneId
    display_name: Optional[str] = None
    map_points: Optional[List[Point]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SpaceObservation:
    """One model prediction for one ROI at one moment in time."""

    observation_id: str
    camera_id: CameraId
    roi_id: ROIId
    physical_space_id: PhysicalSpaceId
    zone_id: ZoneId
    is_occupied: bool
    confidence: float
    observed_at: datetime
    model_name: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ConsolidatedSpaceState:
    """Final fused state for one physical parking stall."""

    physical_space_id: PhysicalSpaceId
    zone_id: ZoneId
    is_occupied: bool
    confidence: float
    updated_at: datetime
    source_observation_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
