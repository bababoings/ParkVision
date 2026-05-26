"""Internal multicamera identity models.

These types are preparation only; the current single-camera runtime does not
import or use them yet.
"""

from .models import (
    CameraConfig,
    CameraId,
    CameraROI,
    ConsolidatedSpaceState,
    LayoutId,
    PhysicalSpace,
    PhysicalSpaceId,
    Point,
    ROIId,
    SpaceObservation,
    ZoneId,
)

__all__ = [
    "CameraConfig",
    "CameraId",
    "CameraROI",
    "ConsolidatedSpaceState",
    "LayoutId",
    "PhysicalSpace",
    "PhysicalSpaceId",
    "Point",
    "ROIId",
    "SpaceObservation",
    "ZoneId",
]
