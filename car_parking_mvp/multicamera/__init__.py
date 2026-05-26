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
from .fusion import ObservationFusionPolicy, fuse_observations
from .layouts import (
    default_physical_space_id,
    default_roi_id,
    normalize_points,
    positions_to_camera_rois,
)
from .runtime import ShadowCameraRuntime
from .worker import CameraWorker

__all__ = [
    "CameraConfig",
    "CameraId",
    "CameraROI",
    "CameraWorker",
    "ConsolidatedSpaceState",
    "LayoutId",
    "ObservationFusionPolicy",
    "PhysicalSpace",
    "PhysicalSpaceId",
    "Point",
    "ROIId",
    "ShadowCameraRuntime",
    "SpaceObservation",
    "ZoneId",
    "default_physical_space_id",
    "default_roi_id",
    "fuse_observations",
    "normalize_points",
    "positions_to_camera_rois",
]
