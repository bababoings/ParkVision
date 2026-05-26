"""Adapters from current single-camera layouts to shadow multicamera ROIs."""

from typing import Iterable, List, Mapping, Optional, Sequence

from .models import CameraId, CameraROI, LayoutId, PhysicalSpaceId, Point, ZoneId


def normalize_points(points: Sequence[Sequence[float]]) -> List[Point]:
    """Return four image points as float tuples."""
    if len(points) != 4:
        raise ValueError("CameraROI points must contain exactly four vertices")

    normalized = []
    for point in points:
        if len(point) != 2:
            raise ValueError("Each CameraROI point must contain x and y")
        normalized.append((float(point[0]), float(point[1])))
    return normalized


def default_roi_id(camera_id: CameraId, layout_id: LayoutId, index: int) -> str:
    """Build a stable shadow ROI id from camera, layout, and 0-based index."""
    return f"{camera_id}:{layout_id}:roi-{index + 1:03d}"


def default_physical_space_id(layout_id: LayoutId, index: int) -> str:
    """Build a temporary physical-space id compatible with legacy ordering."""
    return f"{layout_id}-{index + 1:03d}"


def positions_to_camera_rois(
    positions: Iterable[Mapping[str, object]],
    camera_id: CameraId,
    layout_id: LayoutId,
    physical_space_ids: Optional[Sequence[PhysicalSpaceId]] = None,
) -> List[CameraROI]:
    """Convert current {points, zone} layout records into shadow CameraROI objects.

    This is a compatibility adapter only. It does not write to Supabase and does
    not change the current SpacePicker/runtime layout contract.
    """
    rois = []
    for index, position in enumerate(positions):
        raw_points = position.get("points")
        if not isinstance(raw_points, Sequence):
            raise ValueError(f"Position {index + 1} is missing points")

        zone_id = position.get("zone", "Zona A")
        if not isinstance(zone_id, str):
            zone_id = str(zone_id)

        if physical_space_ids is not None and index < len(physical_space_ids):
            physical_space_id = physical_space_ids[index]
        else:
            physical_space_id = default_physical_space_id(layout_id, index)

        rois.append(
            CameraROI(
                roi_id=default_roi_id(camera_id, layout_id, index),
                camera_id=camera_id,
                layout_id=layout_id,
                physical_space_id=physical_space_id,
                zone_id=ZoneId(zone_id),
                points=normalize_points(raw_points),
                label=position.get("label") if isinstance(position.get("label"), str) else None,
            )
        )

    return rois
