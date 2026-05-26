"""Shadow camera worker for multicamera observation experiments.

Workers produce SpaceObservation objects in memory only. They do not write to
Supabase and do not publish to parking_spaces or occupancy.
"""

import os
import threading
import time
from datetime import datetime, timezone
from typing import List, Optional
from uuid import uuid4

import cv2

import inference
from preprocessing import preprocess_frame

from .models import CameraConfig, CameraROI, SpaceObservation


MODEL_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "parking_mobilenetv2.h5")
MODEL_FALLBACK = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model_final.h5")


class CameraWorker:
    """Process one camera source and keep shadow observations in memory."""

    def __init__(
        self,
        camera_config: CameraConfig,
        rois: List[CameraROI],
        confidence_threshold: float = 0.7,
        model_name: str = "parking_mobilenetv2",
        model_file: str = MODEL_FILE,
        model_fallback: str = MODEL_FALLBACK,
        frame_interval_seconds: float = 0.03,
        max_observations: int = 500,
    ):
        self.camera_config = camera_config
        self.rois = list(rois)
        self.confidence_threshold = confidence_threshold
        self.model_name = model_name
        self.model_file = model_file
        self.model_fallback = model_fallback
        self.frame_interval_seconds = frame_interval_seconds
        self.max_observations = max_observations

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._capture = None
        self._recent_observations: List[SpaceObservation] = []

        self._state = {
            "model": None,
            "positions": self._positions_from_rois(),
            "active_layout": camera_config.layout_id,
            "class_dictionary": {0: "Disponible", 1: "Ocupado"},
            "previous_states": [None] * len(self.rois),
            "confidence_threshold": confidence_threshold,
            "model_input_size": (96, 96),
        }

    @property
    def camera_id(self) -> str:
        return self.camera_config.camera_id

    def start(self) -> None:
        """Start the background shadow loop for this camera."""
        if self._thread and self._thread.is_alive():
            return

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self.run_forever,
            name=f"shadow-camera-{self.camera_id}",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop the background shadow loop and release capture resources."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        self._release_capture()

    def run_forever(self) -> None:
        """Read frames until stopped, producing observations in memory."""
        capture = self._open_capture()
        if capture is None:
            return

        while not self._stop_event.is_set():
            success, frame = capture.read()
            if not success:
                if self._is_file_source():
                    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    success, frame = capture.read()
                if not success:
                    time.sleep(self.frame_interval_seconds)
                    continue

            self.process_frame(frame)
            time.sleep(self.frame_interval_seconds)

    def process_frame(self, frame) -> List[SpaceObservation]:
        """Process one frame and return the observations generated from it."""
        processed, _, _ = preprocess_frame(frame)
        if processed is None:
            return []

        _, _, _, _, spaces_status = inference.check_parking_spaces(
            processed,
            self._state,
            self.model_file,
            self.model_fallback,
            self._shadow_space_id,
        )

        observed_at = datetime.now(timezone.utc)
        observations = []
        for index, status in enumerate(spaces_status):
            if index >= len(self.rois):
                continue
            roi = self.rois[index]
            observations.append(
                SpaceObservation(
                    observation_id=str(uuid4()),
                    camera_id=roi.camera_id,
                    roi_id=roi.roi_id,
                    physical_space_id=roi.physical_space_id,
                    zone_id=roi.zone_id,
                    is_occupied=bool(status["is_occupied"]),
                    confidence=float(status["confidence"]),
                    observed_at=observed_at,
                    model_name=self.model_name,
                    metadata={"layout_id": roi.layout_id},
                )
            )

        self._store_observations(observations)
        return observations

    def recent_observations(self) -> List[SpaceObservation]:
        """Return a snapshot of recent in-memory observations."""
        with self._lock:
            return list(self._recent_observations)

    def _positions_from_rois(self) -> List[dict]:
        return [
            {
                "points": [[x, y] for x, y in roi.points],
                "zone": roi.zone_id,
            }
            for roi in self.rois
        ]

    def _shadow_space_id(self, _layout_id: str, index: int) -> str:
        if index < len(self.rois):
            return self.rois[index].physical_space_id
        return f"{self.camera_config.layout_id}-{index + 1:03d}"

    def _store_observations(self, observations: List[SpaceObservation]) -> None:
        if not observations:
            return
        with self._lock:
            self._recent_observations.extend(observations)
            if len(self._recent_observations) > self.max_observations:
                self._recent_observations = self._recent_observations[-self.max_observations:]

    def _open_capture(self):
        source = self.camera_config.source
        if source.isdigit():
            capture = cv2.VideoCapture(int(source))
        else:
            capture = cv2.VideoCapture(source)

        if not capture.isOpened():
            self._capture = None
            return None

        self._capture = capture
        return capture

    def _release_capture(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def _is_file_source(self) -> bool:
        return self.camera_config.source.lower().endswith((".mp4", ".avi", ".mkv", ".mov"))
