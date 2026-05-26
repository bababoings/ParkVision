"""In-memory shadow runtime for one or more camera workers."""

from typing import Dict, Iterable, List, Optional

from .models import CameraId, SpaceObservation
from .worker import CameraWorker


class ShadowCameraRuntime:
    """Coordinate shadow camera workers without publishing public state."""

    def __init__(self, workers: Optional[Iterable[CameraWorker]] = None):
        self._workers: Dict[CameraId, CameraWorker] = {}
        if workers:
            for worker in workers:
                self.add_worker(worker)

    def add_worker(self, worker: CameraWorker) -> None:
        """Register a worker by camera_id."""
        self._workers[worker.camera_id] = worker

    def remove_worker(self, camera_id: CameraId) -> Optional[CameraWorker]:
        """Stop and remove a worker if it exists."""
        worker = self._workers.pop(camera_id, None)
        if worker is not None:
            worker.stop()
        return worker

    def start_all(self) -> None:
        """Start all registered shadow workers."""
        for worker in self._workers.values():
            worker.start()

    def stop_all(self) -> None:
        """Stop all registered shadow workers."""
        for worker in self._workers.values():
            worker.stop()

    def worker(self, camera_id: CameraId) -> Optional[CameraWorker]:
        """Return one registered worker."""
        return self._workers.get(camera_id)

    def camera_ids(self) -> List[CameraId]:
        """Return registered camera ids."""
        return list(self._workers.keys())

    def recent_observations(self, camera_id: Optional[CameraId] = None) -> List[SpaceObservation]:
        """Return recent in-memory observations for one camera or all cameras."""
        if camera_id is not None:
            worker = self._workers.get(camera_id)
            return worker.recent_observations() if worker else []

        observations: List[SpaceObservation] = []
        for worker in self._workers.values():
            observations.extend(worker.recent_observations())
        observations.sort(key=lambda observation: observation.observed_at)
        return observations

    def latest_by_physical_space(self) -> Dict[str, SpaceObservation]:
        """Return the latest raw observation per physical space.

        This is not fusion and must not be published to public tables. It is only
        a convenience view for shadow validation.
        """
        latest: Dict[str, SpaceObservation] = {}
        for observation in self.recent_observations():
            current = latest.get(observation.physical_space_id)
            if current is None or observation.observed_at > current.observed_at:
                latest[observation.physical_space_id] = observation
        return latest
