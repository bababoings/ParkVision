import unittest
from datetime import datetime, timedelta, timezone

from multicamera.fusion import ObservationFusionPolicy, fuse_observations
from multicamera.models import SpaceObservation


def observation(
    observation_id,
    physical_space_id,
    is_occupied,
    confidence,
    observed_at,
    camera_id="cam-a",
):
    return SpaceObservation(
        observation_id=observation_id,
        camera_id=camera_id,
        roi_id=f"{camera_id}-roi",
        physical_space_id=physical_space_id,
        zone_id="zone-a",
        is_occupied=is_occupied,
        confidence=confidence,
        observed_at=observed_at,
    )


class MulticameraFusionTests(unittest.TestCase):
    def setUp(self):
        self.now = datetime(2026, 5, 26, 12, 0, tzinfo=timezone.utc)
        self.policy = ObservationFusionPolicy(freshness_seconds=30, low_margin_threshold=0.1)

    def test_agreement_uses_agreed_state(self):
        states = fuse_observations(
            [
                observation("a", "space-1", True, 0.9, self.now, "cam-a"),
                observation("b", "space-1", True, 0.7, self.now, "cam-b"),
            ],
            self.policy,
            self.now,
        )

        self.assertTrue(states["space-1"].is_occupied)
        self.assertAlmostEqual(states["space-1"].confidence, 0.8)
        self.assertEqual(states["space-1"].source_observation_ids, ["a", "b"])

    def test_conflict_sums_confidence_by_class(self):
        states = fuse_observations(
            [
                observation("a", "space-1", True, 0.9, self.now, "cam-a"),
                observation("b", "space-1", False, 0.4, self.now, "cam-b"),
            ],
            self.policy,
            self.now,
        )

        self.assertTrue(states["space-1"].is_occupied)
        self.assertAlmostEqual(states["space-1"].confidence, 0.9 / 1.3)

    def test_low_margin_conflict_prefers_occupied(self):
        states = fuse_observations(
            [
                observation("a", "space-1", True, 0.49, self.now, "cam-a"),
                observation("b", "space-1", False, 0.51, self.now, "cam-b"),
            ],
            self.policy,
            self.now,
        )

        self.assertTrue(states["space-1"].is_occupied)
        self.assertAlmostEqual(states["space-1"].confidence, 0.49)

    def test_camera_down_uses_remaining_fresh_observation(self):
        states = fuse_observations(
            [observation("a", "space-1", False, 0.8, self.now, "cam-a")],
            self.policy,
            self.now,
        )

        self.assertFalse(states["space-1"].is_occupied)
        self.assertAlmostEqual(states["space-1"].confidence, 0.8)

    def test_old_observations_are_ignored(self):
        states = fuse_observations(
            [
                observation("old", "space-1", True, 0.95, self.now - timedelta(seconds=31)),
                observation("fresh", "space-1", False, 0.8, self.now),
            ],
            self.policy,
            self.now,
        )

        self.assertFalse(states["space-1"].is_occupied)
        self.assertEqual(states["space-1"].source_observation_ids, ["fresh"])

    def test_no_fresh_observations_produces_no_state(self):
        states = fuse_observations(
            [observation("old", "space-1", True, 0.95, self.now - timedelta(seconds=31))],
            self.policy,
            self.now,
        )

        self.assertNotIn("space-1", states)


if __name__ == "__main__":
    unittest.main()
