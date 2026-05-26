"""In-memory fusion for shadow multicamera observations."""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, Iterable, List, Optional

from .models import ConsolidatedSpaceState, PhysicalSpaceId, SpaceObservation


@dataclass(frozen=True)
class ObservationFusionPolicy:
    """Initial conservative policy for fusing observations by physical space."""

    freshness_seconds: float = 30.0
    low_margin_threshold: float = 0.1
    prefer_occupied_on_low_margin: bool = True


def fuse_observations(
    observations: Iterable[SpaceObservation],
    policy: Optional[ObservationFusionPolicy] = None,
    now: Optional[datetime] = None,
) -> Dict[PhysicalSpaceId, ConsolidatedSpaceState]:
    """Fuse fresh raw observations into consolidated in-memory space states."""
    policy = policy or ObservationFusionPolicy()
    now = _as_aware_utc(now or datetime.now(timezone.utc))
    grouped = _group_fresh_observations(observations, policy, now)

    fused: Dict[PhysicalSpaceId, ConsolidatedSpaceState] = {}
    for physical_space_id, group in grouped.items():
        state = _fuse_group(physical_space_id, group, policy)
        if state is not None:
            fused[physical_space_id] = state
    return fused


def _group_fresh_observations(
    observations: Iterable[SpaceObservation],
    policy: ObservationFusionPolicy,
    now: datetime,
) -> Dict[PhysicalSpaceId, List[SpaceObservation]]:
    max_age = timedelta(seconds=policy.freshness_seconds)
    grouped: Dict[PhysicalSpaceId, List[SpaceObservation]] = {}

    for observation in observations:
        observed_at = _as_aware_utc(observation.observed_at)
        if now - observed_at > max_age:
            continue
        grouped.setdefault(observation.physical_space_id, []).append(observation)

    return grouped


def _fuse_group(
    physical_space_id: PhysicalSpaceId,
    observations: List[SpaceObservation],
    policy: ObservationFusionPolicy,
) -> Optional[ConsolidatedSpaceState]:
    if not observations:
        return None

    observations = sorted(observations, key=lambda obs: _as_aware_utc(obs.observed_at))
    latest_observation = observations[-1]
    occupied_score = sum(_clamp_confidence(obs.confidence) for obs in observations if obs.is_occupied)
    available_score = sum(_clamp_confidence(obs.confidence) for obs in observations if not obs.is_occupied)
    total_score = occupied_score + available_score

    if total_score <= 0:
        is_occupied = policy.prefer_occupied_on_low_margin
        confidence = 0.0
    elif occupied_score == 0 or available_score == 0:
        is_occupied = occupied_score > 0
        winning_observations = [obs for obs in observations if obs.is_occupied == is_occupied]
        confidence = _average_confidence(winning_observations)
    else:
        margin = abs(occupied_score - available_score) / total_score
        if margin < policy.low_margin_threshold and policy.prefer_occupied_on_low_margin:
            is_occupied = True
        else:
            is_occupied = occupied_score >= available_score

        chosen_score = occupied_score if is_occupied else available_score
        confidence = _clamp_confidence(chosen_score / total_score)

    return ConsolidatedSpaceState(
        physical_space_id=physical_space_id,
        zone_id=latest_observation.zone_id,
        is_occupied=is_occupied,
        confidence=confidence,
        updated_at=_as_aware_utc(latest_observation.observed_at),
        source_observation_ids=[obs.observation_id for obs in observations],
        metadata={
            "fresh_observation_count": len(observations),
            "occupied_score": round(occupied_score, 4),
            "available_score": round(available_score, 4),
        },
    )


def _average_confidence(observations: List[SpaceObservation]) -> float:
    if not observations:
        return 0.0
    return _clamp_confidence(
        sum(_clamp_confidence(obs.confidence) for obs in observations) / len(observations)
    )


def _clamp_confidence(confidence: float) -> float:
    return max(0.0, min(1.0, float(confidence)))


def _as_aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)
