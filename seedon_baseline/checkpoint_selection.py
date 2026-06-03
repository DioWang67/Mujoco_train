"""Checkpoint selection helpers for Seedon training callbacks."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ForwardCheckpointThresholds:
    """Thresholds used to reject unstable forward-progress checkpoints.

    Args:
        max_fall_rate: Maximum accepted fraction of fallen evaluation episodes.
        min_mean_length: Minimum accepted mean episode length.
        max_no_contact_ratio: Maximum accepted fraction of airborne evaluation steps.
        min_both_contact_ratio: Minimum accepted fraction of two-foot contact steps.
        min_stable_base_x: Minimum final forward displacement for stable-forward saves.
    """

    max_fall_rate: float = 0.0
    min_mean_length: float = 360.0
    max_no_contact_ratio: float = 0.02
    min_both_contact_ratio: float = 0.90
    min_stable_base_x: float = 0.04


def is_safe_forward_candidate(
    metrics: dict[str, float],
    thresholds: ForwardCheckpointThresholds,
) -> bool:
    """Return whether a forward checkpoint is physically usable.

    Args:
        metrics: Evaluation metrics produced by the training callback.
        thresholds: Contact and stability thresholds.

    Returns:
        True when the checkpoint moves forward without relying on falls or airtime.
    """

    return (
        metrics["mean_forward_velocity"] > 0.0
        and metrics["fall_rate"] <= thresholds.max_fall_rate
        and metrics["mean_length"] >= thresholds.min_mean_length
        and metrics["no_contact_ratio"] <= thresholds.max_no_contact_ratio
        and metrics["both_contact_ratio"] >= thresholds.min_both_contact_ratio
    )


def is_stable_forward_candidate(
    metrics: dict[str, float],
    thresholds: ForwardCheckpointThresholds,
    *,
    min_base_height: float,
    min_upright: float,
    max_episode_steps: int,
) -> bool:
    """Return whether a checkpoint is stable and meaningfully moves forward.

    Args:
        metrics: Evaluation metrics produced by the training callback.
        thresholds: Contact and stability thresholds.
        min_base_height: Minimum final base height accepted as standing.
        min_upright: Minimum final upright value accepted as standing.
        max_episode_steps: Full evaluation episode length.

    Returns:
        True when the checkpoint survives the full eval and makes non-trivial progress.
    """

    if not is_safe_forward_candidate(metrics, thresholds):
        return False
    return (
        metrics["mean_length"] >= float(max_episode_steps)
        and metrics["mean_final_base_z"] >= min_base_height
        and metrics["mean_final_upright"] >= min_upright
        and metrics["mean_final_base_x"] >= thresholds.min_stable_base_x
    )
