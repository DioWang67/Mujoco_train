from sedon_baseline.checkpoint_selection import (
    ForwardCheckpointThresholds,
    is_safe_forward_candidate,
    is_stable_forward_candidate,
)


def _metrics(**overrides: float) -> dict[str, float]:
    metrics = {
        "mean_forward_velocity": 0.04,
        "fall_rate": 0.0,
        "mean_length": 400.0,
        "mean_final_base_x": 0.06,
        "mean_final_base_z": 0.44,
        "mean_final_upright": 0.99,
        "both_contact_ratio": 0.98,
        "single_contact_ratio": 0.02,
        "no_contact_ratio": 0.0,
    }
    metrics.update(overrides)
    return metrics


def test_safe_forward_rejects_fast_falling_checkpoint() -> None:
    thresholds = ForwardCheckpointThresholds()

    assert not is_safe_forward_candidate(
        _metrics(mean_forward_velocity=0.75, fall_rate=1.0, mean_length=27.0),
        thresholds,
    )


def test_safe_forward_rejects_airborne_shortcut() -> None:
    thresholds = ForwardCheckpointThresholds(max_no_contact_ratio=0.02)

    assert not is_safe_forward_candidate(
        _metrics(no_contact_ratio=0.63, both_contact_ratio=0.11),
        thresholds,
    )


def test_stable_forward_requires_meaningful_displacement() -> None:
    thresholds = ForwardCheckpointThresholds(min_stable_base_x=0.04)

    assert not is_stable_forward_candidate(
        _metrics(mean_final_base_x=0.028),
        thresholds,
        min_base_height=0.30,
        min_upright=0.80,
        max_episode_steps=400,
    )
    assert is_stable_forward_candidate(
        _metrics(mean_final_base_x=0.055),
        thresholds,
        min_base_height=0.30,
        min_upright=0.80,
        max_episode_steps=400,
    )
