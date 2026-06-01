import json
import sys
import types

import numpy as np
import pytest

sys.modules.setdefault("mujoco", types.ModuleType("mujoco"))

from sedon_baseline.env import JOINT_NAMES
from tools.debug_sedon_gait_viewer import _load_gait_seed, _seed_target_at_step
from tools.debug_sedon_pose_editor import (
    PoseEntry,
    _foot_force_state,
    _support_side_guess,
    _write_pose_entries,
)


def test_pose_entry_defaults_duration_for_legacy_pose() -> None:
    raw = {
        "name": "Pose A",
        "support_mode": "double",
        "joint_targets": [0.0] * len(JOINT_NAMES),
    }

    entry = PoseEntry.from_dict(raw)

    assert entry.duration_steps == 60
    assert entry.joint_targets == [0.0] * len(JOINT_NAMES)


def test_pose_entries_are_written_as_json_list(tmp_path) -> None:
    path = tmp_path / "poses.json"
    entry = PoseEntry(
        name="right_micro_lift",
        support_mode="left",
        joint_targets=[0.1] * len(JOINT_NAMES),
        duration_steps=24,
        note="right foot unload and micro-lift",
    )

    _write_pose_entries(path, [entry])

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload[0]["name"] == "right_micro_lift"
    assert payload[0]["support_mode"] == "left"
    assert payload[0]["duration_steps"] == 24


def test_gait_seed_loader_rejects_wrong_joint_names(tmp_path) -> None:
    path = tmp_path / "bad_seed.json"
    path.write_text(
        json.dumps(
            {
                "schema": "sedon_gait_seed.v1",
                "joint_names": ["wrong_joint"],
                "keyframes": [],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="joint_names"):
        _load_gait_seed(path)


def test_gait_seed_interpolates_between_keyframes(tmp_path) -> None:
    path = tmp_path / "seed.json"
    zeros = [0.0] * len(JOINT_NAMES)
    ones = [1.0] * len(JOINT_NAMES)
    path.write_text(
        json.dumps(
            {
                "schema": "sedon_gait_seed.v1",
                "joint_names": list(JOINT_NAMES),
                "keyframes": [
                    {
                        "name": "Pose A",
                        "support_mode": "double",
                        "joint_targets": zeros,
                        "duration_steps": 4,
                    },
                    {
                        "name": "Pose B",
                        "support_mode": "left",
                        "joint_targets": ones,
                        "duration_steps": 4,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    seed = _load_gait_seed(path)
    target, phase_name, support_mode = _seed_target_at_step(seed, 2)

    assert phase_name == "Pose A"
    assert support_mode == "double"
    assert np.all(target > 0.0)
    assert np.all(target < 1.0)


def test_foot_force_state_uses_requested_ratio_formula() -> None:
    state = _foot_force_state(left_force=30.0, right_force=10.0)

    assert state.left_force == 30.0
    assert state.right_force == 10.0
    assert state.force_ratio_left == pytest.approx(30.0 / (30.0 + 10.0 + 1e-6))
    assert state.force_ratio_right == pytest.approx(10.0 / (30.0 + 10.0 + 1e-6))
    assert state.support_side_guess == "double"


@pytest.mark.parametrize(
    ("left_force_z", "right_force_z", "expected"),
    [
        (6.0, 2.0, "left"),
        (2.0, 6.0, "right"),
        (6.0, 7.0, "double"),
        (5.0, 5.0, "none"),
    ],
)
def test_support_side_guess_uses_strict_five_newton_threshold(
    left_force_z: float,
    right_force_z: float,
    expected: str,
) -> None:
    assert _support_side_guess(left_force_z, right_force_z) == expected
