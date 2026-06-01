import sys
import types

import numpy as np

sys.modules.setdefault("mujoco", types.ModuleType("mujoco"))

from tools.residual_safety_sweep import (
    ResidualActionGenerator,
    ResidualSweepCase,
    build_parser,
    build_sweep_cases,
)


def test_build_sweep_cases_covers_requested_matrix() -> None:
    cases = build_sweep_cases()

    assert len(cases) == 40
    assert {case.action_joint_delta_scale for case in cases} == {
        0.001,
        0.002,
        0.003,
        0.005,
        0.008,
    }
    assert {case.update_interval for case in cases if case.mode == "low_frequency"} == {
        10,
        20,
        40,
    }
    assert {case.gaussian_sigma for case in cases if case.mode == "gaussian"} == {
        0.05,
        0.1,
        0.2,
    }


def test_constant_residual_holds_one_action_for_episode() -> None:
    generator = ResidualActionGenerator(
        ResidualSweepCase("constant", 0.001),
        action_shape=(10,),
        seed=7,
    )

    first = generator.action_at_step(0)
    later = generator.action_at_step(25)

    assert np.array_equal(first, later)


def test_low_frequency_residual_holds_until_interval_boundary() -> None:
    generator = ResidualActionGenerator(
        ResidualSweepCase("low_frequency", 0.001, update_interval=10),
        action_shape=(10,),
        seed=7,
    )

    first = generator.action_at_step(0)
    held = generator.action_at_step(9)
    updated = generator.action_at_step(10)

    assert np.array_equal(first, held)
    assert not np.array_equal(first, updated)


def test_parser_defaults_to_requested_output() -> None:
    args = build_parser().parse_args(["--steps", "480"])

    assert args.steps == 480
    assert args.landing_impact_multiplier == 1.15
