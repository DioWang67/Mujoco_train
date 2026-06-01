"""Sweep scripted Sedon hip-roll assist for Phase 1A load-transfer reachability."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sedon_baseline.env import (
    LEFT_HIP_ROLL_ACTUATOR_INDEX,
    RIGHT_HIP_ROLL_ACTUATOR_INDEX,
    SedonStandingConfig,
    SedonStandingEnv,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "sedon" / "march_phase1a_v2_load_transfer_roll_0030.json"
DEFAULT_OUT_CSV = REPO_ROOT / "artifacts" / "sedon_debug" / "phase1a_hip_roll_assist_sweep.csv"
DEFAULT_AMPLITUDES = (0.01, 0.02, 0.04, 0.06, 0.08)


@dataclass(frozen=True)
class SweepResult:
    """Aggregated metrics for one scripted hip-roll assist amplitude."""

    amplitude: float
    steps: int
    stable: bool
    pass_v1: bool
    phase1b_reachable: bool
    terminated: bool
    termination_reason: str
    max_support_ratio: float
    target_side_force_ratio_mean: float
    target_side_force_ratio_max: float
    com_y_shift_max: float
    base_y_drift_max: float
    base_roll_abs_max: float
    base_pitch_abs_max: float
    both_contact_ratio: float
    foot_penetration_max: float
    actuator_ctrl_saturation_max: float
    actuator_force_saturation_max: float


def _load_config(path: Path) -> SedonStandingConfig:
    """Load a SedonStandingConfig from a JSON override file."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Sedon config must be a JSON object: {path}")
    return SedonStandingConfig(**{**SedonStandingConfig().__dict__, **payload})


def _assist_vector(env: SedonStandingEnv, amplitude: float) -> np.ndarray:
    """Return scripted hip-roll offset for the current target support side."""
    phase = env._task_phase_metadata()
    support_side = str(phase["support_side"])
    assist = np.zeros(env.action_space.shape, dtype=np.float64)
    if support_side == "right":
        assist[RIGHT_HIP_ROLL_ACTUATOR_INDEX] = amplitude
        assist[LEFT_HIP_ROLL_ACTUATOR_INDEX] = -amplitude
    elif support_side == "left":
        assist[RIGHT_HIP_ROLL_ACTUATOR_INDEX] = -amplitude
        assist[LEFT_HIP_ROLL_ACTUATOR_INDEX] = amplitude
    return assist


def _install_scripted_assist(env: SedonStandingEnv, amplitude: float) -> None:
    """Patch this env instance so scripted gait includes hip-roll assist directly."""
    base_gait_offsets = env._gait_joint_offsets

    def assisted_gait_offsets() -> np.ndarray:
        return base_gait_offsets() + _assist_vector(env, amplitude)

    env._gait_joint_offsets = assisted_gait_offsets  # type: ignore[method-assign]


def _actuator_ctrl_saturation(env: SedonStandingEnv) -> float:
    """Return fraction of actuators close to ctrl range limits."""
    return env._torque_saturation_fraction()


def _actuator_force_saturation(env: SedonStandingEnv) -> float:
    """Return fraction of actuators close to force range limits when available."""
    if not hasattr(env.model, "actuator_forcerange") or not hasattr(env.data, "actuator_force"):
        return 0.0
    force_range = np.asarray(env.model.actuator_forcerange, dtype=np.float64)
    actuator_force = np.asarray(env.data.actuator_force, dtype=np.float64)
    if force_range.shape[0] != actuator_force.shape[0] or actuator_force.size == 0:
        return 0.0
    span = np.maximum(force_range[:, 1] - force_range[:, 0], 1e-9)
    finite_range = np.isfinite(force_range).all(axis=1) & (span > 1e-8)
    if not np.any(finite_range):
        return 0.0
    lower_margin = np.abs(actuator_force - force_range[:, 0])
    upper_margin = np.abs(actuator_force - force_range[:, 1])
    saturated = np.minimum(lower_margin, upper_margin) <= 0.02 * span
    return float(np.count_nonzero(saturated & finite_range) / np.count_nonzero(finite_range))


def _run_amplitude(
    config: SedonStandingConfig,
    *,
    amplitude: float,
    steps: int,
    seed: int,
    max_base_drift: float,
    max_base_pitch: float,
    min_both_contact_ratio: float,
) -> SweepResult:
    """Run one zero-policy scripted assist rollout."""
    env = SedonStandingEnv(reset_noise_scale=0.0, reward_config=config)
    infos: list[dict[str, object]] = []
    ctrl_saturations: list[float] = []
    force_saturations: list[float] = []
    try:
        env.reset(seed=seed)
        _install_scripted_assist(env, amplitude)
        initial_com_y = env._overall_com_y()
        zero_action = np.zeros(env.action_space.shape, dtype=np.float64)
        for _ in range(steps):
            _, _, terminated, truncated, info = env.step(zero_action)
            infos.append(info)
            ctrl_saturations.append(_actuator_ctrl_saturation(env))
            force_saturations.append(_actuator_force_saturation(env))
            if terminated or truncated:
                break
    finally:
        env.close()

    if not infos:
        return SweepResult(
            amplitude=amplitude,
            steps=0,
            stable=False,
            pass_v1=False,
            phase1b_reachable=False,
            terminated=True,
            termination_reason="no_steps",
            max_support_ratio=0.0,
            target_side_force_ratio_mean=0.0,
            target_side_force_ratio_max=0.0,
            com_y_shift_max=0.0,
            base_y_drift_max=float("inf"),
            base_roll_abs_max=float("inf"),
            base_pitch_abs_max=float("inf"),
            both_contact_ratio=0.0,
            foot_penetration_max=float("inf"),
            actuator_ctrl_saturation_max=1.0,
            actuator_force_saturation_max=1.0,
        )

    both_contact_steps = sum(
        1 for info in infos if bool(info["right_contact"]) and bool(info["left_contact"])
    )
    foot_bottoms = [
        float(info["support_foot_bottom_z"])
        for info in infos
        if np.isfinite(float(info["support_foot_bottom_z"]))
    ] + [
        float(info["swing_foot_bottom_z"])
        for info in infos
        if np.isfinite(float(info["swing_foot_bottom_z"]))
    ]
    max_support_ratio = max(float(info["force_ratio"]) for info in infos)
    target_side_ratios = [float(info["target_side_force_ratio"]) for info in infos]
    base_y_drift_max = max(abs(float(info["base_y_position"])) for info in infos)
    base_pitch_abs_max = max(abs(float(info["base_pitch"])) for info in infos)
    both_contact_ratio = both_contact_steps / max(1, len(infos))
    terminated = str(infos[-1].get("termination_reason", "none")) != "none"
    stable = (
        not terminated
        and both_contact_ratio >= min_both_contact_ratio
        and base_y_drift_max <= max_base_drift
        and base_pitch_abs_max <= max_base_pitch
    )
    return SweepResult(
        amplitude=amplitude,
        steps=len(infos),
        stable=stable,
        pass_v1=stable and max_support_ratio >= 0.54,
        phase1b_reachable=stable and max_support_ratio >= 0.58,
        terminated=terminated,
        termination_reason=str(infos[-1].get("termination_reason", "none")),
        max_support_ratio=max_support_ratio,
        target_side_force_ratio_mean=float(np.mean(target_side_ratios)),
        target_side_force_ratio_max=float(np.max(target_side_ratios)),
        com_y_shift_max=max(abs(float(info["COM_y"]) - initial_com_y) for info in infos),
        base_y_drift_max=base_y_drift_max,
        base_roll_abs_max=max(abs(float(info["base_roll"])) for info in infos),
        base_pitch_abs_max=base_pitch_abs_max,
        both_contact_ratio=both_contact_ratio,
        foot_penetration_max=max(0.0, -min(foot_bottoms)) if foot_bottoms else 0.0,
        actuator_ctrl_saturation_max=max(ctrl_saturations) if ctrl_saturations else 0.0,
        actuator_force_saturation_max=max(force_saturations) if force_saturations else 0.0,
    )


def _write_csv(path: Path, rows: list[SweepResult]) -> None:
    """Write sweep results to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(SweepResult.__dataclass_fields__))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def _print_result(row: SweepResult) -> None:
    """Print one compact sweep result row."""
    print(
        f"amp={row.amplitude:>5.3f} stable={row.stable} "
        f"pass_v1={row.pass_v1} phase1b={row.phase1b_reachable} "
        f"steps={row.steps} term={row.terminated}:{row.termination_reason}"
    )
    print(
        "  "
        f"support max={row.max_support_ratio:.3f} "
        f"target mean/max={row.target_side_force_ratio_mean:.3f}/{row.target_side_force_ratio_max:.3f} "
        f"com_y_shift={row.com_y_shift_max:.4f} base_y_drift={row.base_y_drift_max:.4f}"
    )
    print(
        "  "
        f"roll={row.base_roll_abs_max:.4f} pitch={row.base_pitch_abs_max:.4f} "
        f"both_contact={row.both_contact_ratio:.1%} penetration={row.foot_penetration_max:.4f} "
        f"ctrl_sat={row.actuator_ctrl_saturation_max:.3f} force_sat={row.actuator_force_saturation_max:.3f}"
    )


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for the scripted hip-roll assist sweep."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amplitudes", type=float, nargs="+", default=list(DEFAULT_AMPLITUDES))
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    parser.add_argument("--max-base-drift", type=float, default=0.05)
    parser.add_argument("--max-base-pitch", type=float, default=0.35)
    parser.add_argument("--min-both-contact-ratio", type=float, default=0.95)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the scripted hip-roll assist sweep."""
    args = build_parser().parse_args(argv)
    if args.steps <= 0:
        raise ValueError("--steps must be positive.")
    config = _load_config(args.config)
    rows = [
        _run_amplitude(
            config,
            amplitude=float(amplitude),
            steps=args.steps,
            seed=args.seed,
            max_base_drift=args.max_base_drift,
            max_base_pitch=args.max_base_pitch,
            min_both_contact_ratio=args.min_both_contact_ratio,
        )
        for amplitude in args.amplitudes
    ]
    for row in rows:
        _print_result(row)
    _write_csv(args.out_csv, rows)
    print(f"\ncsv: {args.out_csv}")
    if not any(row.pass_v1 for row in rows):
        print(
            "No stable amplitude reached max_support_ratio >= 0.54. "
            "Stop PPO tuning and inspect hip_roll axis, actuator mapping, gear, "
            "forcerange, stance width, and COM lateral sensitivity."
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
