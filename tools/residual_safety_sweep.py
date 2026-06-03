"""Sweep residual action patterns against the Seedon teacher reference."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from seedon_baseline.env import SeedonStandingConfig, SeedonStandingEnv
from tools.audit_seedon_shuffle_v0 import _count_contact_none_bursts, _load_config


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "seedon" / "reference_teacher_pose_1_4_imitation.json"
DEFAULT_OUT_CSV = REPO_ROOT / "artifacts" / "seedon_debug" / "residual_safety_sweep.csv"
DEFAULT_SCALES = (0.001, 0.002, 0.003, 0.005, 0.008)
DEFAULT_LOW_FREQUENCY_INTERVALS = (10, 20, 40)
DEFAULT_GAUSSIAN_SIGMAS = (0.05, 0.1, 0.2)
DEFAULT_LANDING_IMPACT_MULTIPLIER = 1.15


@dataclass(frozen=True)
class ResidualSweepCase:
    """One residual safety sweep configuration.

    Args:
        mode: Residual action pattern name.
        action_joint_delta_scale: Physical joint target scale applied by the env.
        update_interval: Step hold interval for low-frequency residuals.
        gaussian_sigma: Standard deviation for Gaussian residuals.
    """

    mode: str
    action_joint_delta_scale: float
    update_interval: int | None = None
    gaussian_sigma: float | None = None


@dataclass(frozen=True)
class ResidualSweepRow:
    """One residual safety audit result row."""

    mode: str
    action_joint_delta_scale: float
    update_interval: int | None
    gaussian_sigma: float | None
    seed: int
    steps: int
    terminated: bool
    termination_reason: str
    contact_none_ratio: float
    jump_count: int
    peak_support_ratio: float
    clearance: float
    base_height_drop: float
    landing_impact: float
    mean_tracking_error: float
    max_tracking_error: float
    teacher_passed: bool


class ResidualActionGenerator:
    """Generate normalized residual actions for one rollout.

    Args:
        case: Residual sweep case to generate.
        action_shape: Gym action shape expected by the Seedon environment.
        seed: Random seed for reproducible residual sequences.

    Raises:
        ValueError: If the case mode is unknown or missing required parameters.
    """

    def __init__(
        self,
        case: ResidualSweepCase,
        action_shape: tuple[int, ...],
        seed: int,
    ) -> None:
        self._case = case
        self._action_shape = action_shape
        self._rng = np.random.default_rng(seed)
        self._held_action = np.zeros(action_shape, dtype=np.float64)

        if case.mode == "constant":
            self._held_action = self._uniform_action()
        elif case.mode == "low_frequency":
            if case.update_interval is None or case.update_interval <= 0:
                raise ValueError("low_frequency residual requires a positive update_interval.")
        elif case.mode == "gaussian":
            if case.gaussian_sigma is None or case.gaussian_sigma <= 0.0:
                raise ValueError("gaussian residual requires a positive gaussian_sigma.")
        elif case.mode == "filtered":
            pass
        else:
            raise ValueError(f"Unknown residual mode: {case.mode}")

    def action_at_step(self, step: int) -> np.ndarray:
        """Return the normalized residual action for a rollout step."""
        if self._case.mode == "constant":
            return self._held_action.copy()
        if self._case.mode == "low_frequency":
            if step % int(self._case.update_interval) == 0:
                self._held_action = self._uniform_action()
            return self._held_action.copy()
        if self._case.mode == "gaussian":
            action = self._rng.normal(
                loc=0.0,
                scale=float(self._case.gaussian_sigma),
                size=self._action_shape,
            )
            return np.clip(action, -1.0, 1.0).astype(np.float64)
        if self._case.mode == "filtered":
            noise = self._uniform_action()
            self._held_action = 0.9 * self._held_action + 0.1 * noise
            return np.clip(self._held_action, -1.0, 1.0).astype(np.float64)
        raise ValueError(f"Unknown residual mode: {self._case.mode}")

    def _uniform_action(self) -> np.ndarray:
        """Return one normalized uniform residual action."""
        return self._rng.uniform(-1.0, 1.0, size=self._action_shape).astype(np.float64)


def build_sweep_cases(
    scales: tuple[float, ...] = DEFAULT_SCALES,
    low_frequency_intervals: tuple[int, ...] = DEFAULT_LOW_FREQUENCY_INTERVALS,
    gaussian_sigmas: tuple[float, ...] = DEFAULT_GAUSSIAN_SIGMAS,
) -> list[ResidualSweepCase]:
    """Build the default residual safety sweep matrix.

    Args:
        scales: Physical action scales to test.
        low_frequency_intervals: Hold intervals for low-frequency residuals.
        gaussian_sigmas: Standard deviations for Gaussian residuals.

    Returns:
        Ordered list of residual sweep cases.
    """
    cases: list[ResidualSweepCase] = []
    for scale in scales:
        cases.append(ResidualSweepCase("constant", scale))
        cases.extend(
            ResidualSweepCase("low_frequency", scale, update_interval=interval)
            for interval in low_frequency_intervals
        )
        cases.extend(
            ResidualSweepCase("gaussian", scale, gaussian_sigma=sigma)
            for sigma in gaussian_sigmas
        )
        cases.append(ResidualSweepCase("filtered", scale))
    return cases


def _audit_rollout(
    config: SeedonStandingConfig,
    case: ResidualSweepCase,
    *,
    steps: int,
    seed: int,
    landing_impact_limit: float,
) -> ResidualSweepRow:
    """Run one residual rollout and aggregate teacher safety metrics."""
    env = SeedonStandingEnv(reset_noise_scale=0.0, reward_config=config)
    obs, _ = env.reset(seed=seed)
    generator = ResidualActionGenerator(case, env.action_space.shape, seed)
    infos: list[dict[str, object]] = []
    terminated = False
    termination_reason = "none"
    try:
        for step in range(steps):
            action = generator.action_at_step(step)
            obs, _, terminated, _, info = env.step(action)
            infos.append(info)
            termination_reason = str(info.get("termination_reason", "none"))
            if terminated:
                break

        return _summarize_infos(
            env,
            case,
            infos,
            seed=seed,
            terminated=terminated,
            termination_reason=termination_reason,
            landing_impact_limit=landing_impact_limit,
        )
    finally:
        env.close()


def _summarize_infos(
    env: SeedonStandingEnv,
    case: ResidualSweepCase,
    infos: list[dict[str, object]],
    *,
    seed: int,
    terminated: bool,
    termination_reason: str,
    landing_impact_limit: float,
) -> ResidualSweepRow:
    """Aggregate raw rollout infos into a residual sweep row."""
    if not infos:
        return ResidualSweepRow(
            mode=case.mode,
            action_joint_delta_scale=case.action_joint_delta_scale,
            update_interval=case.update_interval,
            gaussian_sigma=case.gaussian_sigma,
            seed=seed,
            steps=0,
            terminated=True,
            termination_reason="no_steps",
            contact_none_ratio=1.0,
            jump_count=1,
            peak_support_ratio=0.0,
            clearance=0.0,
            base_height_drop=float("inf"),
            landing_impact=float("inf"),
            mean_tracking_error=float("inf"),
            max_tracking_error=float("inf"),
            teacher_passed=False,
        )

    total_steps = max(1, len(infos))
    none_steps = sum(
        1
        for info in infos
        if not bool(info["right_contact"]) and not bool(info["left_contact"])
    )
    total_weight = float(np.sum(env.model.body_mass) * 9.81)
    initial_base_height = float(infos[0]["base_height"])
    tracking_errors = [
        float(
            np.sqrt(
                float(info.get("joint_position_error_l2", 0.0))
                / max(int(env.action_space.shape[0]), 1)
            )
        )
        for info in infos
    ]
    contact_none_ratio = none_steps / total_steps
    peak_support_ratio = max(float(info["force_ratio"]) for info in infos)
    clearance = max(float(info["foot_clearance"]) for info in infos)
    base_height_drop = max(
        0.0,
        initial_base_height - min(float(info["base_height"]) for info in infos),
    )
    landing_impact = max(
        (float(info["left_normal_force"]) + float(info["right_normal_force"]))
        / max(total_weight, 1e-6)
        for info in infos
    )
    jump_count = _count_contact_none_bursts(infos)
    teacher_passed = (
        contact_none_ratio == 0.0
        and jump_count == 0
        and peak_support_ratio >= 0.58
        and clearance >= 0.0008
        and base_height_drop <= 0.025
        and landing_impact <= landing_impact_limit
    )
    return ResidualSweepRow(
        mode=case.mode,
        action_joint_delta_scale=case.action_joint_delta_scale,
        update_interval=case.update_interval,
        gaussian_sigma=case.gaussian_sigma,
        seed=seed,
        steps=len(infos),
        terminated=bool(terminated),
        termination_reason=termination_reason,
        contact_none_ratio=float(contact_none_ratio),
        jump_count=jump_count,
        peak_support_ratio=peak_support_ratio,
        clearance=clearance,
        base_height_drop=base_height_drop,
        landing_impact=landing_impact,
        mean_tracking_error=float(np.mean(tracking_errors)) if tracking_errors else 0.0,
        max_tracking_error=float(max(tracking_errors, default=0.0)),
        teacher_passed=teacher_passed,
    )


def run_sweep(
    config_path: Path,
    *,
    steps: int,
    seed: int,
    landing_impact_multiplier: float,
) -> list[ResidualSweepRow]:
    """Run the full residual safety sweep.

    Args:
        config_path: Teacher reference config path.
        steps: Maximum rollout steps per case.
        seed: Random seed reused for deterministic case comparison.
        landing_impact_multiplier: Allowed multiple of teacher baseline landing impact.

    Returns:
        Sweep result rows in deterministic matrix order.
    """
    base_config = _load_config(config_path)
    baseline_case = ResidualSweepCase("constant", base_config.action_joint_delta_scale)
    baseline_config = replace(base_config, action_joint_delta_scale=0.0)
    baseline = _audit_rollout(
        baseline_config,
        baseline_case,
        steps=steps,
        seed=seed,
        landing_impact_limit=float("inf"),
    )
    landing_impact_limit = baseline.landing_impact * landing_impact_multiplier

    rows: list[ResidualSweepRow] = []
    for case in build_sweep_cases():
        config = replace(base_config, action_joint_delta_scale=case.action_joint_delta_scale)
        row = _audit_rollout(
            config,
            case,
            steps=steps,
            seed=seed,
            landing_impact_limit=landing_impact_limit,
        )
        rows.append(row)
    return rows


def _write_csv(path: Path, rows: list[ResidualSweepRow]) -> None:
    """Write residual sweep rows as CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--steps", type=int, default=480)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    parser.add_argument(
        "--landing-impact-multiplier",
        type=float,
        default=DEFAULT_LANDING_IMPACT_MULTIPLIER,
        help="Allowed landing impact multiple relative to zero-residual teacher baseline.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run residual safety sweep and write CSV output."""
    args = build_parser().parse_args(argv)
    rows = run_sweep(
        args.config,
        steps=args.steps,
        seed=args.seed,
        landing_impact_multiplier=args.landing_impact_multiplier,
    )
    _write_csv(args.out_csv, rows)
    passed = sum(1 for row in rows if row.teacher_passed)
    print(f"wrote: {args.out_csv}")
    print(f"passed: {passed}/{len(rows)}")
    for row in rows:
        status = "PASS" if row.teacher_passed else "FAIL"
        detail = row.mode
        if row.update_interval is not None:
            detail += f":{row.update_interval}"
        if row.gaussian_sigma is not None:
            detail += f":sigma={row.gaussian_sigma}"
        print(
            f"{status} scale={row.action_joint_delta_scale:.3f} {detail} "
            f"none={row.contact_none_ratio:.3f} jump={row.jump_count} "
            f"support={row.peak_support_ratio:.3f} clearance={row.clearance:.6f} "
            f"drop={row.base_height_drop:.5f} impact={row.landing_impact:.3f} "
            f"track={row.mean_tracking_error:.5f}/{row.max_tracking_error:.5f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
