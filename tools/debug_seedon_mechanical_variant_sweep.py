"""Sweep Seedon mechanical proxy variants against a force-unload gate.

This is a decisive diagnostic: instead of retuning PPO/reward/controller on the
same failing model, it temporarily changes foot support size, base inertial COM,
and stance width in memory, then checks whether any variant can hold a stable
single-side load bias.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path

import mujoco
import numpy as np

from seedon_baseline.env import DEFAULT_SCENE_PATH, SeedonStandingConfig, SeedonStandingEnv
from tools.seedon_debug_common import (
    BASE_PROXY_GEOM,
    DEBUG_OUT_DIR,
    FLOOR_GEOM,
    LEFT_FOOT_GEOM,
    RIGHT_FOOT_GEOM,
    contact_pairs,
    geom_id,
)


DEFAULT_OUT_CSV = DEBUG_OUT_DIR / "mechanical_variant_sweep.csv"
HIP_ROLL_INDEX = {"right": 1, "left": 6}
STANCE_ROOT_BODIES = {"right": "R_link_hip_yaw", "left": "L_link_hip_yaw"}
BASE_BODY_NAME = "base_link"
FOOT_GEOM_BY_SIDE = {"left": LEFT_FOOT_GEOM, "right": RIGHT_FOOT_GEOM}


@dataclass(frozen=True)
class MechanicalVariant:
    """One temporary mechanical proxy variant."""

    support_side: str
    foot_xy_scale: float
    foot_z_scale: float
    base_com_z_offset: float
    stance_width_scale: float
    foot_friction: float
    target_base_height: float
    max_support_roll: float

    @property
    def case_name(self) -> str:
        """Return a compact stable label for reports."""
        return (
            f"{self.support_side}"
            f"__footxy_{self.foot_xy_scale:.2f}"
            f"__footz_{self.foot_z_scale:.2f}"
            f"__comz_{self.base_com_z_offset:+.3f}"
            f"__stance_{self.stance_width_scale:.2f}"
            f"__fric_{self.foot_friction:.2f}"
            f"__baseh_{self.target_base_height:.3f}"
            f"__roll_{self.max_support_roll:.2f}"
        )


@dataclass(frozen=True)
class MechanicalSweepResult:
    """One evaluated variant row."""

    case_name: str
    support_side: str
    foot_xy_scale: float
    foot_z_scale: float
    base_com_z_offset: float
    stance_width_scale: float
    foot_friction: float
    target_base_height: float
    max_support_roll: float
    steps: int
    terminated: bool
    terminated_step: int | None
    max_support_fraction: float
    mean_support_fraction_last_50: float
    min_swing_fraction_last_50: float
    stable_gate_steps: int
    longest_stable_gate_streak: int
    min_support_margin_y: float
    none_contact_steps: int
    base_proxy_floor_steps: int
    max_support_roll_cmd: float
    score: float
    diagnosis: str


def _parse_float_list(raw_value: str) -> list[float]:
    """Parse comma-separated floats."""
    values = [float(part.strip()) for part in raw_value.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one float.")
    return values


def _overall_com(env: SeedonStandingEnv) -> np.ndarray:
    """Return whole-body COM in world coordinates."""
    masses = env.model.body_mass
    total_mass = float(np.sum(masses))
    if total_mass <= 0.0:
        raise ValueError("Seedon model has no positive body mass.")
    return np.sum(env.data.xipos * masses[:, None], axis=0) / total_mass


def _foot_floor_load(env: SeedonStandingEnv, side: str) -> tuple[int, float]:
    """Return floor-contact count and summed normal force for one foot."""
    foot_geom_name = FOOT_GEOM_BY_SIDE[side]
    contact_count = 0
    normal_force_sum = 0.0
    wrench = np.zeros(6, dtype=np.float64)
    for contact_index in range(env.data.ncon):
        contact = env.data.contact[contact_index]
        name_a = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom1))
        name_b = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom2))
        if {name_a, name_b} != {FLOOR_GEOM, foot_geom_name}:
            continue
        mujoco.mj_contactForce(env.model, env.data, contact_index, wrench)
        contact_count += 1
        normal_force_sum += abs(float(wrench[0]))
    return contact_count, normal_force_sum


def _contact_state(env: SeedonStandingEnv) -> tuple[str, bool]:
    """Return compact foot-contact state and base-proxy flag."""
    left = False
    right = False
    base = False
    for name_a, name_b, _ in contact_pairs(env.model, env.data):
        pair = {name_a, name_b}
        if pair == {FLOOR_GEOM, LEFT_FOOT_GEOM}:
            left = True
        elif pair == {FLOOR_GEOM, RIGHT_FOOT_GEOM}:
            right = True
        elif pair == {FLOOR_GEOM, BASE_PROXY_GEOM}:
            base = True
    if left and right:
        return "both", base
    if left:
        return "left_only", base
    if right:
        return "right_only", base
    return "none", base


def _apply_foot_scale(env: SeedonStandingEnv, xy_scale: float, z_scale: float) -> None:
    """Scale both foot collision box sizes in memory."""
    for name in (RIGHT_FOOT_GEOM, LEFT_FOOT_GEOM):
        foot_id = geom_id(env.model, name)
        original_size = env.model.geom_size[foot_id].copy()
        env.model.geom_size[foot_id] = np.array(
            [
                original_size[0] * xy_scale,
                original_size[1] * xy_scale,
                original_size[2] * z_scale,
            ],
            dtype=np.float64,
        )
        env.model.geom_rbound[foot_id] = float(np.linalg.norm(env.model.geom_size[foot_id]))


def _apply_foot_friction(env: SeedonStandingEnv, friction: float) -> None:
    """Set main sliding friction on both foot collision boxes."""
    if friction <= 0.0:
        raise ValueError("foot friction must be positive.")
    for name in (RIGHT_FOOT_GEOM, LEFT_FOOT_GEOM):
        foot_id = geom_id(env.model, name)
        env.model.geom_friction[foot_id][0] = friction


def _apply_base_com_offset(env: SeedonStandingEnv, z_offset: float) -> None:
    """Shift the base inertial COM z location in memory."""
    base_id = env._body_id(BASE_BODY_NAME)
    env.model.body_ipos[base_id][2] += z_offset


def _apply_stance_width_scale(env: SeedonStandingEnv, scale: float) -> None:
    """Scale hip-yaw body lateral offsets in memory."""
    for side, body_name in STANCE_ROOT_BODIES.items():
        body_id = env._body_id(body_name)
        direction = -1.0 if side == "right" else 1.0
        env.model.body_pos[body_id][1] = direction * abs(float(env.model.body_pos[body_id][1])) * scale


def _support_roll_offsets(support_side: str, magnitude: float) -> tuple[float, float]:
    """Return right/left hip-roll offsets for one support side."""
    if support_side == "left":
        return magnitude, -magnitude
    if support_side == "right":
        return -magnitude, magnitude
    raise ValueError(f"Unsupported support_side: {support_side}")


def _target(env: SeedonStandingEnv, support_side: str, support_roll: float) -> np.ndarray:
    """Return a support-roll target."""
    target = env._nominal_joint_qpos.copy()
    right_roll, left_roll = _support_roll_offsets(support_side, support_roll)
    target[HIP_ROLL_INDEX["right"]] += right_roll
    target[HIP_ROLL_INDEX["left"]] += left_roll
    return env._apply_safe_joint_target_clamps(target)


def _rate_limit(value: float, previous: float, max_delta: float) -> float:
    """Limit per-step command changes."""
    return previous + float(np.clip(value - previous, -max_delta, max_delta))


def _longest_true_streak(values: list[bool]) -> int:
    """Return longest consecutive true streak."""
    longest = 0
    current = 0
    for value in values:
        if value:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _diagnosis(
    *,
    longest_streak: int,
    required_stable_steps: int,
    max_support_fraction: float,
    target_support_fraction: float,
    terminated: bool,
) -> str:
    """Return a compact diagnosis label."""
    if longest_streak >= required_stable_steps and not terminated:
        return "mechanical_variant_unloads"
    if longest_streak >= required_stable_steps:
        return "mechanical_variant_unloads_but_unstable"
    if max_support_fraction >= target_support_fraction:
        return "force_ratio_spikes_only"
    return "still_cannot_unload"


def _score(
    *,
    longest_streak: int,
    stable_gate_steps: int,
    mean_support_fraction_last_50: float,
    min_swing_fraction_last_50: float,
    none_contact_steps: int,
    base_proxy_floor_steps: int,
    terminated: bool,
) -> float:
    """Rank variants by stable unloading quality."""
    return (
        longest_streak * 10.0
        + stable_gate_steps * 2.0
        + mean_support_fraction_last_50 * 100.0
        - min_swing_fraction_last_50 * 40.0
        - none_contact_steps * 3.0
        - base_proxy_floor_steps * 8.0
        - (50.0 if terminated else 0.0)
    )


def _run_case(
    variant: MechanicalVariant,
    *,
    steps: int,
    warmup_steps: int,
    force_kp: float,
    max_roll_delta: float,
    target_support_fraction: float,
    stable_fraction_gate: float,
    swing_fraction_gate: float,
    required_stable_steps: int,
    seed: int,
    scene_path: Path | None,
) -> MechanicalSweepResult:
    """Run one mechanical variant through the force-unload gate."""
    env = SeedonStandingEnv(
        scene_path=scene_path or DEFAULT_SCENE_PATH,
        reset_noise_scale=0.0,
        reward_config=SeedonStandingConfig(
            gait_mode="fsm",
            target_base_height=variant.target_base_height,
            min_base_height=min(0.34, variant.target_base_height - 0.10),
            max_base_height=max(0.65, variant.target_base_height + 0.20),
        ),
    )
    try:
        _apply_foot_scale(env, variant.foot_xy_scale, variant.foot_z_scale)
        _apply_foot_friction(env, variant.foot_friction)
        if abs(variant.base_com_z_offset) > 1e-12:
            _apply_base_com_offset(env, variant.base_com_z_offset)
        _apply_stance_width_scale(env, variant.stance_width_scale)
        mujoco.mj_setConst(env.model, env.data)
        env.reset(seed=seed)
        mujoco.mj_forward(env.model, env.data)

        support_roll_cmd = 0.0
        support_fractions: list[float] = []
        swing_fractions: list[float] = []
        support_margins: list[float] = []
        stable_flags: list[bool] = []
        none_contact_steps = 0
        base_proxy_steps = 0
        terminated = False
        terminated_step: int | None = None

        for step in range(1, steps + 1):
            if step <= warmup_steps or not support_fractions:
                desired_roll = 0.0
            else:
                error = target_support_fraction - support_fractions[-1]
                desired_roll = float(
                    np.clip(
                        support_roll_cmd + force_kp * error,
                        0.0,
                        variant.max_support_roll,
                    )
                )
            support_roll_cmd = _rate_limit(desired_roll, support_roll_cmd, max_roll_delta)
            env._do_pd_simulation(_target(env, variant.support_side, support_roll_cmd))

            swing_side = "right" if variant.support_side == "left" else "left"
            _, support_force = _foot_floor_load(env, variant.support_side)
            _, swing_force = _foot_floor_load(env, swing_side)
            total_force = support_force + swing_force
            support_fraction = support_force / max(total_force, 1e-9)
            swing_fraction = swing_force / max(total_force, 1e-9)
            contact_state, base_proxy = _contact_state(env)
            if contact_state == "none":
                none_contact_steps += 1
            if base_proxy:
                base_proxy_steps += 1

            support_geom = geom_id(env.model, FOOT_GEOM_BY_SIDE[variant.support_side])
            support_margin = abs(float(_overall_com(env)[1]) - float(env.data.geom_xpos[support_geom][1]))
            base_z = env._base_height()
            upright = env._base_upright()
            terminated = env._is_terminated(base_z, upright, env._get_obs())

            support_fractions.append(support_fraction)
            swing_fractions.append(swing_fraction)
            support_margins.append(support_margin)
            stable_flags.append(
                support_fraction >= stable_fraction_gate
                and swing_fraction <= swing_fraction_gate
                and contact_state in ("both", f"{variant.support_side}_only")
                and not base_proxy
                and not terminated
            )
            if terminated:
                terminated_step = step
                break
    finally:
        env.close()

    tail_support = support_fractions[-50:] if support_fractions else [0.0]
    tail_swing = swing_fractions[-50:] if swing_fractions else [1.0]
    longest = _longest_true_streak(stable_flags)
    stable_steps = sum(stable_flags)
    mean_tail = float(np.mean(tail_support))
    min_swing_tail = float(min(tail_swing))
    max_support = float(max(support_fractions)) if support_fractions else 0.0
    diagnosis = _diagnosis(
        longest_streak=longest,
        required_stable_steps=required_stable_steps,
        max_support_fraction=max_support,
        target_support_fraction=target_support_fraction,
        terminated=terminated,
    )
    return MechanicalSweepResult(
        case_name=variant.case_name,
        support_side=variant.support_side,
        foot_xy_scale=variant.foot_xy_scale,
        foot_z_scale=variant.foot_z_scale,
        base_com_z_offset=variant.base_com_z_offset,
        stance_width_scale=variant.stance_width_scale,
        foot_friction=variant.foot_friction,
        target_base_height=variant.target_base_height,
        max_support_roll=variant.max_support_roll,
        steps=len(support_fractions),
        terminated=terminated,
        terminated_step=terminated_step,
        max_support_fraction=max_support,
        mean_support_fraction_last_50=mean_tail,
        min_swing_fraction_last_50=min_swing_tail,
        stable_gate_steps=stable_steps,
        longest_stable_gate_streak=longest,
        min_support_margin_y=float(min(support_margins)) if support_margins else float("nan"),
        none_contact_steps=none_contact_steps,
        base_proxy_floor_steps=base_proxy_steps,
        max_support_roll_cmd=variant.max_support_roll,
        score=_score(
            longest_streak=longest,
            stable_gate_steps=stable_steps,
            mean_support_fraction_last_50=mean_tail,
            min_swing_fraction_last_50=min_swing_tail,
            none_contact_steps=none_contact_steps,
            base_proxy_floor_steps=base_proxy_steps,
            terminated=terminated,
        ),
        diagnosis=diagnosis,
    )


def _write_csv(path: Path, rows: list[MechanicalSweepResult]) -> None:
    """Write sweep results to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        writer.writerows([asdict(row) for row in rows])


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--support-sides", default="left,right")
    parser.add_argument("--foot-xy-scales", default="1.0,1.5,2.0,2.5")
    parser.add_argument("--foot-z-scales", default="1.0")
    parser.add_argument("--base-com-z-offsets", default="0,-0.02,-0.05,-0.08")
    parser.add_argument("--stance-width-scales", default="1.0,1.2,1.5")
    parser.add_argument("--foot-frictions", default="1.0")
    parser.add_argument("--target-base-heights", default="0.446")
    parser.add_argument("--max-support-rolls", default="0.04,0.08,0.12")
    parser.add_argument("--steps", type=int, default=220)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--force-kp", type=float, default=0.08)
    parser.add_argument("--max-roll-delta", type=float, default=0.004)
    parser.add_argument("--target-support-fraction", type=float, default=0.65)
    parser.add_argument("--stable-fraction-gate", type=float, default=0.65)
    parser.add_argument("--swing-fraction-gate", type=float, default=0.35)
    parser.add_argument("--required-stable-steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scene-path", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    return parser


def _iter_variants(args: argparse.Namespace) -> list[MechanicalVariant]:
    """Expand CLI grids into variants."""
    support_sides = [part.strip() for part in args.support_sides.split(",") if part.strip()]
    invalid = sorted(set(support_sides) - {"left", "right"})
    if invalid:
        raise ValueError(f"Unsupported support side(s): {', '.join(invalid)}")
    return [
        MechanicalVariant(
            support_side=support_side,
            foot_xy_scale=foot_xy_scale,
            foot_z_scale=foot_z_scale,
            base_com_z_offset=base_com_z_offset,
            stance_width_scale=stance_width_scale,
            foot_friction=foot_friction,
            target_base_height=target_base_height,
            max_support_roll=max_support_roll,
        )
        for support_side, foot_xy_scale, foot_z_scale, base_com_z_offset, stance_width_scale, foot_friction, target_base_height, max_support_roll in product(
            support_sides,
            _parse_float_list(args.foot_xy_scales),
            _parse_float_list(args.foot_z_scales),
            _parse_float_list(args.base_com_z_offsets),
            _parse_float_list(args.stance_width_scales),
            _parse_float_list(args.foot_frictions),
            _parse_float_list(args.target_base_heights),
            _parse_float_list(args.max_support_rolls),
        )
    ]


def main(argv: list[str] | None = None) -> int:
    """Run the mechanical variant sweep."""
    args = build_parser().parse_args(argv)
    variants = _iter_variants(args)
    results = [
        _run_case(
            variant,
            steps=args.steps,
            warmup_steps=args.warmup_steps,
            force_kp=args.force_kp,
            max_roll_delta=args.max_roll_delta,
            target_support_fraction=args.target_support_fraction,
            stable_fraction_gate=args.stable_fraction_gate,
            swing_fraction_gate=args.swing_fraction_gate,
            required_stable_steps=args.required_stable_steps,
            seed=args.seed,
            scene_path=args.scene_path,
        )
        for variant in variants
    ]
    results.sort(key=lambda row: row.score, reverse=True)
    _write_csv(args.out_csv, results)

    print(
        "rank side footxy comz stance fric baseh roll max_frac mean_last50 "
        "stable longest none base term score diagnosis"
    )
    for rank, row in enumerate(results[: args.top_k], start=1):
        print(
            f"{rank:>4} {row.support_side:>5} "
            f"{row.foot_xy_scale:>6.2f} "
            f"{row.base_com_z_offset:>+6.3f} "
            f"{row.stance_width_scale:>6.2f} "
            f"{row.foot_friction:>5.2f} "
            f"{row.target_base_height:>5.3f} "
            f"{row.max_support_roll:>5.2f} "
            f"{row.max_support_fraction:>8.3f} "
            f"{row.mean_support_fraction_last_50:>11.3f} "
            f"{row.stable_gate_steps:>6} "
            f"{row.longest_stable_gate_streak:>7} "
            f"{row.none_contact_steps:>4} "
            f"{row.base_proxy_floor_steps:>4} "
            f"{str(row.terminated):>5} "
            f"{row.score:>7.1f} "
            f"{row.diagnosis}"
        )

    print(f"\nCSV: {args.out_csv}")
    if results and results[0].diagnosis == "mechanical_variant_unloads":
        print("interpretation: a mechanical proxy variant can unload; compare this variant to the baseline and remove cheats one by one.")
    elif results and results[0].max_support_fraction >= args.stable_fraction_gate:
        print("interpretation: variants can spike support force but not hold it; contact/authority remains marginal.")
    else:
        print("interpretation: even large proxy changes cannot create stable unload; inspect joint mapping/contact-force path/model setup.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
