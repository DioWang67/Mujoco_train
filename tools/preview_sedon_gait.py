"""Preview deterministic Sedon gait targets without PPO training or loading."""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np

from sedon_baseline.env import SedonStandingEnv, load_sedon_config_from_env
from tools.sedon_debug_common import DEBUG_OUT_DIR, contact_pairs


DEFAULT_OUT_CSV = DEBUG_OUT_DIR / "preview_sedon_gait.csv"


def _contact_state(env: SedonStandingEnv) -> str:
    """Return a compact floor-contact state label."""
    right = False
    left = False
    base = False
    for name_a, name_b, _ in contact_pairs(env.model, env.data):
        pair = {name_a, name_b}
        if pair == {"floor", "R_foot_collision"}:
            right = True
        elif pair == {"floor", "L_foot_collision"}:
            left = True
        elif pair == {"floor", "base_proxy"}:
            base = True
    if right and left:
        state = "both"
    elif right:
        state = "right_only"
    elif left:
        state = "left_only"
    else:
        state = "none"
    if base:
        state = f"{state}+base"
    return state


def _blue_step_phase(env: SedonStandingEnv) -> tuple[str, float]:
    """Return preview-only blue-step phase name and normalized local alpha."""
    cycle_steps = max(1, env._reward_config.gait_cycle_steps)
    phase = (env._gait_step % cycle_steps) / cycle_steps
    boundaries = (
        ("center_hold_right", 0.14),
        ("right_lift", 0.18),
        ("right_lower", 0.18),
        ("center_hold_left", 0.14),
        ("left_lift", 0.18),
        ("left_lower", 0.18),
    )
    cursor = 0.0
    for phase_name, span in boundaries:
        next_cursor = cursor + span
        if phase < next_cursor:
            local_alpha = (phase - cursor) / span if span > 0.0 else 0.0
            return phase_name, local_alpha
        cursor = next_cursor
    return "left_lower", 1.0


def _blue_step_joint_offsets(env: SedonStandingEnv) -> tuple[np.ndarray, str]:
    """Return a preview-only blue-style deterministic gait target."""
    cfg = env._reward_config
    phase_name, alpha = _blue_step_phase(env)
    swing = env._smoothstep(alpha)
    reverse_swing = 1.0 - swing
    settle = 1.0 - abs(2.0 * alpha - 1.0)

    roll_amp = max(abs(cfg.gait_hip_roll_amp), abs(cfg.com_shift_support_roll_amp), 0.060)
    hip_amp = cfg.gait_hip_pitch_amp if abs(cfg.gait_hip_pitch_amp) > 1e-9 else -0.035
    knee_amp = cfg.gait_knee_pitch_amp if abs(cfg.gait_knee_pitch_amp) > 1e-9 else -0.100
    ankle_amp = cfg.gait_ankle_pitch_amp if abs(cfg.gait_ankle_pitch_amp) > 1e-9 else 0.080
    settle_scale = 0.18
    support_relax_knee = 0.020
    support_relax_ankle = -0.012

    offsets = np.zeros_like(env._nominal_joint_qpos)
    offsets[2] += hip_amp * settle_scale * settle
    offsets[3] += knee_amp * settle_scale * settle
    offsets[4] += ankle_amp * settle_scale * settle
    offsets[7] += hip_amp * settle_scale * settle
    offsets[8] += knee_amp * settle_scale * settle
    offsets[9] += ankle_amp * settle_scale * settle

    def apply_right_support(scale: float) -> None:
        offsets[1] += -roll_amp * scale
        offsets[6] += roll_amp * scale
        offsets[8] += support_relax_knee * scale
        offsets[9] += support_relax_ankle * scale

    def apply_left_support(scale: float) -> None:
        offsets[1] += roll_amp * scale
        offsets[6] += -roll_amp * scale
        offsets[3] += support_relax_knee * scale
        offsets[4] += support_relax_ankle * scale

    if phase_name == "right_lift":
        apply_left_support(swing)
        offsets[2] += hip_amp * swing
        offsets[3] += knee_amp * swing
        offsets[4] += ankle_amp * swing
    elif phase_name == "right_lower":
        apply_left_support(reverse_swing)
        offsets[2] += hip_amp * reverse_swing
        offsets[3] += knee_amp * reverse_swing
        offsets[4] += ankle_amp * reverse_swing
    elif phase_name == "left_lift":
        apply_right_support(swing)
        offsets[7] += hip_amp * swing
        offsets[8] += knee_amp * swing
        offsets[9] += ankle_amp * swing
    elif phase_name == "left_lower":
        apply_right_support(reverse_swing)
        offsets[7] += hip_amp * reverse_swing
        offsets[8] += knee_amp * reverse_swing
        offsets[9] += ankle_amp * reverse_swing

    return offsets, phase_name


def _target_positions(env: SedonStandingEnv, gait_mode: str) -> tuple[np.ndarray, str]:
    """Return deterministic target positions and current phase label."""
    nominal = env._nominal_joint_qpos
    if gait_mode == "blue_step":
        offsets, phase_name = _blue_step_joint_offsets(env)
        return env._apply_safe_joint_target_clamps(nominal + offsets), phase_name
    if gait_mode == "fsm":
        return env._apply_safe_joint_target_clamps(nominal + env._gait_joint_offsets()), "fsm"
    if gait_mode == "com_shift":
        phase_name = str(env._task_phase_metadata()["phase_name"])
        return env._apply_safe_joint_target_clamps(nominal + env._gait_joint_offsets()), phase_name
    raise ValueError(f"Unsupported gait_mode: {gait_mode}")


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    """Write preview rows to CSV when available."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _physics_mode_label(*, fixed_base: bool, no_floor: bool) -> str:
    """Return a compact physics-mode label."""
    if fixed_base:
        return "fixed_base"
    if no_floor:
        return "no_floor"
    return "with_floor"


def _pin_base(env: SedonStandingEnv, base_qpos: np.ndarray) -> None:
    """Restore the floating base pose and clear floating-base velocity."""
    env.data.qpos[0:7] = base_qpos
    env.data.qvel[0:6] = 0.0


def _apply_target_step(
    env: SedonStandingEnv,
    target_positions: np.ndarray,
    *,
    fixed_base: bool,
    fixed_base_qpos: np.ndarray | None,
) -> None:
    """Run one preview step while optionally pinning the floating base."""
    for _ in range(env.frame_skip):
        env.data.ctrl[:] = env._pd_control(target_positions)
        mujoco.mj_step(env.model, env.data)
        if fixed_base:
            if fixed_base_qpos is None:
                raise ValueError("fixed_base_qpos is required when fixed_base=True.")
            _pin_base(env, fixed_base_qpos)
            mujoco.mj_forward(env.model, env.data)


def _set_floor_contact_enabled(env: SedonStandingEnv, enabled: bool) -> tuple[int, int]:
    """Enable or disable floor contact and return the previous flags."""
    floor_geom_id = env._geom_id("floor")
    previous = (
        int(env.model.geom_contype[floor_geom_id]),
        int(env.model.geom_conaffinity[floor_geom_id]),
    )
    if enabled:
        env.model.geom_contype[floor_geom_id] = previous[0]
        env.model.geom_conaffinity[floor_geom_id] = previous[1]
    else:
        env.model.geom_contype[floor_geom_id] = 0
        env.model.geom_conaffinity[floor_geom_id] = 0
    mujoco.mj_forward(env.model, env.data)
    return previous


def _restore_floor_contact(env: SedonStandingEnv, previous: tuple[int, int]) -> None:
    """Restore the floor contact flags."""
    floor_geom_id = env._geom_id("floor")
    env.model.geom_contype[floor_geom_id] = previous[0]
    env.model.geom_conaffinity[floor_geom_id] = previous[1]
    mujoco.mj_forward(env.model, env.data)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gait-mode", choices=("fsm", "blue_step", "com_shift"), default="fsm")
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    parser.add_argument("--render-viewer", action="store_true")
    parser.add_argument("--viewer-sleep", type=float, default=0.0)
    physics_group = parser.add_mutually_exclusive_group()
    physics_group.add_argument("--fixed-base", action="store_true")
    physics_group.add_argument("--no-floor", action="store_true")
    return parser


def _run_preview_step(
    env: SedonStandingEnv,
    gait_mode: str,
    *,
    fixed_base: bool,
    fixed_base_qpos: np.ndarray | None,
    physics_mode: str,
) -> dict[str, object]:
    """Advance one deterministic target-tracking step and return diagnostics."""
    target_positions, phase_name = _target_positions(env, gait_mode)
    _apply_target_step(
        env,
        target_positions,
        fixed_base=fixed_base,
        fixed_base_qpos=fixed_base_qpos,
    )
    obs = env._get_obs()
    base_z = env._base_height()
    upright = env._base_upright()
    terminated = env._is_terminated(base_z, upright, obs)
    joint_positions = env._joint_positions()
    foot_bottoms = env._foot_bottom_heights()
    row = {
        "step": 0,
        "gait_mode": gait_mode,
        "physics_mode": physics_mode,
        "phase_name": phase_name,
        "target_right_knee_qpos": float(target_positions[3]),
        "target_left_knee_qpos": float(target_positions[8]),
        "right_knee_qpos": float(joint_positions[3]),
        "left_knee_qpos": float(joint_positions[8]),
        "right_foot_bottom_z": float(foot_bottoms[0]),
        "left_foot_bottom_z": float(foot_bottoms[1]),
        "contact_state": _contact_state(env),
        "base_z": float(base_z),
        "upright": float(upright),
        "terminated": bool(terminated),
    }
    env._gait_step += 1
    return row


def _run_loop(
    env: SedonStandingEnv,
    *,
    gait_mode: str,
    steps: int,
    seed: int,
    out_csv: Path,
    render_viewer: bool,
    viewer_sleep: float,
    fixed_base: bool,
    no_floor: bool,
) -> list[dict[str, object]]:
    """Run one preview rollout with the selected physics mode."""
    rows: list[dict[str, object]] = []
    env.reset(seed=seed)
    physics_mode = _physics_mode_label(fixed_base=fixed_base, no_floor=no_floor)
    fixed_base_qpos = env.data.qpos[0:7].copy() if fixed_base else None
    floor_previous = None
    if no_floor:
        floor_previous = _set_floor_contact_enabled(env, enabled=False)

    print("step phase R_knee_qpos L_knee_qpos R_foot_z L_foot_z contact_state base_z upright")
    try:
        viewer = None
        if render_viewer:
            try:
                import mujoco.viewer
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(
                    "mujoco.viewer is unavailable in this Python environment. "
                    "Run without --render-viewer for headless preview."
                ) from exc
            viewer = mujoco.viewer.launch_passive(env.model, env.data)
            viewer.__enter__()

        try:
            for step in range(1, steps + 1):
                row = _run_preview_step(
                    env,
                    gait_mode,
                    fixed_base=fixed_base,
                    fixed_base_qpos=fixed_base_qpos,
                    physics_mode=physics_mode,
                )
                row["step"] = step
                rows.append(row)
                print(
                    f"{step:>4} {row['phase_name']:>16} "
                    f"{row['right_knee_qpos']:>11.4f} {row['left_knee_qpos']:>11.4f} "
                    f"{row['right_foot_bottom_z']:>8.4f} {row['left_foot_bottom_z']:>8.4f} "
                    f"{row['contact_state']:>12} {row['base_z']:>7.4f} {row['upright']:>7.4f}"
                )
                if viewer is not None:
                    viewer.sync()
                    if viewer_sleep > 0.0:
                        time.sleep(viewer_sleep)
                    if not viewer.is_running():
                        break
                if row["terminated"]:
                    break
        finally:
            if viewer is not None:
                viewer.__exit__(None, None, None)
    finally:
        if floor_previous is not None:
            _restore_floor_contact(env, floor_previous)

    _write_rows(out_csv, rows)
    return rows


def main(argv: list[str] | None = None) -> int:
    """Run the deterministic gait preview."""
    args = build_parser().parse_args(argv)
    if args.steps <= 0:
        raise ValueError("--steps must be positive.")
    if args.viewer_sleep < 0.0:
        raise ValueError("--viewer-sleep must be non-negative.")

    base_config = load_sedon_config_from_env()
    if args.gait_mode == "com_shift":
        reward_config = replace(base_config, task_mode="com_shift", gait_mode="fsm")
    else:
        reward_config = replace(base_config, task_mode="walk", gait_mode="fsm")

    env = SedonStandingEnv(reset_noise_scale=0.0, reward_config=reward_config)
    try:
        rows = _run_loop(
            env,
            gait_mode=args.gait_mode,
            steps=args.steps,
            seed=args.seed,
            out_csv=args.out_csv,
            render_viewer=args.render_viewer,
            viewer_sleep=args.viewer_sleep,
            fixed_base=args.fixed_base,
            no_floor=args.no_floor,
        )
    finally:
        env.close()

    print(f"\ncsv: {args.out_csv}")
    print(f"steps: {len(rows)}")
    if rows:
        print(
            "final: "
            f"physics={rows[-1]['physics_mode']} "
            f"phase={rows[-1]['phase_name']} "
            f"R_knee={rows[-1]['right_knee_qpos']:.4f} "
            f"L_knee={rows[-1]['left_knee_qpos']:.4f} "
            f"base_z={rows[-1]['base_z']:.4f} "
            f"upright={rows[-1]['upright']:.4f} "
            f"contact={rows[-1]['contact_state']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
