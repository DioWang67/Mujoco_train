"""Sweep temporary Sedon foot proxy redesigns for lateral load-transfer diagnostics.

This tool does not modify the committed training scene. It creates temporary
XML copies with alternative foot contact proxies, then evaluates each proxy
with a simple COM-feedback balance controller.
"""

from __future__ import annotations

import argparse
import csv
import math
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path

import mujoco
import numpy as np

from sedon_baseline.env import SedonStandingEnv, load_sedon_config_from_env
from tools.sedon_debug_common import (
    BASE_PROXY_GEOM,
    DEBUG_OUT_DIR,
    DEFAULT_SCENE_PATH,
    FLOOR_GEOM,
    LEFT_FOOT_GEOM,
    RIGHT_FOOT_GEOM,
    contact_pairs,
)


DEFAULT_OUT_CSV = DEBUG_OUT_DIR / "foot_proxy_redesign_sweep.csv"
HIP_ROLL_INDEX = {"right": 1, "left": 6}
KNEE_PITCH_INDEX = {"right": 3, "left": 8}
SUPPORT_TO_SWING = {"left": "right", "right": "left"}
PROXY_CASES = (
    "original_box",
    "four_corner_spheres",
    "toe_heel_boxes",
    "rounded_sole_proxy",
    "lateral_edge_boxes",
)


@dataclass(frozen=True)
class ProxyCaseDefinition:
    """Temporary XML metadata for one proxy case."""

    case_name: str
    proxy_type: str
    scene_path: Path
    contact_geom_names: dict[str, tuple[str, ...]]


@dataclass(frozen=True)
class ProxySweepResult:
    """One foot-proxy redesign result row."""

    case_name: str
    proxy_type: str
    side: str
    terminated: bool
    terminated_step: int
    initial_contact_count: int
    settled_contact_count: int
    both_contact_ratio: float
    support_only_steps: int
    none_contact_ratio: float
    max_support_force_ratio: float
    mean_support_force_ratio_last_50: float
    mean_swing_force_ratio_last_50: float
    max_abs_com_y_delta: float
    mean_support_com_shift_last_50: float
    max_base_roll_delta: float
    left_normal_force: float
    right_normal_force: float
    score: float


def _overall_com(env: SedonStandingEnv) -> np.ndarray:
    """Return whole-body COM in world coordinates."""
    masses = env.model.body_mass
    total_mass = float(np.sum(masses))
    if total_mass <= 0.0:
        raise ValueError("Model has no positive body mass.")
    return np.sum(env.data.xipos * masses[:, None], axis=0) / total_mass


def _quat_to_roll(quat: np.ndarray) -> float:
    """Return roll angle from a MuJoCo quaternion."""
    w, x, y, z = [float(value) for value in quat]
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    return float(math.atan2(sinr_cosp, cosr_cosp))


def _vector_to_text(values: tuple[float, ...]) -> str:
    """Convert a numeric tuple to MuJoCo XML vector text."""
    return " ".join(f"{value:.6f}" for value in values)


def _parse_vector(raw_value: str) -> tuple[float, ...]:
    """Parse a MuJoCo XML vector attribute."""
    return tuple(float(part) for part in raw_value.split())


def _brace_target(env: SedonStandingEnv, support_side: str) -> np.ndarray:
    """Return the conservative brace pose used in prior feedback diagnostics."""
    swing_side = SUPPORT_TO_SWING[support_side]
    target = env._nominal_joint_qpos.copy()
    target[KNEE_PITCH_INDEX[support_side]] += 0.04
    target[KNEE_PITCH_INDEX[swing_side]] += -0.06
    return target


def _apply_support_roll(target: np.ndarray, support_side: str, magnitude: float) -> None:
    """Apply mirrored hip-roll offsets that bias support onto one foot."""
    if support_side == "left":
        target[HIP_ROLL_INDEX["right"]] += magnitude
        target[HIP_ROLL_INDEX["left"]] -= magnitude
        return
    if support_side == "right":
        target[HIP_ROLL_INDEX["right"]] -= magnitude
        target[HIP_ROLL_INDEX["left"]] += magnitude
        return
    raise ValueError(f"Unsupported support side: {support_side}")


def _contact_state(env: SedonStandingEnv, contact_geom_names: dict[str, tuple[str, ...]]) -> tuple[str, bool]:
    """Return compact foot-contact state plus base-proxy floor flag."""
    left_names = set(contact_geom_names["left"])
    right_names = set(contact_geom_names["right"])
    left = False
    right = False
    base = False
    for name_a, name_b, _ in contact_pairs(env.model, env.data):
        pair = {name_a, name_b}
        if FLOOR_GEOM not in pair:
            continue
        other = next(iter(pair - {FLOOR_GEOM}))
        if other in left_names:
            left = True
        elif other in right_names:
            right = True
        elif other == BASE_PROXY_GEOM:
            base = True
    if left and right:
        return "both", base
    if left:
        return "left_only", base
    if right:
        return "right_only", base
    return "none", base


def _foot_floor_load(
    env: SedonStandingEnv,
    geom_names: tuple[str, ...],
) -> tuple[int, float]:
    """Return floor-contact count and normal-force sum for a foot proxy group."""
    contact_count = 0
    normal_force_sum = 0.0
    wanted_names = set(geom_names)
    wrench = np.zeros(6, dtype=np.float64)
    for contact_index in range(env.data.ncon):
        contact = env.data.contact[contact_index]
        name_a = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom1))
        name_b = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom2))
        pair = {name_a, name_b}
        if FLOOR_GEOM not in pair:
            continue
        other = next(iter(pair - {FLOOR_GEOM}))
        if other not in wanted_names:
            continue
        mujoco.mj_contactForce(env.model, env.data, contact_index, wrench)
        contact_count += 1
        normal_force_sum += abs(float(wrench[0]))
    return contact_count, normal_force_sum


def _total_proxy_contact_count(env: SedonStandingEnv, contact_geom_names: dict[str, tuple[str, ...]]) -> int:
    """Return total active floor contacts across both foot proxy groups."""
    left_count, _ = _foot_floor_load(env, contact_geom_names["left"])
    right_count, _ = _foot_floor_load(env, contact_geom_names["right"])
    return left_count + right_count


def _score_result(
    *,
    max_support_force_ratio: float,
    mean_support_force_ratio_last_50: float,
    mean_support_com_shift_last_50: float,
    both_contact_ratio: float,
    none_contact_ratio: float,
    terminated: bool,
) -> float:
    """Return a ranking score that favors stable usable support transfer."""
    threshold_bonus = 1000.0 if (
        not terminated
        and mean_support_force_ratio_last_50 > 0.65
        and mean_support_com_shift_last_50 > 0.008
    ) else 0.0
    return (
        threshold_bonus
        + max_support_force_ratio * 80.0
        + mean_support_force_ratio_last_50 * 120.0
        + mean_support_com_shift_last_50 * 1400.0
        + both_contact_ratio * 12.0
        - none_contact_ratio * 30.0
        - (80.0 if terminated else 0.0)
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settle-steps", type=int, default=20)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-support-com-shift", type=float, default=0.005)
    parser.add_argument("--com-kp", type=float, default=12.0)
    parser.add_argument("--com-kd", type=float, default=0.2)
    parser.add_argument("--roll-kp", type=float, default=0.0)
    parser.add_argument("--max-support-roll", type=float, default=0.12)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    return parser


def _set_meshdir_absolute(root: ET.Element) -> None:
    """Point the temporary XML compiler meshdir at the original mesh source."""
    compiler = root.find("compiler")
    if compiler is None:
        raise ValueError("Expected <compiler> element in Sedon training scene.")
    meshdir = (DEFAULT_SCENE_PATH.parent / "mjcf_source").resolve()
    compiler.set("meshdir", meshdir.as_posix())


def _deactivate_original_contact_geom(geom_element: ET.Element) -> None:
    """Keep the original named geom for env bookkeeping but disable contact."""
    geom_element.set("contype", "0")
    geom_element.set("conaffinity", "0")
    geom_element.set("rgba", "0.12 0.12 0.12 0")


def _append_geom(parent: ET.Element, **attributes: str) -> None:
    """Append one MuJoCo geom element."""
    ET.SubElement(parent, "geom", attrib=attributes)


def _build_replacement_geoms(
    side: str,
    proxy_type: str,
    *,
    original_pos: tuple[float, float, float],
    original_size: tuple[float, float, float],
    friction_text: str,
) -> tuple[list[dict[str, str]], tuple[str, ...]]:
    """Return replacement geoms and the active contact-geom names for one foot."""
    if proxy_type == "original_box":
        original_name = RIGHT_FOOT_GEOM if side == "right" else LEFT_FOOT_GEOM
        return [], (original_name,)

    original_name = RIGHT_FOOT_GEOM if side == "right" else LEFT_FOOT_GEOM
    prefix = "R" if side == "right" else "L"
    x_center, y_center, z_center = original_pos
    half_x, half_y, half_z = original_size
    bottom_z = z_center - half_z
    contact_geoms: list[dict[str, str]] = []
    active_names: list[str] = []

    def add(name_suffix: str, **attrs: str) -> None:
        name = f"{prefix}_foot_proxy_{name_suffix}"
        attrs["name"] = name
        attrs["friction"] = friction_text
        attrs["rgba"] = "0.12 0.12 0.12 0"
        contact_geoms.append(attrs)
        active_names.append(name)

    if proxy_type == "four_corner_spheres":
        radius = min(half_y, half_z) * 0.35
        sphere_z = bottom_z + radius
        for label, x_offset, y_offset in (
            ("fl", half_x * 0.65, half_y * 0.75),
            ("fr", half_x * 0.65, -half_y * 0.75),
            ("hl", -half_x * 0.65, half_y * 0.75),
            ("hr", -half_x * 0.65, -half_y * 0.75),
        ):
            add(
                label,
                type="sphere",
                pos=_vector_to_text((x_center + x_offset, y_center + y_offset, sphere_z)),
                size=f"{radius:.6f}",
            )
    elif proxy_type == "toe_heel_boxes":
        add(
            "heel",
            type="box",
            pos=_vector_to_text((x_center - half_x * 0.42, y_center, z_center)),
            size=_vector_to_text((half_x * 0.38, half_y * 0.78, half_z)),
        )
        add(
            "toe",
            type="box",
            pos=_vector_to_text((x_center + half_x * 0.42, y_center, z_center)),
            size=_vector_to_text((half_x * 0.46, half_y * 0.78, half_z)),
        )
    elif proxy_type == "rounded_sole_proxy":
        radius = half_z * 0.9
        capsule_z = bottom_z + radius
        add(
            "capsule",
            type="capsule",
            fromto=_vector_to_text(
                (
                    x_center - half_x * 0.70,
                    y_center,
                    capsule_z,
                    x_center + half_x * 0.70,
                    y_center,
                    capsule_z,
                )
            ),
            size=f"{radius:.6f}",
        )
    elif proxy_type == "lateral_edge_boxes":
        edge_half_y = max(0.010, half_y * 0.28)
        edge_offset = half_y * 0.72
        add(
            "outer",
            type="box",
            pos=_vector_to_text((x_center, y_center + edge_offset, z_center)),
            size=_vector_to_text((half_x * 0.92, edge_half_y, half_z)),
        )
        add(
            "inner",
            type="box",
            pos=_vector_to_text((x_center, y_center - edge_offset, z_center)),
            size=_vector_to_text((half_x * 0.92, edge_half_y, half_z)),
        )
    else:
        raise ValueError(f"Unsupported proxy_type: {proxy_type}")

    return contact_geoms, tuple(active_names)


def _make_case_definition(proxy_type: str, temp_dir: Path) -> ProxyCaseDefinition:
    """Create one temporary XML scene for a proxy redesign case."""
    tree = ET.parse(DEFAULT_SCENE_PATH)
    root = tree.getroot()
    _set_meshdir_absolute(root)

    contact_geom_names: dict[str, tuple[str, ...]] = {}
    for side, geom_name in (("right", RIGHT_FOOT_GEOM), ("left", LEFT_FOOT_GEOM)):
        body = None
        original_geom = None
        for candidate_body in root.findall(".//body"):
            for geom in candidate_body.findall("geom"):
                if geom.get("name") == geom_name:
                    body = candidate_body
                    original_geom = geom
                    break
            if body is not None and original_geom is not None:
                break
        if body is None or original_geom is None:
            raise ValueError(f"Could not find foot geom '{geom_name}' in training scene.")

        original_pos = _parse_vector(original_geom.get("pos", "0 0 0"))
        original_size = _parse_vector(original_geom.get("size", "0 0 0"))
        friction_text = original_geom.get("friction", "1.0 0.005 0.0001")
        replacement_geoms, active_names = _build_replacement_geoms(
            side,
            proxy_type,
            original_pos=original_pos,
            original_size=original_size,
            friction_text=friction_text,
        )
        if proxy_type != "original_box":
            _deactivate_original_contact_geom(original_geom)
            for geom_attributes in replacement_geoms:
                _append_geom(body, **geom_attributes)
        contact_geom_names[side] = active_names

    output_path = temp_dir / f"sedon_training_scene_{proxy_type}.xml"
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    return ProxyCaseDefinition(
        case_name=proxy_type,
        proxy_type=proxy_type,
        scene_path=output_path,
        contact_geom_names=contact_geom_names,
    )


def _run_case_for_side(
    case_definition: ProxyCaseDefinition,
    *,
    side: str,
    settle_steps: int,
    steps: int,
    seed: int,
    target_support_com_shift: float,
    com_kp: float,
    com_kd: float,
    roll_kp: float,
    max_support_roll: float,
) -> ProxySweepResult:
    """Run one proxy case for one support side."""
    reward_config = load_sedon_config_from_env()
    env = SedonStandingEnv(
        scene_path=case_definition.scene_path,
        reset_noise_scale=0.0,
        reward_config=reward_config,
    )
    try:
        env.reset(seed=seed)
        initial_contact_count = _total_proxy_contact_count(env, case_definition.contact_geom_names)

        nominal_target = env._apply_safe_joint_target_clamps(env._nominal_joint_qpos.copy())
        for _ in range(settle_steps):
            env._do_pd_simulation(nominal_target)

        settled_contact_count = _total_proxy_contact_count(env, case_definition.contact_geom_names)
        brace_target = _brace_target(env, side)
        left_foot_geom_id = env._geom_id(LEFT_FOOT_GEOM)
        right_foot_geom_id = env._geom_id(RIGHT_FOOT_GEOM)
        initial_com_y = float(_overall_com(env)[1])
        initial_base_roll = _quat_to_roll(env.data.xquat[env._base_body_id])
        left_foot_y = float(env.data.geom_xpos[left_foot_geom_id][1])
        right_foot_y = float(env.data.geom_xpos[right_foot_geom_id][1])
        center_y = 0.5 * (left_foot_y + right_foot_y)
        initial_support_shift = (
            initial_com_y - center_y if side == "left" else center_y - initial_com_y
        )

        previous_support_shift = 0.0
        support_force_ratios: list[float] = []
        swing_force_ratios: list[float] = []
        support_shift_history: list[float] = []
        left_force_history: list[float] = []
        right_force_history: list[float] = []
        max_support_force_ratio = 0.0
        max_abs_com_y_delta = 0.0
        max_base_roll_delta = 0.0
        both_steps = 0
        support_only_steps = 0
        none_steps = 0
        terminated = False
        terminated_step = 0

        for step_index in range(1, steps + 1):
            com_y = float(_overall_com(env)[1])
            left_foot_y = float(env.data.geom_xpos[left_foot_geom_id][1])
            right_foot_y = float(env.data.geom_xpos[right_foot_geom_id][1])
            center_y = 0.5 * (left_foot_y + right_foot_y)
            actual_support_shift = com_y - center_y if side == "left" else center_y - com_y
            support_shift_delta = actual_support_shift - initial_support_shift
            support_shift_velocity = support_shift_delta - previous_support_shift
            previous_support_shift = support_shift_delta

            base_roll = _quat_to_roll(env.data.xquat[env._base_body_id])
            control_signal = (
                com_kp * (target_support_com_shift - support_shift_delta)
                - com_kd * support_shift_velocity
                - roll_kp * base_roll
            )
            support_roll = float(np.clip(control_signal, 0.0, max_support_roll))
            target = brace_target.copy()
            _apply_support_roll(target, side, support_roll)
            target = env._apply_safe_joint_target_clamps(target)
            env._do_pd_simulation(target)

            com_y = float(_overall_com(env)[1])
            left_foot_y = float(env.data.geom_xpos[left_foot_geom_id][1])
            right_foot_y = float(env.data.geom_xpos[right_foot_geom_id][1])
            center_y = 0.5 * (left_foot_y + right_foot_y)
            actual_support_shift = com_y - center_y if side == "left" else center_y - com_y
            support_shift_delta = actual_support_shift - initial_support_shift
            support_shift_history.append(support_shift_delta)
            max_abs_com_y_delta = max(max_abs_com_y_delta, abs(com_y - initial_com_y))
            max_base_roll_delta = max(
                max_base_roll_delta,
                abs(_quat_to_roll(env.data.xquat[env._base_body_id]) - initial_base_roll),
            )

            left_count, left_force = _foot_floor_load(env, case_definition.contact_geom_names["left"])
            right_count, right_force = _foot_floor_load(env, case_definition.contact_geom_names["right"])
            total_force = left_force + right_force
            if total_force > 1e-9:
                left_force_ratio = left_force / total_force
                right_force_ratio = right_force / total_force
            else:
                left_force_ratio = 0.0
                right_force_ratio = 0.0
            support_force_ratio = left_force_ratio if side == "left" else right_force_ratio
            swing_force_ratio = right_force_ratio if side == "left" else left_force_ratio
            support_force_ratios.append(support_force_ratio)
            swing_force_ratios.append(swing_force_ratio)
            left_force_history.append(left_force)
            right_force_history.append(right_force)
            max_support_force_ratio = max(max_support_force_ratio, support_force_ratio)

            contact_state, _ = _contact_state(env, case_definition.contact_geom_names)
            if contact_state == "both":
                both_steps += 1
            elif contact_state == f"{side}_only":
                support_only_steps += 1
            elif contact_state == "none":
                none_steps += 1

            base_z = env._base_height()
            upright = env._base_upright()
            observation = env._get_obs()
            terminated = env._is_terminated(base_z, upright, observation)
            if terminated:
                terminated_step = step_index
                break

        if not terminated:
            terminated_step = steps

        executed_steps = max(len(support_force_ratios), 1)
        force_tail = support_force_ratios[-50:] if len(support_force_ratios) >= 50 else support_force_ratios
        swing_tail = swing_force_ratios[-50:] if len(swing_force_ratios) >= 50 else swing_force_ratios
        shift_tail = support_shift_history[-50:] if len(support_shift_history) >= 50 else support_shift_history
        left_force_tail = left_force_history[-50:] if len(left_force_history) >= 50 else left_force_history
        right_force_tail = right_force_history[-50:] if len(right_force_history) >= 50 else right_force_history
        both_contact_ratio = both_steps / executed_steps
        none_contact_ratio = none_steps / executed_steps
        mean_support_force_ratio_last_50 = float(np.mean(force_tail)) if force_tail else 0.0
        mean_swing_force_ratio_last_50 = float(np.mean(swing_tail)) if swing_tail else 0.0
        mean_support_com_shift_last_50 = float(np.mean(shift_tail)) if shift_tail else 0.0
        left_normal_force = float(np.mean(left_force_tail)) if left_force_tail else 0.0
        right_normal_force = float(np.mean(right_force_tail)) if right_force_tail else 0.0
        score = _score_result(
            max_support_force_ratio=max_support_force_ratio,
            mean_support_force_ratio_last_50=mean_support_force_ratio_last_50,
            mean_support_com_shift_last_50=mean_support_com_shift_last_50,
            both_contact_ratio=both_contact_ratio,
            none_contact_ratio=none_contact_ratio,
            terminated=terminated,
        )

        return ProxySweepResult(
            case_name=case_definition.case_name,
            proxy_type=case_definition.proxy_type,
            side=side,
            terminated=terminated,
            terminated_step=terminated_step,
            initial_contact_count=initial_contact_count,
            settled_contact_count=settled_contact_count,
            both_contact_ratio=both_contact_ratio,
            support_only_steps=support_only_steps,
            none_contact_ratio=none_contact_ratio,
            max_support_force_ratio=max_support_force_ratio,
            mean_support_force_ratio_last_50=mean_support_force_ratio_last_50,
            mean_swing_force_ratio_last_50=mean_swing_force_ratio_last_50,
            max_abs_com_y_delta=max_abs_com_y_delta,
            mean_support_com_shift_last_50=mean_support_com_shift_last_50,
            max_base_roll_delta=max_base_roll_delta,
            left_normal_force=left_normal_force,
            right_normal_force=right_normal_force,
            score=score,
        )
    finally:
        env.close()


def _write_csv(path: Path, rows: list[ProxySweepResult]) -> None:
    """Write sweep rows to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        writer.writerows([asdict(row) for row in rows])


def main(argv: list[str] | None = None) -> int:
    """Run the Sedon foot-proxy redesign sweep."""
    args = build_parser().parse_args(argv)
    if args.settle_steps < 0:
        raise ValueError("--settle-steps must be non-negative.")
    if args.steps <= 0:
        raise ValueError("--steps must be positive.")
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive.")

    results: list[ProxySweepResult] = []
    with tempfile.TemporaryDirectory(prefix="sedon_foot_proxy_redesign_") as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        case_definitions = [_make_case_definition(proxy_type, tmp_dir) for proxy_type in PROXY_CASES]
        for case_definition in case_definitions:
            for side in ("left", "right"):
                results.append(
                    _run_case_for_side(
                        case_definition,
                        side=side,
                        settle_steps=args.settle_steps,
                        steps=args.steps,
                        seed=args.seed,
                        target_support_com_shift=args.target_support_com_shift,
                        com_kp=args.com_kp,
                        com_kd=args.com_kd,
                        roll_kp=args.roll_kp,
                        max_support_roll=args.max_support_roll,
                    )
                )

    _write_csv(args.out_csv, results)
    ranked = sorted(
        results,
        key=lambda row: (
            row.score,
            not row.terminated,
            row.mean_support_force_ratio_last_50,
            row.mean_support_com_shift_last_50,
        ),
        reverse=True,
    )

    threshold_cases = [
        row
        for row in results
        if not row.terminated
        and row.mean_support_force_ratio_last_50 > 0.65
        and row.mean_support_com_shift_last_50 > 0.008
    ]

    print(f"cases: {len(results)}")
    print(f"cases_meeting_threshold: {len(threshold_cases)}")
    print(
        "rank case side support_ratio com_shift both_ratio "
        "support_only none_ratio term term_step score"
    )
    for index, row in enumerate(ranked[: args.top_k], start=1):
        print(
            f"{index:>4} {row.case_name:>20} {row.side:>5} "
            f"{row.mean_support_force_ratio_last_50:>13.3f} {row.mean_support_com_shift_last_50:>9.4f} "
            f"{row.both_contact_ratio:>10.3f} {row.support_only_steps:>12} "
            f"{row.none_contact_ratio:>10.3f} {str(row.terminated):>5} "
            f"{row.terminated_step:>9} {row.score:>8.2f}"
        )

    print(f"\ncsv: {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
