"""Interactive Sedon pose editor built around the MuJoCo viewer.

This tool opens Sedon in a MuJoCo viewer and provides a secondary Tkinter
slider panel for joint target adjustment, pose saving/loading, keyframe
sequence export, and basic contact/support diagnostics.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import tkinter as tk
from tkinter import ttk

from sedon_baseline.env import (
    JOINT_NAMES,
    SedonStandingEnv,
    load_sedon_config_from_env,
)
from tools.sedon_debug_common import (
    BASE_PROXY_GEOM,
    DEBUG_OUT_DIR,
    DEFAULT_SCENE_PATH,
    FLOOR_GEOM,
    LEFT_FOOT_GEOM,
    RIGHT_FOOT_GEOM,
    contact_pairs,
    require_scene,
)

DEFAULT_POSE_PATH = DEBUG_OUT_DIR / "sedon_pose_editor_poses.json"
DEFAULT_SEQUENCE_PATH = DEBUG_OUT_DIR / "sedon_reference_gait_seed.json"
SUPPORTED_SUPPORT_MODES = ("double", "left", "right")
DEFAULT_SETTLE_STEPS = 120
DEFAULT_LEAN_ROLL_DELTA = 0.03
DEFAULT_LEAN_TAU = 6.0


@dataclass
class PoseEntry:
    name: str
    support_mode: str
    joint_targets: list[float]
    duration_steps: int = 60
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "PoseEntry":
        return cls(
            name=str(raw["name"]),
            support_mode=str(raw.get("support_mode", "double")),
            joint_targets=[float(x) for x in raw["joint_targets"]],
            duration_steps=int(raw.get("duration_steps", 60)),
            note=str(raw.get("note", "")),
        )


def _load_pose_entries(path: Path) -> list[PoseEntry]:
    if not path.is_file():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Unable to parse pose file: {path}") from exc
    if not isinstance(raw, list):
        raise ValueError(f"Pose file must contain a JSON list: {path}")
    return [PoseEntry.from_dict(item) for item in raw]


def _write_pose_entries(path: Path, entries: list[PoseEntry]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([entry.to_dict() for entry in entries], indent=2), encoding="utf-8")


def _is_zero_pose(entry: PoseEntry, *, tolerance: float = 1e-9) -> bool:
    """Return whether a pose contains effectively zero joint targets."""
    return all(abs(value) <= tolerance for value in entry.joint_targets)


class PoseEditor:
    def __init__(self, scene_path: Path, pose_path: Path, sequence_path: Path) -> None:
        self.scene_path = require_scene(scene_path)
        self.pose_path = pose_path
        self.sequence_path = sequence_path
        self.saved_poses = _load_pose_entries(self.pose_path)
        self.sequence: list[PoseEntry] = []
        self.paused = True
        self.sequence_preview_active = False
        self.sequence_preview_index = 0
        self.sequence_preview_step = 0

        reward_config = load_sedon_config_from_env()
        self.env = SedonStandingEnv(scene_path=self.scene_path, reset_noise_scale=0.0, reward_config=reward_config)
        self.env.reset(seed=0)

        self.model = self.env.model
        self.data = self.env.data
        self.joint_qpos_addresses = [self.model.jnt_qposadr[joint_id] for joint_id in self.env._joint_ids]
        self.joint_limits = self._build_joint_limits()
        self.root = tk.Tk()
        self.root.title("Sedon Pose Editor")
        self.joint_vars: dict[str, tk.DoubleVar] = {}
        self.support_mode_var = tk.StringVar(value="double")
        self.pose_name_var = tk.StringVar(value="pose_1")
        self.duration_steps_var = tk.IntVar(value=60)
        self.settle_steps_var = tk.IntVar(value=DEFAULT_SETTLE_STEPS)
        self.lean_roll_delta_var = tk.DoubleVar(value=DEFAULT_LEAN_ROLL_DELTA)
        self.lean_tau_var = tk.DoubleVar(value=DEFAULT_LEAN_TAU)
        self.note_var = tk.StringVar(value="")
        self.selected_pose_var = tk.StringVar(value="")
        self.sequence_label_var = tk.StringVar(value="sequence: 0 poses")
        self.metrics_text_var = tk.StringVar(value="")
        self._build_ui()
        self._update_pose_list_menu()
        self._reset_to_nominal_pose()

    def _build_joint_limits(self) -> dict[str, tuple[float, float]]:
        limits: dict[str, tuple[float, float]] = {}
        for name, joint_id in zip(JOINT_NAMES, self.env._joint_ids):
            lower, upper = self.model.jnt_range[joint_id]
            if lower == upper == 0.0:
                lower, upper = -1.5, 1.5
            limits[name] = (float(lower), float(upper))
        return limits

    def _build_ui(self) -> None:
        control_frame = ttk.Frame(self.root, padding=8)
        control_frame.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        slider_frame = ttk.LabelFrame(control_frame, text="Joint targets")
        slider_frame.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)

        for index, joint_name in enumerate(JOINT_NAMES):
            limit_min, limit_max = self.joint_limits[joint_name]
            value = float(self.env._nominal_joint_qpos[index])
            var = tk.DoubleVar(value=value)
            self.joint_vars[joint_name] = var
            row = index // 2
            col = (index % 2) * 2
            ttk.Label(slider_frame, text=joint_name).grid(row=row * 2, column=col, sticky="w", padx=2, pady=2)
            slider = ttk.Scale(
                slider_frame,
                variable=var,
                from_=limit_min,
                to=limit_max,
                orient="horizontal",
                command=lambda _value, name=joint_name: self._on_slider_move(name),
                length=260,
            )
            slider.grid(row=row * 2 + 1, column=col, columnspan=2, sticky="ew", padx=2, pady=2)
            value_label = ttk.Label(slider_frame, text=f"{value:.3f}")
            value_label.grid(row=row * 2, column=col + 1, sticky="e", padx=2)
            var.trace_add("write", lambda *args, name=joint_name, label=value_label: label.config(text=f"{self.joint_vars[name].get():.3f}"))

        state_frame = ttk.LabelFrame(control_frame, text="Pose controls")
        state_frame.grid(row=1, column=0, sticky="ew", padx=4, pady=4)
        state_frame.columnconfigure(1, weight=1)

        ttk.Label(state_frame, text="Pose name:").grid(row=0, column=0, sticky="w", padx=2, pady=2)
        ttk.Entry(state_frame, textvariable=self.pose_name_var).grid(row=0, column=1, sticky="ew", padx=2, pady=2)
        ttk.Label(state_frame, text="Support mode:").grid(row=1, column=0, sticky="w", padx=2, pady=2)
        ttk.OptionMenu(state_frame, self.support_mode_var, self.support_mode_var.get(), *SUPPORTED_SUPPORT_MODES).grid(row=1, column=1, sticky="ew", padx=2, pady=2)
        ttk.Label(state_frame, text="Duration steps:").grid(row=2, column=0, sticky="w", padx=2, pady=2)
        ttk.Spinbox(state_frame, from_=1, to=2000, textvariable=self.duration_steps_var, width=8).grid(row=2, column=1, sticky="w", padx=2, pady=2)
        ttk.Label(state_frame, text="Settle steps:").grid(row=3, column=0, sticky="w", padx=2, pady=2)
        ttk.Spinbox(state_frame, from_=1, to=2000, textvariable=self.settle_steps_var, width=8).grid(row=3, column=1, sticky="w", padx=2, pady=2)
        ttk.Label(state_frame, text="Lean roll:").grid(row=4, column=0, sticky="w", padx=2, pady=2)
        ttk.Spinbox(state_frame, from_=0.0, to=0.3, increment=0.005, textvariable=self.lean_roll_delta_var, width=8).grid(row=4, column=1, sticky="w", padx=2, pady=2)
        ttk.Label(state_frame, text="Lean tau:").grid(row=5, column=0, sticky="w", padx=2, pady=2)
        ttk.Spinbox(state_frame, from_=0.0, to=40.0, increment=0.5, textvariable=self.lean_tau_var, width=8).grid(row=5, column=1, sticky="w", padx=2, pady=2)
        ttk.Label(state_frame, text="Note:").grid(row=6, column=0, sticky="w", padx=2, pady=2)
        ttk.Entry(state_frame, textvariable=self.note_var).grid(row=6, column=1, sticky="ew", padx=2, pady=2)

        button_frame = ttk.Frame(control_frame)
        button_frame.grid(row=2, column=0, sticky="ew", padx=4, pady=4)
        button_frame.columnconfigure((0, 1, 2), weight=1)

        ttk.Button(button_frame, text="Pause/Resume", command=self._toggle_pause).grid(row=0, column=0, sticky="ew", padx=2, pady=2)
        ttk.Button(button_frame, text="Reset", command=self._reset_to_nominal_pose).grid(row=0, column=1, sticky="ew", padx=2, pady=2)
        ttk.Button(button_frame, text="Save pose", command=self._save_current_pose).grid(row=0, column=2, sticky="ew", padx=2, pady=2)
        ttk.Button(button_frame, text="Add sequence", command=self._append_to_sequence).grid(row=1, column=0, sticky="ew", padx=2, pady=2)
        ttk.Button(button_frame, text="Export sequence", command=self._export_sequence).grid(row=1, column=1, sticky="ew", padx=2, pady=2)
        ttk.Button(button_frame, text="Load pose", command=self._load_selected_pose).grid(row=1, column=2, sticky="ew", padx=2, pady=2)
        ttk.Button(button_frame, text="Add selected", command=self._append_selected_pose).grid(row=2, column=0, sticky="ew", padx=2, pady=2)
        ttk.Button(button_frame, text="Use saved poses", command=self._sequence_from_saved_poses).grid(row=2, column=1, sticky="ew", padx=2, pady=2)
        ttk.Button(button_frame, text="Clear sequence", command=self._clear_sequence).grid(row=2, column=2, sticky="ew", padx=2, pady=2)
        ttk.Button(button_frame, text="Settle pose", command=self._settle_current_pose).grid(row=3, column=0, sticky="ew", padx=2, pady=2)
        ttk.Button(button_frame, text="Preview sequence", command=self._start_sequence_preview).grid(row=3, column=1, sticky="ew", padx=2, pady=2)
        ttk.Button(button_frame, text="Stop preview", command=self._stop_sequence_preview).grid(row=3, column=2, sticky="ew", padx=2, pady=2)
        ttk.Button(button_frame, text="Lean left", command=lambda: self._lean_support("left")).grid(row=4, column=0, sticky="ew", padx=2, pady=2)
        ttk.Button(button_frame, text="Lean right", command=lambda: self._lean_support("right")).grid(row=4, column=1, sticky="ew", padx=2, pady=2)

        selection_frame = ttk.Frame(control_frame)
        selection_frame.grid(row=3, column=0, sticky="ew", padx=4, pady=4)
        selection_frame.columnconfigure(1, weight=1)
        ttk.Label(selection_frame, text="Saved pose:").grid(row=0, column=0, sticky="w", padx=2, pady=2)
        self.pose_menu = ttk.OptionMenu(selection_frame, self.selected_pose_var, "", *())
        self.pose_menu.grid(row=0, column=1, sticky="ew", padx=2, pady=2)

        ttk.Label(control_frame, textvariable=self.sequence_label_var, anchor="w").grid(row=4, column=0, sticky="ew", padx=4, pady=2)
        self.metrics_label = ttk.Label(control_frame, textvariable=self.metrics_text_var, anchor="w", justify="left")
        self.metrics_label.grid(row=5, column=0, sticky="ew", padx=4, pady=4)

    def _on_slider_move(self, joint_name: str) -> None:
        self._apply_target_pose()

    def _reset_to_nominal_pose(self) -> None:
        nominal = self.env._nominal_joint_qpos
        for name, value in zip(JOINT_NAMES, nominal):
            self.joint_vars[name].set(float(value))
        self.support_mode_var.set("double")
        self.duration_steps_var.set(60)
        self.note_var.set("")
        self._stop_sequence_preview()
        self._apply_target_pose(force=True)

    def _build_target_qpos(self) -> np.ndarray:
        target = self.env._default_qpos.copy()
        self.env._set_base_pose(target)
        for joint_name, addr in zip(JOINT_NAMES, self.joint_qpos_addresses):
            target[addr] = float(self.joint_vars[joint_name].get())
        return target

    def _apply_target_pose(self, force: bool = False) -> None:
        target_qpos = self._build_target_qpos()
        if self.paused or force:
            self.env.set_state(target_qpos, np.zeros_like(self.env.data.qvel))
            mujoco.mj_forward(self.model, self.data)
        else:
            self.env.data.ctrl[:] = self.env._pd_control(self.env._extract_joint_positions(target_qpos))
            mujoco.mj_step(self.model, self.data)
        self._update_metrics()

    def _toggle_pause(self) -> None:
        self.paused = not self.paused
        self._update_metrics()

    def _save_current_pose(self) -> None:
        name = self.pose_name_var.get().strip() or f"pose_{len(self.saved_poses) + 1}"
        pose = PoseEntry(
            name=name,
            support_mode=self.support_mode_var.get(),
            joint_targets=[float(self.joint_vars[name].get()) for name in JOINT_NAMES],
            duration_steps=max(1, int(self.duration_steps_var.get())),
            note=self.note_var.get().strip(),
        )
        self.saved_poses = [entry for entry in self.saved_poses if entry.name != pose.name] + [pose]
        _write_pose_entries(self.pose_path, self.saved_poses)
        self._update_pose_list_menu()
        self.selected_pose_var.set(pose.name)
        self._update_metrics()

    def _load_selected_pose(self) -> None:
        chosen_name = self.selected_pose_var.get()
        entry = next((entry for entry in self.saved_poses if entry.name == chosen_name), None)
        if entry is None:
            return
        for name, value in zip(JOINT_NAMES, entry.joint_targets):
            self.joint_vars[name].set(value)
        self.support_mode_var.set(entry.support_mode)
        self.pose_name_var.set(entry.name)
        self.duration_steps_var.set(max(1, int(entry.duration_steps)))
        self.note_var.set(entry.note)
        self._apply_target_pose(force=True)

    def _append_to_sequence(self) -> None:
        entry = PoseEntry(
            name=self.pose_name_var.get().strip() or f"pose_{len(self.sequence) + 1}",
            support_mode=self.support_mode_var.get(),
            joint_targets=[float(self.joint_vars[name].get()) for name in JOINT_NAMES],
            duration_steps=max(1, int(self.duration_steps_var.get())),
            note=self.note_var.get().strip(),
        )
        self.sequence.append(entry)
        self.sequence_label_var.set(f"sequence: {len(self.sequence)} poses")
        self._update_metrics()

    def _append_selected_pose(self) -> None:
        """Append the currently selected saved pose to the active sequence."""
        chosen_name = self.selected_pose_var.get()
        entry = next((entry for entry in self.saved_poses if entry.name == chosen_name), None)
        if entry is None:
            self.metrics_text_var.set("No saved pose selected.")
            return
        self.sequence.append(entry)
        self.sequence_label_var.set(f"sequence: {len(self.sequence)} poses")
        self.metrics_text_var.set(f"Added saved pose '{entry.name}' to sequence.")

    def _sequence_from_saved_poses(self) -> None:
        """Replace the active sequence with saved poses, skipping empty zero poses."""
        entries = [entry for entry in self.saved_poses if not _is_zero_pose(entry)]
        if not entries:
            self.metrics_text_var.set("No non-zero saved poses available for sequence export.")
            return
        self.sequence = list(entries)
        self.sequence_label_var.set(f"sequence: {len(self.sequence)} poses")
        self.metrics_text_var.set(
            "Sequence replaced with saved poses in file order "
            f"({', '.join(entry.name for entry in self.sequence)})."
        )

    def _clear_sequence(self) -> None:
        """Clear the active sequence builder."""
        self.sequence.clear()
        self.sequence_label_var.set("sequence: 0 poses")
        self.metrics_text_var.set("Sequence cleared.")

    def _settle_current_pose(self) -> None:
        """Simulate the current joint targets from the current floating-base state."""
        self._stop_sequence_preview()
        settle_steps = max(1, int(self.settle_steps_var.get()))
        target_positions = np.array(
            [float(self.joint_vars[name].get()) for name in JOINT_NAMES],
            dtype=np.float64,
        )
        for _ in range(settle_steps):
            self.env._do_pd_simulation(target_positions)
        self.paused = True
        self.metrics_text_var.set(
            f"Settled current pose for {settle_steps} control steps from the current physical state."
        )
        self._update_metrics()

    def _lean_support(self, side: str) -> None:
        """Bias the body toward one support side using paired hip-roll targets and torque assist."""
        if side not in ("left", "right"):
            raise ValueError(f"Unsupported lean side: {side}")
        self._stop_sequence_preview()
        settle_steps = max(1, int(self.settle_steps_var.get()))
        roll_delta = float(self.lean_roll_delta_var.get())
        tau = float(self.lean_tau_var.get())
        target_positions = np.array(
            [float(self.joint_vars[name].get()) for name in JOINT_NAMES],
            dtype=np.float64,
        )
        if side == "left":
            target_positions[1] -= roll_delta
            target_positions[6] += roll_delta
            left_tau_assist = tau
            right_tau_assist = -tau
            self.support_mode_var.set("left")
        else:
            target_positions[1] += roll_delta
            target_positions[6] -= roll_delta
            left_tau_assist = -tau
            right_tau_assist = tau
            self.support_mode_var.set("right")
        for joint_name, value in zip(JOINT_NAMES, target_positions):
            self.joint_vars[joint_name].set(float(value))
        for _ in range(settle_steps):
            self.env._do_pd_simulation_with_torque_assist(
                target_positions,
                left_tau_assist=left_tau_assist,
                right_tau_assist=right_tau_assist,
            )
        left_delta, right_delta = self.env.last_hip_roll_ctrl_assist_delta()
        self.paused = True
        self.metrics_text_var.set(
            f"Leaned {side} for {settle_steps} control steps "
            f"(roll_delta={roll_delta:.3f}, tau={tau:.2f}, applied=({left_delta:.2f},{right_delta:.2f}))."
        )
        self._update_metrics()

    def _start_sequence_preview(self) -> None:
        """Play the current keyframe sequence from the current floating-base state."""
        if not self.sequence:
            self.metrics_text_var.set("No sequence to preview.")
            return
        self.sequence_preview_active = True
        self.sequence_preview_index = 0
        self.sequence_preview_step = 0
        self.paused = False
        self.metrics_text_var.set(
            f"Previewing sequence from the current physical state ({len(self.sequence)} poses)."
        )

    def _stop_sequence_preview(self) -> None:
        """Stop sequence preview and leave the robot in its current physical state."""
        self.sequence_preview_active = False
        self.sequence_preview_index = 0
        self.sequence_preview_step = 0

    def _step_sequence_preview(self) -> None:
        """Advance one preview step without resetting the floating base."""
        if not self.sequence_preview_active or not self.sequence:
            return
        entry = self.sequence[self.sequence_preview_index]
        target_positions = np.asarray(entry.joint_targets, dtype=np.float64)
        self.env._do_pd_simulation(target_positions)
        self.sequence_preview_step += 1
        if self.sequence_preview_step >= max(1, int(entry.duration_steps)):
            self.sequence_preview_step = 0
            self.sequence_preview_index = (self.sequence_preview_index + 1) % len(self.sequence)
        self.metrics_text_var.set(
            f"Previewing sequence: {entry.name} "
            f"({self.sequence_preview_index + 1}/{len(self.sequence)}) "
            f"step {self.sequence_preview_step + 1}/{max(1, int(entry.duration_steps))}"
        )

    def _export_sequence(self) -> None:
        if not self.sequence:
            self.metrics_text_var.set("No keyframe sequence to export.")
            return
        self.sequence_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": "sedon_gait_seed.v1",
            "target_type": "absolute",
            "description": "Reference gait keyframes exported by debug_sedon_pose_editor.",
            "joint_names": list(JOINT_NAMES),
            "keyframes": [entry.to_dict() for entry in self.sequence],
        }
        self.sequence_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self.metrics_text_var.set(f"Gait seed exported to {self.sequence_path}")

    def _update_pose_list_menu(self) -> None:
        menu = self.pose_menu["menu"]
        menu.delete(0, "end")
        for entry in self.saved_poses:
            menu.add_command(label=entry.name, command=lambda name=entry.name: self.selected_pose_var.set(name))
        if self.saved_poses:
            self.selected_pose_var.set(self.saved_poses[-1].name)
        else:
            self.selected_pose_var.set("")

    def _compute_com(self) -> np.ndarray:
        masses = self.model.body_mass
        total_mass = float(np.sum(masses))
        com = np.sum(self.data.xipos * masses[:, None], axis=0) / total_mass
        return com

    def _add_viewer_marker(
        self,
        viewer: object,
        *,
        position: np.ndarray,
        radius: float,
        rgba: tuple[float, float, float, float],
    ) -> None:
        """Add a simple sphere marker for COM and foot support diagnostics."""
        scene = getattr(viewer, "user_scn", None)
        if scene is None or scene.ngeom >= scene.maxgeom:
            return
        mujoco.mjv_initGeom(
            scene.geoms[scene.ngeom],
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([radius, radius, radius], dtype=np.float64),
            np.asarray(position, dtype=np.float64),
            np.eye(3, dtype=np.float64).ravel(),
            np.array(rgba, dtype=np.float32),
        )
        scene.ngeom += 1

    def _update_viewer_markers(self, viewer: object) -> None:
        """Refresh COM and foot-contact markers in the MuJoCo viewer."""
        scene = getattr(viewer, "user_scn", None)
        if scene is None:
            return
        scene.ngeom = 0
        left_contact, right_contact, _, _, _ = self._contact_state()
        self._add_viewer_marker(
            viewer,
            position=self._compute_com(),
            radius=0.025,
            rgba=(1.0, 0.1, 0.1, 0.85),
        )
        right_position = self.data.geom_xpos[self.env._foot_geom_ids[0]].copy()
        left_position = self.data.geom_xpos[self.env._foot_geom_ids[1]].copy()
        right_position[2] += 0.035
        left_position[2] += 0.035
        self._add_viewer_marker(
            viewer,
            position=right_position,
            radius=0.018,
            rgba=(0.1, 0.3, 1.0, 0.85 if right_contact else 0.25),
        )
        self._add_viewer_marker(
            viewer,
            position=left_position,
            radius=0.018,
            rgba=(0.1, 1.0, 0.3, 0.85 if left_contact else 0.25),
        )

    def _contact_state(self) -> tuple[bool, bool, bool, bool, list[str]]:
        contacts = contact_pairs(self.model, self.data)
        left = False
        right = False
        base = False
        foot_foot = False
        pairs: list[str] = []
        for name_a, name_b, _ in contacts:
            pairs.append(f"{name_a}-{name_b}")
            if {name_a, name_b} == {FLOOR_GEOM, LEFT_FOOT_GEOM}:
                left = True
            elif {name_a, name_b} == {FLOOR_GEOM, RIGHT_FOOT_GEOM}:
                right = True
            elif {name_a, name_b} == {FLOOR_GEOM, BASE_PROXY_GEOM}:
                base = True
            elif {name_a, name_b} == {LEFT_FOOT_GEOM, RIGHT_FOOT_GEOM}:
                foot_foot = True
        return left, right, base, foot_foot, pairs

    def _update_metrics(self) -> None:
        com = self._compute_com()
        base_height = float(self.data.xpos[self.env._base_body_id][2])
        upright = self.env._base_upright()
        joint_positions = self.env._joint_positions()
        left_knee = float(joint_positions[8])
        right_knee = float(joint_positions[3])
        left_contact, right_contact, base_contact, foot_foot, pairs = self._contact_state()
        foot_positions = [self.data.geom_xpos[self.env._foot_geom_ids[0]][:2], self.data.geom_xpos[self.env._foot_geom_ids[1]][:2]]
        support_poly = []
        if left_contact:
            support_poly.append(tuple(map(float, foot_positions[1].tolist())))
        if right_contact:
            support_poly.append(tuple(map(float, foot_positions[0].tolist())))
        support_poly_text = ", ".join([f"({x:.3f},{y:.3f})" for x, y in support_poly]) or "none"
        contact_text = f"L={left_contact} R={right_contact} base={base_contact} foot_collision={foot_foot}"
        self.metrics_text_var.set(
            "\n".join(
                [
                    f"COM={com[0]:+.4f},{com[1]:+.4f},{com[2]:+.4f}",
                    f"base_height={base_height:.4f} upright={upright:.4f}",
                    f"knee_pitch: right={right_knee:.4f} left={left_knee:.4f}",
                    f"support_poly={support_poly_text}",
                    f"contacts={contact_text}",
                    f"support_mode={self.support_mode_var.get()}  paused={self.paused}",
                    f"saved={len(self.saved_poses)} pose(s)  sequence={len(self.sequence)}",
                    f"pair_debug={';'.join(pairs)}",
                ]
            )
        )

    def run(self) -> int:
        try:
            import mujoco.viewer
        except Exception as exc:
            raise RuntimeError(
                "MuJoCo viewer is unavailable in this Python environment. Install MuJoCo viewer extras."
            ) from exc

        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
            self._update_metrics()
            while viewer.is_running():
                self.root.update_idletasks()
                self.root.update()
                if self.sequence_preview_active:
                    self._step_sequence_preview()
                    self._update_metrics()
                else:
                    self._apply_target_pose()
                self._update_viewer_markers(viewer)
                viewer.sync()
                time.sleep(self.model.opt.timestep)
        return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE_PATH)
    parser.add_argument("--pose-file", type=Path, default=DEFAULT_POSE_PATH)
    parser.add_argument("--sequence-file", type=Path, default=DEFAULT_SEQUENCE_PATH)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    editor = PoseEditor(scene_path=args.scene, pose_path=args.pose_file, sequence_path=args.sequence_file)
    return editor.run()


if __name__ == "__main__":
    raise SystemExit(main())
