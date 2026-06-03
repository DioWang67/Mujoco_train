"""Interactive Seedon pose editor built around the MuJoCo viewer.

This tool opens Seedon in a MuJoCo viewer and provides a secondary Tkinter
slider panel for joint target adjustment, pose saving/loading, keyframe
sequence export, and basic contact/support diagnostics.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import tkinter as tk
from tkinter import ttk

from seedon_baseline.env import (
    JOINT_NAMES,
    SeedonStandingEnv,
    load_seedon_config_from_env,
)
from tools.seedon_debug_common import (
    BASE_PROXY_GEOM,
    DEBUG_OUT_DIR,
    DEFAULT_SCENE_PATH,
    FLOOR_GEOM,
    LEFT_FOOT_GEOM,
    RIGHT_FOOT_GEOM,
    contact_pairs,
    require_scene,
)

DEFAULT_POSE_PATH = DEBUG_OUT_DIR / "seedon_pose_editor_poses.json"
DEFAULT_SEQUENCE_PATH = DEBUG_OUT_DIR / "seedon_reference_gait_seed.json"
SUPPORTED_SUPPORT_MODES = ("double", "left", "right")
DEFAULT_SETTLE_STEPS = 120
DEFAULT_LEAN_ROLL_DELTA = 0.03
DEFAULT_LEAN_TAU = 6.0
FOOT_CONTACT_FORCE_THRESHOLD_N = 5.0
FORCE_RATIO_EPSILON = 1e-6
DEFAULT_JOINT_TARGET_STEP = 0.005
FORCE_MOVING_AVERAGE_FRAMES = 30
POST_SETTLE_DEBUG_STEPS = 45
GRAVITY_HOLD_TEST_STEPS = 120
SHOW_CONTACT_GEOM_PAIRS = True


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


@dataclass(frozen=True)
class FootContactDiagnostics:
    """Contact diagnostics for one foot against the floor."""

    contact_count: int
    total_normal_force: float
    max_normal_force: float
    max_penetration: float
    foot_geom_z: float
    foot_vertical_velocity: float
    world_z_force: float
    normalized_force: float


@dataclass(frozen=True)
class ContactForceDetail:
    """Contact force details for one active MuJoCo contact."""

    index: int
    geom1: str
    geom2: str
    dist: float
    normal_force: float
    frame_z: float
    efc_address: int
    world_z_force: float


@dataclass(frozen=True)
class AllContactDiagnostics:
    """Aggregate force diagnostics across every active contact."""

    total_contact_force_all: float
    world_z_force_sum_all_contacts: float
    details: list[ContactForceDetail]


@dataclass(frozen=True)
class GravityHoldTestResult:
    """Summary metrics from the neutral gravity-only hold test."""

    avg_contact_force: float
    avg_world_z: float
    avg_ncon: float


@dataclass(frozen=True)
class FootForceState:
    """Aggregated ground reaction force diagnostics for the two Seedon feet."""

    left_force: float
    right_force: float
    force_ratio_left: float
    force_ratio_right: float
    support_side_guess: str


def _support_side_guess(
    left_force: float,
    right_force: float,
    *,
    threshold: float = FOOT_CONTACT_FORCE_THRESHOLD_N,
) -> str:
    """Infer support side from vertical foot contact force.

    Args:
        left_force: Left foot contact force in Newtons.
        right_force: Right foot contact force in Newtons.
        threshold: Minimum z force treated as planted contact.

    Returns:
        One of ``left``, ``right``, ``double``, or ``none``.
    """
    left_contact = left_force > threshold
    right_contact = right_force > threshold
    if left_contact and right_contact:
        return "double"
    if left_contact:
        return "left"
    if right_contact:
        return "right"
    return "none"


def _foot_force_state(left_force: float, right_force: float) -> FootForceState:
    """Build ratio and support-side diagnostics from left/right GRF values.

    Args:
        left_force: Left foot ground reaction force in Newtons.
        right_force: Right foot ground reaction force in Newtons.

    Returns:
        Normalized force ratio and support-side estimate.
    """
    total_force = left_force + right_force + FORCE_RATIO_EPSILON
    return FootForceState(
        left_force=float(left_force),
        right_force=float(right_force),
        force_ratio_left=float(left_force / total_force),
        force_ratio_right=float(right_force / total_force),
        support_side_guess=_support_side_guess(left_force, right_force),
    )


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
        self.dynamic_debug_valid = False
        self.left_force_history: deque[float] = deque(maxlen=FORCE_MOVING_AVERAGE_FRAMES)
        self.right_force_history: deque[float] = deque(maxlen=FORCE_MOVING_AVERAGE_FRAMES)

        reward_config = load_seedon_config_from_env()
        self.env = SeedonStandingEnv(scene_path=self.scene_path, reset_noise_scale=0.0, reward_config=reward_config)
        self.env.reset(seed=0)

        self.model = self.env.model
        self.data = self.env.data
        self.joint_qpos_addresses = [self.model.jnt_qposadr[joint_id] for joint_id in self.env._joint_ids]
        self.joint_limits = self._build_joint_limits()
        self.root = tk.Tk()
        self.root.title("Seedon Pose Editor")
        self.joint_vars: dict[str, tk.DoubleVar] = {}
        self._suppress_joint_target_apply = False
        self.support_mode_var = tk.StringVar(value="double")
        self.pose_name_var = tk.StringVar(value="pose_1")
        self.duration_steps_var = tk.IntVar(value=60)
        self.settle_steps_var = tk.IntVar(value=DEFAULT_SETTLE_STEPS)
        self.joint_step_var = tk.DoubleVar(value=DEFAULT_JOINT_TARGET_STEP)
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
            col = (index % 2) * 4
            ttk.Label(slider_frame, text=joint_name).grid(row=row * 2, column=col, sticky="w", padx=2, pady=2)
            ttk.Button(
                slider_frame,
                text="-",
                width=3,
                command=lambda name=joint_name: self._nudge_joint_target(name, direction=-1),
            ).grid(row=row * 2, column=col + 1, sticky="e", padx=1, pady=2)
            ttk.Spinbox(
                slider_frame,
                from_=limit_min,
                to=limit_max,
                increment=DEFAULT_JOINT_TARGET_STEP,
                textvariable=var,
                width=9,
                format="%.4f",
            ).grid(row=row * 2, column=col + 2, sticky="ew", padx=1, pady=2)
            ttk.Button(
                slider_frame,
                text="+",
                width=3,
                command=lambda name=joint_name: self._nudge_joint_target(name, direction=1),
            ).grid(row=row * 2, column=col + 3, sticky="w", padx=1, pady=2)
            slider = ttk.Scale(
                slider_frame,
                variable=var,
                from_=limit_min,
                to=limit_max,
                orient="horizontal",
                length=220,
            )
            slider.grid(row=row * 2 + 1, column=col, columnspan=4, sticky="ew", padx=2, pady=2)
            var.trace_add("write", lambda *args, name=joint_name: self._on_joint_target_change(name))

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
        ttk.Label(state_frame, text="Joint step:").grid(row=4, column=0, sticky="w", padx=2, pady=2)
        ttk.Spinbox(state_frame, from_=0.001, to=0.2, increment=0.001, textvariable=self.joint_step_var, width=8, format="%.3f").grid(row=4, column=1, sticky="w", padx=2, pady=2)
        ttk.Label(state_frame, text="Lean roll:").grid(row=5, column=0, sticky="w", padx=2, pady=2)
        ttk.Spinbox(state_frame, from_=0.0, to=0.3, increment=0.005, textvariable=self.lean_roll_delta_var, width=8).grid(row=5, column=1, sticky="w", padx=2, pady=2)
        ttk.Label(state_frame, text="Lean tau:").grid(row=6, column=0, sticky="w", padx=2, pady=2)
        ttk.Spinbox(state_frame, from_=0.0, to=40.0, increment=0.5, textvariable=self.lean_tau_var, width=8).grid(row=6, column=1, sticky="w", padx=2, pady=2)
        ttk.Label(state_frame, text="Note:").grid(row=7, column=0, sticky="w", padx=2, pady=2)
        ttk.Entry(state_frame, textvariable=self.note_var).grid(row=7, column=1, sticky="ew", padx=2, pady=2)

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
        ttk.Button(button_frame, text="Dynamic settle", command=self._settle_current_pose).grid(row=3, column=0, sticky="ew", padx=2, pady=2)
        ttk.Button(button_frame, text="Dynamic preview", command=self._start_sequence_preview).grid(row=3, column=1, sticky="ew", padx=2, pady=2)
        ttk.Button(button_frame, text="Stop preview", command=self._stop_sequence_preview).grid(row=3, column=2, sticky="ew", padx=2, pady=2)
        ttk.Button(button_frame, text="Lean left", command=lambda: self._lean_support("left")).grid(row=4, column=0, sticky="ew", padx=2, pady=2)
        ttk.Button(button_frame, text="Lean right", command=lambda: self._lean_support("right")).grid(row=4, column=1, sticky="ew", padx=2, pady=2)
        ttk.Button(button_frame, text="Gravity hold test", command=self._run_gravity_hold_test).grid(row=5, column=0, columnspan=3, sticky="ew", padx=2, pady=2)

        selection_frame = ttk.Frame(control_frame)
        selection_frame.grid(row=3, column=0, sticky="ew", padx=4, pady=4)
        selection_frame.columnconfigure(1, weight=1)
        ttk.Label(selection_frame, text="Saved pose:").grid(row=0, column=0, sticky="w", padx=2, pady=2)
        self.pose_menu = ttk.OptionMenu(selection_frame, self.selected_pose_var, "", *())
        self.pose_menu.grid(row=0, column=1, sticky="ew", padx=2, pady=2)

        ttk.Label(control_frame, textvariable=self.sequence_label_var, anchor="w").grid(row=4, column=0, sticky="ew", padx=4, pady=2)
        metrics_frame = ttk.Frame(control_frame)
        metrics_frame.grid(row=5, column=0, sticky="nsew", padx=4, pady=4)
        metrics_frame.columnconfigure(0, weight=1)
        metrics_frame.rowconfigure(0, weight=1)
        control_frame.rowconfigure(5, weight=1)
        self.metrics_text = tk.Text(metrics_frame, height=22, width=120, wrap="none", state="disabled")
        self.metrics_text.grid(row=0, column=0, sticky="nsew")
        self.metrics_text.bind("<MouseWheel>", self._on_metrics_mousewheel)
        metrics_y_scroll = ttk.Scrollbar(metrics_frame, orient="vertical", command=self.metrics_text.yview)
        metrics_y_scroll.grid(row=0, column=1, sticky="ns")
        metrics_x_scroll = ttk.Scrollbar(metrics_frame, orient="horizontal", command=self.metrics_text.xview)
        metrics_x_scroll.grid(row=1, column=0, sticky="ew")
        self.metrics_text.configure(yscrollcommand=metrics_y_scroll.set, xscrollcommand=metrics_x_scroll.set)
        self.metrics_text_var.trace_add("write", lambda *_args: self._sync_metrics_text())
        ttk.Button(metrics_frame, text="Copy log", command=self._copy_metrics_log).grid(row=1, column=1, sticky="ew")

    def _on_joint_target_change(self, joint_name: str) -> None:
        if self._suppress_joint_target_apply:
            return
        self._apply_target_pose(force=True)

    def _sync_metrics_text(self) -> None:
        """Refresh the scrollable debug text widget from the metrics string."""
        if not hasattr(self, "metrics_text"):
            return
        yview = self.metrics_text.yview()
        xview = self.metrics_text.xview()
        self.metrics_text.configure(state="normal")
        self.metrics_text.delete("1.0", "end")
        self.metrics_text.insert("1.0", self.metrics_text_var.get())
        self.metrics_text.configure(state="disabled")
        self.metrics_text.yview_moveto(yview[0])
        self.metrics_text.xview_moveto(xview[0])

    def _on_metrics_mousewheel(self, event: tk.Event) -> str:
        """Scroll the debug text even while the widget is read-only."""
        delta = -1 if event.delta > 0 else 1
        self.metrics_text.yview_scroll(delta * 3, "units")
        return "break"

    def _copy_metrics_log(self) -> None:
        """Copy the current debug log text to the system clipboard."""
        log_text = self.metrics_text_var.get()
        self.root.clipboard_clear()
        self.root.clipboard_append(log_text)
        self.root.update()

    def _set_joint_targets(self, values: np.ndarray | list[float]) -> None:
        """Set all joint target widgets without applying each intermediate value."""
        self._suppress_joint_target_apply = True
        try:
            for name, value in zip(JOINT_NAMES, values):
                self.joint_vars[name].set(float(value))
        finally:
            self._suppress_joint_target_apply = False

    def _nudge_joint_target(self, joint_name: str, *, direction: int) -> None:
        """Move one joint target by the configured step while respecting limits."""
        step = max(1e-6, abs(float(self.joint_step_var.get())))
        limit_min, limit_max = self.joint_limits[joint_name]
        current_value = float(self.joint_vars[joint_name].get())
        next_value = float(np.clip(current_value + direction * step, limit_min, limit_max))
        self.joint_vars[joint_name].set(next_value)

    def _reset_to_nominal_pose(self) -> None:
        nominal = self.env._nominal_joint_qpos
        self._set_joint_targets(nominal)
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
        if self.paused and self.dynamic_debug_valid and not force:
            self._update_metrics()
            return
        if self.paused or force:
            self.env.set_state(target_qpos, np.zeros_like(self.env.data.qvel))
            mujoco.mj_forward(self.model, self.data)
            self.dynamic_debug_valid = False
            self.left_force_history.clear()
            self.right_force_history.clear()
        else:
            self.env.data.ctrl[:] = self.env._pd_control(self.env._extract_joint_positions(target_qpos))
            mujoco.mj_step(self.model, self.data)
            self.dynamic_debug_valid = True
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
        self._set_joint_targets(entry.joint_targets)
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
            self.dynamic_debug_valid = True
            self._update_metrics()
        for _ in range(POST_SETTLE_DEBUG_STEPS):
            mujoco.mj_step(self.model, self.data)
            self.dynamic_debug_valid = True
            self._update_metrics()
        self.paused = True
        self.metrics_text_var.set(
            f"Settled current pose for {settle_steps} control steps, then stepped "
            f"{POST_SETTLE_DEBUG_STEPS} debug frames from the current physical state."
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
        self._set_joint_targets(target_positions)
        for _ in range(settle_steps):
            self.env._do_pd_simulation_with_torque_assist(
                target_positions,
                left_tau_assist=left_tau_assist,
                right_tau_assist=right_tau_assist,
            )
            self.dynamic_debug_valid = True
            self._update_metrics()
        left_delta, right_delta = self.env.last_hip_roll_ctrl_assist_delta()
        self.paused = True
        self.metrics_text_var.set(
            f"Leaned {side} for {settle_steps} control steps "
            f"(roll_delta={roll_delta:.3f}, tau={tau:.2f}, applied=({left_delta:.2f},{right_delta:.2f}))."
        )
        self._update_metrics()

    def _run_gravity_hold_test(self) -> None:
        """Run a neutral gravity-only hold test and report averaged contact metrics."""
        self._stop_sequence_preview()
        self.paused = True
        self.support_mode_var.set("double")
        self._set_joint_targets(self.env._nominal_joint_qpos)
        target_qpos = self._build_target_qpos()
        self.env.set_state(target_qpos, np.zeros_like(self.env.data.qvel))
        self.data.ctrl[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.dynamic_debug_valid = False

        total_contact_forces: list[float] = []
        world_z_forces: list[float] = []
        ncon_values: list[int] = []
        for _ in range(GRAVITY_HOLD_TEST_STEPS):
            self.data.ctrl[:] = 0.0
            mujoco.mj_step(self.model, self.data)
            self.dynamic_debug_valid = True
            diagnostics = self._all_contact_diagnostics()
            total_contact_forces.append(diagnostics.total_contact_force_all)
            world_z_forces.append(diagnostics.world_z_force_sum_all_contacts)
            ncon_values.append(int(self.data.ncon))

        result = GravityHoldTestResult(
            avg_contact_force=float(np.mean(total_contact_forces)) if total_contact_forces else 0.0,
            avg_world_z=float(np.mean(world_z_forces)) if world_z_forces else 0.0,
            avg_ncon=float(np.mean(ncon_values)) if ncon_values else 0.0,
        )
        self._update_metrics(
            extra_lines=[
                "gravity_hold_test:",
                f"  steps={GRAVITY_HOLD_TEST_STEPS}",
                f"  avg_contact_force={result.avg_contact_force:.2f}N",
                f"  avg_world_z={result.avg_world_z:.2f}N",
                f"  avg_ncon={result.avg_ncon:.2f}",
            ]
        )

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
        self.dynamic_debug_valid = True
        self._update_metrics()
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
            "schema": "seedon_gait_seed.v1",
            "target_type": "absolute",
            "description": "Reference gait keyframes exported by debug_seedon_pose_editor.",
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

    def _robot_weight(self) -> float:
        """Return total robot weight in Newtons."""
        return float(np.sum(self.model.body_mass) * 9.81)

    def _contact_force_detail(self, contact_index: int) -> ContactForceDetail:
        """Return force diagnostics for one MuJoCo contact."""
        contact = self.data.contact[contact_index]
        geom1_name = mujoco.mj_id2name(
            self.model,
            mujoco.mjtObj.mjOBJ_GEOM,
            int(contact.geom1),
        ) or f"geom_{int(contact.geom1)}"
        geom2_name = mujoco.mj_id2name(
            self.model,
            mujoco.mjtObj.mjOBJ_GEOM,
            int(contact.geom2),
        ) or f"geom_{int(contact.geom2)}"
        wrench = np.zeros(6, dtype=np.float64)
        mujoco.mj_contactForce(self.model, self.data, contact_index, wrench)
        contact_frame = np.asarray(contact.frame, dtype=np.float64).reshape(3, 3)
        world_force = contact_frame.T @ wrench[:3]
        efc_address = getattr(contact, "efc_address", -1)
        return ContactForceDetail(
            index=int(contact_index),
            geom1=geom1_name,
            geom2=geom2_name,
            dist=float(contact.dist),
            normal_force=abs(float(wrench[0])),
            frame_z=float(contact_frame[0, 2]),
            efc_address=int(efc_address),
            world_z_force=abs(float(world_force[2])),
        )

    def _all_contact_diagnostics(self) -> AllContactDiagnostics:
        """Return aggregate diagnostics across all current MuJoCo contacts."""
        details = [
            self._contact_force_detail(contact_index)
            for contact_index in range(self.data.ncon)
        ]
        return AllContactDiagnostics(
            total_contact_force_all=float(sum(detail.normal_force for detail in details)),
            world_z_force_sum_all_contacts=float(sum(detail.world_z_force for detail in details)),
            details=details,
        )

    def _geom_vertical_velocity(self, geom_id: int) -> float:
        """Return world-frame vertical velocity for a geom."""
        velocity = np.zeros(6, dtype=np.float64)
        try:
            mujoco.mj_objectVelocity(
                self.model,
                self.data,
                mujoco.mjtObj.mjOBJ_GEOM,
                int(geom_id),
                velocity,
                0,
            )
            return float(velocity[5])
        except Exception:
            body_id = int(self.model.geom_bodyid[geom_id])
            return float(self.data.cvel[body_id][5])

    def _contact_diagnostics_for_foot(
        self,
        foot_geom_name: str,
        *,
        total_robot_weight: float,
        all_contacts: AllContactDiagnostics,
    ) -> FootContactDiagnostics:
        """Return diagnostics from all floor contacts for one foot.

        The method intentionally receives diagnostics built by scanning every
        active MuJoCo contact and filters to the requested foot-floor pair.
        ``normal_force`` is the contact-frame normal force, while
        ``world_z_force`` is the contact wrench converted to world Z magnitude.
        """
        foot_geom_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_GEOM,
            foot_geom_name,
        )
        if foot_geom_id < 0:
            raise ValueError(f"Foot geom not found: {foot_geom_name}")

        contact_count = 0
        total_normal_force = 0.0
        max_normal_force = 0.0
        max_penetration = 0.0
        world_z_force = 0.0

        for detail in all_contacts.details:
            pair = {detail.geom1, detail.geom2}
            if pair != {FLOOR_GEOM, foot_geom_name}:
                continue

            contact_count += 1
            total_normal_force += detail.normal_force
            max_normal_force = max(max_normal_force, detail.normal_force)
            max_penetration = max(max_penetration, max(0.0, -detail.dist))
            world_z_force += detail.world_z_force

        return FootContactDiagnostics(
            contact_count=contact_count,
            total_normal_force=float(total_normal_force),
            max_normal_force=float(max_normal_force),
            max_penetration=float(max_penetration),
            foot_geom_z=float(self.data.geom_xpos[foot_geom_id][2]),
            foot_vertical_velocity=self._geom_vertical_velocity(foot_geom_id),
            world_z_force=float(world_z_force),
            normalized_force=float(total_normal_force / max(total_robot_weight, FORCE_RATIO_EPSILON)),
        )

    def _compute_contact_diagnostics(
        self,
    ) -> tuple[FootContactDiagnostics, FootContactDiagnostics, FootForceState, AllContactDiagnostics]:
        """Return foot contact diagnostics, force ratios, and all-contact aggregates."""
        total_robot_weight = self._robot_weight()
        all_contacts = self._all_contact_diagnostics()
        left = self._contact_diagnostics_for_foot(
            LEFT_FOOT_GEOM,
            total_robot_weight=total_robot_weight,
            all_contacts=all_contacts,
        )
        right = self._contact_diagnostics_for_foot(
            RIGHT_FOOT_GEOM,
            total_robot_weight=total_robot_weight,
            all_contacts=all_contacts,
        )
        force_state = _foot_force_state(
            left_force=left.total_normal_force,
            right_force=right.total_normal_force,
        )
        return left, right, force_state, all_contacts

    def _format_foot_contact_diagnostics(self, label: str, diagnostics: FootContactDiagnostics) -> str:
        """Return compact one-line contact diagnostics for the debug panel."""
        return (
            f"{label}: contact_count={diagnostics.contact_count} "
            f"total_normal_force={diagnostics.total_normal_force:.2f}N "
            f"max_normal_force={diagnostics.max_normal_force:.2f}N "
            f"max_penetration={diagnostics.max_penetration * 1000.0:.2f}mm "
            f"foot_geom_z={diagnostics.foot_geom_z:.4f} "
            f"foot_vertical_velocity={diagnostics.foot_vertical_velocity:+.4f} "
            f"normalized_force={diagnostics.normalized_force:.3f} "
            f"world_z_force={diagnostics.world_z_force:.2f}N"
        )

    def _moving_average(self, values: deque[float]) -> float:
        """Return a finite moving average for a short debug history."""
        if not values:
            return 0.0
        return float(sum(values) / len(values))

    def _format_contact_force_details(self, diagnostics: AllContactDiagnostics) -> list[str]:
        """Return per-contact force detail lines for debug mode."""
        if not SHOW_CONTACT_GEOM_PAIRS:
            return []
        if not diagnostics.details:
            return ["DEBUG MODE: show_contact_geom_pairs=True", "contact_pairs: none"]
        lines = ["DEBUG MODE: show_contact_geom_pairs=True"]
        for detail in diagnostics.details:
            lines.append(
                f"contact[{detail.index}]: geom1={detail.geom1} geom2={detail.geom2} "
                f"dist={detail.dist:+.6f} normal_force={detail.normal_force:.2f}N "
                f"frame_z={detail.frame_z:+.4f} efc_address={detail.efc_address} "
                f"world_z_force={detail.world_z_force:.2f}N"
            )
        return lines

    def _format_pd_diagnostics(self) -> list[str]:
        """Return per-joint target, state, error, and estimated PD torque lines."""
        target_positions = np.array(
            [float(self.joint_vars[name].get()) for name in JOINT_NAMES],
            dtype=np.float64,
        )
        joint_qpos = self.env._joint_positions()
        joint_qvel = self.env._joint_velocities()
        joint_error = target_positions - joint_qpos
        raw_pd_torque = (
            self.env._reward_config.pd_stiffness * joint_error
            - self.env._reward_config.pd_damping * joint_qvel
        )
        estimated_pd_torque = np.clip(
            raw_pd_torque,
            -self.env._reward_config.torque_scale,
            self.env._reward_config.torque_scale,
        )
        estimated_pd_torque = np.clip(
            estimated_pd_torque,
            self.env._ctrl_range[:, 0],
            self.env._ctrl_range[:, 1],
        )
        lines = [
            "PD diagnostics:",
            f"  kp={self.env._reward_config.pd_stiffness:.3f} kd={self.env._reward_config.pd_damping:.3f}",
        ]
        for joint_name, target, qpos, error, torque in zip(
            JOINT_NAMES,
            target_positions,
            joint_qpos,
            joint_error,
            estimated_pd_torque,
        ):
            lines.append(
                f"  {joint_name}: joint_target={target:+.4f} joint_qpos={qpos:+.4f} "
                f"joint_error={error:+.4f} estimated_pd_torque={torque:+.3f}"
            )
        return lines

    def _update_metrics(self, extra_lines: list[str] | None = None) -> None:
        com = self._compute_com()
        base_height = float(self.data.xpos[self.env._base_body_id][2])
        upright = self.env._base_upright()
        joint_positions = self.env._joint_positions()
        left_knee = float(joint_positions[8])
        right_knee = float(joint_positions[3])
        total_robot_weight = self._robot_weight()
        sum_body_mass = float(np.sum(self.model.body_mass))
        lines = [
            f"debug_mode={'DYNAMIC' if self.dynamic_debug_valid else 'KINEMATIC ONLY'} paused={self.paused}",
            f"data.time={float(self.data.time):.4f} data.ncon={int(self.data.ncon)}",
            f"model.opt.gravity=({self.model.opt.gravity[0]:+.4f},{self.model.opt.gravity[1]:+.4f},{self.model.opt.gravity[2]:+.4f}) "
            f"model.opt.timestep={float(self.model.opt.timestep):.6f}",
            f"sum_body_mass={sum_body_mass:.4f}kg",
            f"COM={com[0]:+.4f},{com[1]:+.4f},{com[2]:+.4f}",
            f"base_height={base_height:.4f} upright={upright:.4f} total_robot_weight={total_robot_weight:.2f}N",
            f"knee_pitch: right={right_knee:.4f} left={left_knee:.4f}",
            f"support_mode={self.support_mode_var.get()}  saved={len(self.saved_poses)} pose(s)  sequence={len(self.sequence)}",
        ]

        contact_detail_lines: list[str] = []
        if self.dynamic_debug_valid:
            left_contact, right_contact, base_contact, foot_foot, pairs = self._contact_state()
            foot_positions = [
                self.data.geom_xpos[self.env._foot_geom_ids[0]][:2],
                self.data.geom_xpos[self.env._foot_geom_ids[1]][:2],
            ]
            support_poly = []
            if left_contact:
                support_poly.append(tuple(map(float, foot_positions[1].tolist())))
            if right_contact:
                support_poly.append(tuple(map(float, foot_positions[0].tolist())))
            support_poly_text = ", ".join([f"({x:.3f},{y:.3f})" for x, y in support_poly]) or "none"
            contact_text = f"L={left_contact} R={right_contact} base={base_contact} foot_collision={foot_foot}"
            left_diagnostics, right_diagnostics, force_state, all_contact_diagnostics = self._compute_contact_diagnostics()
            self.left_force_history.append(left_diagnostics.total_normal_force)
            self.right_force_history.append(right_diagnostics.total_normal_force)
            left_force_ma = self._moving_average(self.left_force_history)
            right_force_ma = self._moving_average(self.right_force_history)
            support_force_ratio_physical = float(
                left_diagnostics.world_z_force
                / (left_diagnostics.world_z_force + right_diagnostics.world_z_force + FORCE_RATIO_EPSILON)
            )
            lines.extend(
                [
                    f"total_contact_force_all={all_contact_diagnostics.total_contact_force_all:.2f}N "
                    f"sum(world_z_force_all_contacts)={all_contact_diagnostics.world_z_force_sum_all_contacts:.2f}N",
                    self._format_foot_contact_diagnostics("left_foot", left_diagnostics),
                    self._format_foot_contact_diagnostics("right_foot", right_diagnostics),
                    f"left_force={force_state.left_force:.2f}N right_force={force_state.right_force:.2f}N",
                    f"left_force_ma30={left_force_ma:.2f}N right_force_ma30={right_force_ma:.2f}N",
                    f"force_ratio_left={force_state.force_ratio_left:.3f} force_ratio_right={force_state.force_ratio_right:.3f}",
                    f"support_force_ratio_physical={support_force_ratio_physical:.3f}",
                    f"support_side_guess={force_state.support_side_guess}",
                    f"support_poly={support_poly_text}",
                    f"contacts={contact_text}",
                    f"pair_debug={';'.join(pairs)}",
                ]
            )
            contact_detail_lines = self._format_contact_force_details(all_contact_diagnostics)
        else:
            lines.extend(
                [
                    "KINEMATIC ONLY: contact/force/support_ratio/normalized_force invalid",
                    "Use Dynamic settle or Dynamic preview before reading support/contact diagnostics.",
                ]
            )

        self.metrics_text_var.set(
            "\n".join(
                lines
                + (extra_lines or [])
                + self._format_pd_diagnostics()
                + contact_detail_lines
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
