"""PPO training entrypoint for the Seedon standing baseline."""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import deque
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from seedon_baseline.env import (
    DEFAULT_SCENE_PATH,
    SeedonStandingConfig,
    SeedonStandingEnv,
    load_seedon_config_from_env,
)
from seedon_baseline.checkpoint_selection import (
    ForwardCheckpointThresholds,
    is_safe_forward_candidate,
    is_stable_forward_candidate,
)
from robot_learning.training_config import load_seedon_train_config
from robot_learning.training_paths import resolve_training_paths
from robot_learning.training_runtime import (
    compute_ppo_batch_size,
    ensure_dirs,
    write_json,
    write_run_manifest,
)

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
SEEDON_CONFIG = load_seedon_train_config(REPO_ROOT)
PATHS = resolve_training_paths(
    REPO_ROOT,
    "seedon",
    legacy_model_dir=os.path.join("models", "seedon"),
    legacy_log_dir=os.path.join("logs", "seedon"),
    legacy_tb_dir=os.path.join("logs", "tb", "seedon"),
)

MODEL_ROOT = str(PATHS.models_root)
LOG_ROOT = str(PATHS.logs_root)
TB_ROOT = str(PATHS.tb_root)
VECNORM_PATH = os.path.join(MODEL_ROOT, "vecnorm.pkl")
VECNORM_PREFIX = "seedon_vecnorm"
BEST_MODEL_DIR = os.path.join(MODEL_ROOT, "best")
BEST_VECNORM_PATH = os.path.join(BEST_MODEL_DIR, "vecnorm.pkl")
FORWARD_BEST_MODEL_DIR = os.path.join(MODEL_ROOT, "forward_best")
FORWARD_BEST_VECNORM_PATH = os.path.join(FORWARD_BEST_MODEL_DIR, "vecnorm.pkl")
STABLE_FORWARD_BEST_MODEL_DIR = os.path.join(MODEL_ROOT, "stable_forward_best")
STABLE_FORWARD_BEST_VECNORM_PATH = os.path.join(
    STABLE_FORWARD_BEST_MODEL_DIR,
    "vecnorm.pkl",
)
LATEST_MODEL_PATH = os.path.join(MODEL_ROOT, "latest_model")
CONFIG_PATH = os.path.join(LOG_ROOT, "train_config.json")
MANIFEST_PATH = os.path.join(LOG_ROOT, "run_manifest.json")
EFFECTIVE_REWARD_CONFIG_PATH = os.path.join(LOG_ROOT, "effective_reward_config.json")
TEACHER_AUDIT_DIR = os.path.join(MODEL_ROOT, "teacher_audit")

N_ENVS_DEFAULT = int(os.environ.get("SEEDON_N_ENVS", str(SEEDON_CONFIG.n_envs_default)))
TOTAL_TIMESTEPS = SEEDON_CONFIG.total_timesteps
SMOKE_TIMESTEPS = SEEDON_CONFIG.smoke_timesteps
N_STEPS = SEEDON_CONFIG.n_steps
N_EPOCHS = SEEDON_CONFIG.n_epochs
GAMMA = SEEDON_CONFIG.gamma
GAE_LAMBDA = SEEDON_CONFIG.gae_lambda
LEARNING_RATE = SEEDON_CONFIG.learning_rate
CLIP_RANGE = SEEDON_CONFIG.clip_range
ENT_COEF = SEEDON_CONFIG.ent_coef
VF_COEF = SEEDON_CONFIG.vf_coef
MAX_GRAD_NORM = SEEDON_CONFIG.max_grad_norm
NET_ARCH = SEEDON_CONFIG.net_arch
MAX_EPISODE_STEPS = SEEDON_CONFIG.max_episode_steps
SCENE_PATH = Path(os.environ.get("SEEDON_SCENE_PATH", str(DEFAULT_SCENE_PATH))).expanduser()


class SeedonMetricsCallback(BaseCallback):
    """Record and print compact standing metrics during PPO training."""

    LOG_FREQ = 2_048
    PRINT_FREQ = 10_000

    def __init__(self, total_timesteps: int):
        super().__init__(0)
        self._total_timesteps = total_timesteps
        self._ep_rewards: deque[float] = deque(maxlen=50)
        self._ep_lengths: deque[int] = deque(maxlen=50)
        self._base_heights: deque[float] = deque(maxlen=500)
        self._uprights: deque[float] = deque(maxlen=500)
        self._forward_velocities: deque[float] = deque(maxlen=500)
        self._last_print = 0
        self._last_log = 0
        self._episode_count = 0
        self._best_reward = -np.inf
        self._started_at = 0.0

    def _on_training_start(self) -> None:
        self._started_at = time.time()
        print(
            f"\n{'Steps':>12}  {'Eps':>6}  {'MeanR':>9}  "
            f"{'MeanLen':>8}  {'BaseZ':>7}  {'Upright':>7}  "
            f"{'FwdV':>7}  {'BestR':>8}  {'FPS':>6}  {'ETA':>9}",
        )
        print("-" * 101)

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "base_height" in info:
                self._base_heights.append(float(info["base_height"]))
            if "upright" in info:
                self._uprights.append(float(info["upright"]))
            if "forward_velocity" in info:
                self._forward_velocities.append(float(info["forward_velocity"]))
            episode = info.get("episode")
            if episode:
                reward = float(episode["r"])
                self._ep_rewards.append(reward)
                self._ep_lengths.append(int(episode["l"]))
                self._episode_count += 1
                self._best_reward = max(self._best_reward, reward)

        if self.num_timesteps - self._last_log >= self.LOG_FREQ:
            self._last_log = self.num_timesteps
            if self._ep_rewards:
                self.logger.record("episode/mean_reward", float(np.mean(self._ep_rewards)))
                self.logger.record("episode/mean_length", float(np.mean(self._ep_lengths)))
            if self._base_heights:
                self.logger.record("seedon/base_height", float(np.mean(self._base_heights)))
            if self._uprights:
                self.logger.record("seedon/upright", float(np.mean(self._uprights)))
            if self._forward_velocities:
                self.logger.record(
                    "seedon/forward_velocity",
                    float(np.mean(self._forward_velocities)),
                )

        if self.num_timesteps - self._last_print >= self.PRINT_FREQ:
            self._last_print = self.num_timesteps
            elapsed = time.time() - self._started_at
            fps = int(self.num_timesteps / elapsed) if elapsed > 0 else 0
            remaining = ((self._total_timesteps - self.num_timesteps) / fps) if fps > 0 else 0
            minutes, seconds = divmod(int(remaining), 60)
            hours, minutes = divmod(minutes, 60)
            mean_reward = float(np.mean(self._ep_rewards)) if self._ep_rewards else float("nan")
            mean_length = float(np.mean(self._ep_lengths)) if self._ep_lengths else float("nan")
            base_z = float(np.mean(self._base_heights)) if self._base_heights else float("nan")
            upright = float(np.mean(self._uprights)) if self._uprights else float("nan")
            forward_velocity = (
                float(np.mean(self._forward_velocities))
                if self._forward_velocities
                else float("nan")
            )
            print(
                f"{self.num_timesteps:>12,}  {self._episode_count:>6}  "
                f"{mean_reward:>9.1f}  {mean_length:>8.1f}  "
                f"{base_z:>7.3f}  {upright:>7.3f}  {forward_velocity:>7.3f}  "
                f"{self._best_reward:>8.1f}  {fps:>6}  "
                f"{hours:02d}:{minutes:02d}:{seconds:02d}",
            )
        return True


class SeedonVecNormalizeCheckpointCallback(BaseCallback):
    """Save VecNormalize stats whenever a model checkpoint is written."""

    def __init__(self, save_freq: int):
        super().__init__(0)
        self._save_freq = save_freq

    def _on_step(self) -> bool:
        if self.n_calls % self._save_freq == 0:
            self.training_env.save(VECNORM_PATH)
            versioned_path = os.path.join(
                MODEL_ROOT,
                f"{VECNORM_PREFIX}_{self.num_timesteps}_steps.pkl",
            )
            self.training_env.save(versioned_path)
        return True


class SeedonBestVecNormalizeCallback(BaseCallback):
    """Save VecNormalize stats when EvalCallback updates best_model.zip."""

    def __init__(self, best_model_dir: str, eval_env: VecNormalize):
        super().__init__(0)
        self._best_model_path = os.path.join(best_model_dir, "best_model.zip")
        self._eval_env = eval_env
        self._last_mtime = 0.0

    def _on_step(self) -> bool:
        if os.path.exists(self._best_model_path):
            mtime = os.path.getmtime(self._best_model_path)
            if mtime > self._last_mtime:
                self._last_mtime = mtime
                self._eval_env.save(BEST_VECNORM_PATH)
        return True


def parse_pose_weight_schedule(raw_schedule: str) -> list[tuple[int, float]]:
    """Parse ``step:pose_weight`` schedule entries.

    Args:
        raw_schedule: Comma-separated entries such as ``25000:6,50000:4``.

    Returns:
        Step-sorted schedule pairs.

    Raises:
        ValueError: If an entry is malformed or steps are not positive.
    """
    if not raw_schedule.strip():
        return []
    schedule: list[tuple[int, float]] = []
    for raw_entry in raw_schedule.split(","):
        entry = raw_entry.strip()
        if not entry:
            continue
        if ":" not in entry:
            raise ValueError(f"Invalid pose schedule entry: {entry}")
        raw_step, raw_weight = entry.split(":", 1)
        step = int(raw_step)
        weight = float(raw_weight)
        if step <= 0:
            raise ValueError("pose weight schedule steps must be positive.")
        if weight <= 0.0:
            raise ValueError("pose weights must be positive.")
        schedule.append((step, weight))
    return sorted(schedule)


class SeedonPoseWeightAnnealCallback(BaseCallback):
    """Anneal pose tracking weight during a curriculum run."""

    def __init__(
        self,
        *,
        schedule: list[tuple[int, float]],
        base_config: SeedonStandingConfig,
        config_path: str,
        eval_env: VecNormalize,
    ) -> None:
        super().__init__(0)
        self._schedule = schedule
        self._base_config = base_config
        self._config_path = config_path
        self._eval_env = eval_env
        self._next_index = 0

    def _on_training_start(self) -> None:
        write_json(self._config_path, asdict(self._base_config))
        print(f"Pose weight starts at {self._base_config.pose_weight:.3f}")

    def _on_step(self) -> bool:
        while (
            self._next_index < len(self._schedule)
            and self.num_timesteps >= self._schedule[self._next_index][0]
        ):
            step, pose_weight = self._schedule[self._next_index]
            self._base_config = replace(self._base_config, pose_weight=pose_weight)
            self.training_env.env_method("set_reward_config", self._base_config)
            self._eval_env.env_method("set_reward_config", self._base_config)
            write_json(self._config_path, asdict(self._base_config))
            print(f"Pose weight annealed to {pose_weight:.3f} at {self.num_timesteps} steps")
            self._next_index += 1
        return True


class SeedonTeacherAuditStopCallback(BaseCallback):
    """Stop teacher-imitation training when a checkpoint breaks the teacher gait."""

    def __init__(
        self,
        *,
        audit_freq: int,
        config_path: str,
        baseline_config_path: str | None,
        steps: int,
        seed: int,
        audit_warmup_steps: int = 20,
        landing_impact_limit: float = 1.56,
        tracking_error_multiplier: float = 1.5,
    ) -> None:
        super().__init__(0)
        self._audit_freq = audit_freq
        self._config_path = Path(config_path)
        self._strict_teacher_comparison = baseline_config_path is not None
        self._baseline_config_path = Path(baseline_config_path) if baseline_config_path else Path(config_path)
        self._steps = steps
        self._seed = seed
        self._audit_warmup_steps = audit_warmup_steps
        self._landing_impact_limit = landing_impact_limit
        self._tracking_error_multiplier = tracking_error_multiplier
        self._teacher_landing_impact = float("inf")
        self._teacher_base_height_drop = float("inf")
        self._teacher_tracking_limit = float("inf")

    def _on_training_start(self) -> None:
        from tools.audit_seedon_shuffle_v0 import audit_shuffle

        baseline = audit_shuffle(
            self._baseline_config_path,
            None,
            None,
            self._steps,
            self._seed,
            audit_warmup_steps=self._audit_warmup_steps,
        )
        self._teacher_landing_impact = baseline.landing_impact
        self._teacher_base_height_drop = baseline.base_height_drop
        self._teacher_tracking_limit = (
            baseline.mean_tracking_error * self._tracking_error_multiplier
        )
        print(
            "Teacher audit baseline: "
            f"support={baseline.peak_support_ratio:.3f}, "
            f"clearance={baseline.max_clearance:.6f}, "
            f"drop_post={baseline.base_height_drop_post_warmup:.5f} "
            f"raw_drop={baseline.base_height_drop_raw:.5f}, "
            f"impact_post={baseline.landing_impact_post_warmup:.3f} "
            f"raw_impact={baseline.landing_impact_raw:.3f}, "
            f"force_post={baseline.max_contact_force_post_warmup:.2f}, "
            f"foot_v_post={baseline.foot_velocity_near_contact_post_warmup:.6f}, "
            f"track={baseline.mean_tracking_error:.5f}; "
            f"tracking_limit={self._teacher_tracking_limit:.5f}",
        )

    def _on_step(self) -> bool:
        if self.n_calls % self._audit_freq != 0:
            return True

        from tools.audit_seedon_shuffle_v0 import audit_shuffle, teacher_relative_gate

        os.makedirs(TEACHER_AUDIT_DIR, exist_ok=True)
        stem = f"teacher_audit_{self.num_timesteps}_steps"
        model_path = Path(TEACHER_AUDIT_DIR) / f"{stem}.zip"
        vecnorm_path = Path(TEACHER_AUDIT_DIR) / f"{stem}_vecnorm.pkl"
        self.model.save(model_path)
        self.training_env.save(str(vecnorm_path))
        summary = audit_shuffle(
            self._config_path,
            model_path,
            vecnorm_path,
            self._steps,
            self._seed,
            audit_warmup_steps=self._audit_warmup_steps,
        )
        baseline = audit_shuffle(
            self._baseline_config_path,
            None,
            None,
            self._steps,
            self._seed,
            audit_warmup_steps=self._audit_warmup_steps,
        )
        gate = teacher_relative_gate(
            baseline,
            summary,
            tracking_error_multiplier=self._tracking_error_multiplier,
        )
        print(
            "Teacher checkpoint audit "
            f"@ {self.num_timesteps} steps: "
            f"pass={gate.passed}, none={summary.contact_none_ratio:.3f}, "
            f"jump={summary.jump_count}, support={summary.peak_support_ratio:.3f}, "
            f"clearance={summary.max_clearance:.6f}, "
            f"drop_post={summary.base_height_drop_post_warmup:.5f} "
            f"raw_drop={summary.base_height_drop_raw:.5f}, "
            f"impact_post={summary.landing_impact_post_warmup:.3f} "
            f"raw_impact={summary.landing_impact_raw:.3f}, "
            f"force_post={summary.max_contact_force_post_warmup:.2f}, "
            f"foot_v_post={summary.foot_velocity_near_contact_post_warmup:.6f}, "
            f"track={summary.mean_tracking_error:.5f}/{summary.max_tracking_error:.5f}, "
            f"failed={','.join(gate.reasons) or 'none'}",
        )
        return gate.passed


class SeedonForwardEvalCallback(BaseCallback):
    """Save checkpoints selected by forward progress instead of reward only."""

    def __init__(
        self,
        eval_env: VecNormalize,
        *,
        eval_freq: int,
        n_eval_episodes: int,
    ) -> None:
        super().__init__(0)
        self._eval_env = eval_env
        self._eval_freq = eval_freq
        self._n_eval_episodes = n_eval_episodes
        self._best_forward_velocity = -np.inf
        self._best_stable_forward_velocity = -np.inf
        self._standing_thresholds = SeedonStandingConfig()
        self._forward_thresholds = ForwardCheckpointThresholds(
            min_mean_length=0.9 * MAX_EPISODE_STEPS,
        )

    def _on_step(self) -> bool:
        if self.n_calls % self._eval_freq != 0:
            return True

        metrics = self._evaluate_policy()
        self.logger.record("eval_forward/mean_forward_velocity", metrics["mean_forward_velocity"])
        self.logger.record("eval_forward/mean_final_base_x", metrics["mean_final_base_x"])
        self.logger.record("eval_forward/mean_final_base_z", metrics["mean_final_base_z"])
        self.logger.record("eval_forward/mean_final_upright", metrics["mean_final_upright"])
        self.logger.record("eval_forward/fall_rate", metrics["fall_rate"])
        self.logger.record("eval_forward/both_contact_ratio", metrics["both_contact_ratio"])
        self.logger.record("eval_forward/single_contact_ratio", metrics["single_contact_ratio"])
        self.logger.record("eval_forward/no_contact_ratio", metrics["no_contact_ratio"])

        mean_forward_velocity = metrics["mean_forward_velocity"]
        safe_forward = is_safe_forward_candidate(metrics, self._forward_thresholds)
        if safe_forward and mean_forward_velocity > self._best_forward_velocity:
            self._best_forward_velocity = mean_forward_velocity
            self._save_checkpoint(
                FORWARD_BEST_MODEL_DIR,
                FORWARD_BEST_VECNORM_PATH,
                "forward_best_model",
            )
            print(
                "New best forward checkpoint: "
                f"FwdV={mean_forward_velocity:.3f}, "
                f"Fall={metrics['fall_rate']:.1%}, "
                f"Len={metrics['mean_length']:.1f}, "
                f"NoContact={metrics['no_contact_ratio']:.1%}",
            )

        stable = is_stable_forward_candidate(
            metrics,
            self._forward_thresholds,
            min_base_height=self._standing_thresholds.min_base_height,
            min_upright=self._standing_thresholds.min_upright,
            max_episode_steps=MAX_EPISODE_STEPS,
        )
        if stable and mean_forward_velocity > self._best_stable_forward_velocity:
            self._best_stable_forward_velocity = mean_forward_velocity
            self._save_checkpoint(
                STABLE_FORWARD_BEST_MODEL_DIR,
                STABLE_FORWARD_BEST_VECNORM_PATH,
                "stable_forward_best_model",
            )
            print(
                "New best stable-forward checkpoint: "
                f"FwdV={mean_forward_velocity:.3f}, "
                f"X={metrics['mean_final_base_x']:.3f}, "
                f"BothContact={metrics['both_contact_ratio']:.1%}",
            )
        return True

    def _evaluate_policy(self) -> dict[str, float]:
        episode_lengths: list[int] = []
        final_base_x: list[float] = []
        final_base_z: list[float] = []
        final_upright: list[float] = []
        forward_velocities: list[float] = []
        both_contact_ratios: list[float] = []
        single_contact_ratios: list[float] = []
        no_contact_ratios: list[float] = []
        falls = 0

        for _ in range(self._n_eval_episodes):
            obs = self._eval_env.reset()
            last_info: dict = {}
            forward_velocity_sum = 0.0
            both_contact_steps = 0
            single_contact_steps = 0
            no_contact_steps = 0
            length = 0
            while length < MAX_EPISODE_STEPS:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, _, dones, infos = self._eval_env.step(action)
                last_info = infos[0]
                forward_velocity_sum += float(last_info.get("forward_velocity", 0.0))
                left_contact = bool(last_info.get("left_contact", False))
                right_contact = bool(last_info.get("right_contact", False))
                if left_contact and right_contact:
                    both_contact_steps += 1
                elif left_contact or right_contact:
                    single_contact_steps += 1
                else:
                    no_contact_steps += 1
                length += 1
                if bool(dones[0]):
                    break
            episode_lengths.append(length)
            final_base_x.append(float(last_info.get("base_x_position", 0.0)))
            final_base_z.append(float(last_info.get("base_height", 0.0)))
            final_upright.append(float(last_info.get("upright", 0.0)))
            forward_velocities.append(forward_velocity_sum / max(1, length))
            both_contact_ratios.append(both_contact_steps / max(1, length))
            single_contact_ratios.append(single_contact_steps / max(1, length))
            no_contact_ratios.append(no_contact_steps / max(1, length))
            unhealthy_final_state = (
                final_base_z[-1] < self._standing_thresholds.min_base_height
                or final_upright[-1] < self._standing_thresholds.min_upright
            )
            if length < MAX_EPISODE_STEPS or unhealthy_final_state:
                falls += 1

        return {
            "mean_length": float(np.mean(episode_lengths)),
            "fall_rate": falls / max(1, self._n_eval_episodes),
            "mean_final_base_x": float(np.mean(final_base_x)),
            "mean_final_base_z": float(np.mean(final_base_z)),
            "mean_final_upright": float(np.mean(final_upright)),
            "mean_forward_velocity": float(np.mean(forward_velocities)),
            "both_contact_ratio": float(np.mean(both_contact_ratios)),
            "single_contact_ratio": float(np.mean(single_contact_ratios)),
            "no_contact_ratio": float(np.mean(no_contact_ratios)),
        }

    def _save_checkpoint(
        self,
        model_dir: str,
        vecnorm_path: str,
        model_name: str,
    ) -> None:
        os.makedirs(model_dir, exist_ok=True)
        self.model.save(os.path.join(model_dir, model_name))
        self._eval_env.save(vecnorm_path)


def _compute_batch_size(n_envs: int) -> int:
    """Return a PPO batch size compatible with rollout size."""
    return compute_ppo_batch_size(n_envs, N_STEPS, minimum=128)


def _make_env(seed: int, rank: int, reset_noise_scale: float):
    """Build one monitored Seedon standing environment."""

    def _thunk():
        env = SeedonStandingEnv(scene_path=SCENE_PATH, reset_noise_scale=reset_noise_scale)
        env = TimeLimit(env, max_episode_steps=MAX_EPISODE_STEPS)
        env.reset(seed=seed + rank)
        return Monitor(env)

    return _thunk


def _build_vec_env(n_envs: int, seed: int, reset_noise_scale: float):
    """Create a vectorized Seedon training environment."""
    env_fns = [_make_env(seed, rank, reset_noise_scale) for rank in range(n_envs)]
    if n_envs == 1:
        return DummyVecEnv(env_fns)
    return SubprocVecEnv(env_fns)


def _build_train_env(
    n_envs: int,
    seed: int,
    reset_noise_scale: float,
    resume_vecnorm_path: Path | None,
) -> VecNormalize:
    """Create the normalized training environment."""
    train_vec = _build_vec_env(n_envs, seed, reset_noise_scale)
    if resume_vecnorm_path is not None:
        train_env = VecNormalize.load(str(resume_vecnorm_path), train_vec)
        train_env.training = True
        train_env.norm_reward = True
        return train_env
    return VecNormalize(
        train_vec,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        gamma=GAMMA,
    )


def _resolve_resume_vecnorm_path(
    resume_model_path: str | None,
    explicit_vecnorm_path: str | None,
) -> Path | None:
    """Return VecNormalize stats to use when resuming a PPO checkpoint."""
    if explicit_vecnorm_path:
        vecnorm_path = Path(explicit_vecnorm_path)
        if not vecnorm_path.is_file():
            raise FileNotFoundError(f"--resume-vecnorm not found: {vecnorm_path}")
        return vecnorm_path
    if not resume_model_path:
        return None

    resume_path = Path(resume_model_path)
    candidates = [
        resume_path.with_name("vecnorm.pkl"),
        resume_path.parent / "vecnorm.pkl",
        resume_path.parent.parent / "vecnorm.pkl",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    if os.path.exists(VECNORM_PATH):
        return Path(VECNORM_PATH)
    return None


def _build_eval_env(
    seed: int,
    reset_noise_scale: float,
    train_env: VecNormalize,
) -> VecNormalize:
    """Create a deterministic eval environment sharing observation stats."""
    eval_vec = DummyVecEnv([_make_env(seed + 10_000, 0, reset_noise_scale)])
    eval_env = VecNormalize(
        eval_vec,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
        gamma=GAMMA,
    )
    eval_env.obs_rms = train_env.obs_rms
    eval_env.training = False
    eval_env.norm_reward = False
    return eval_env


def _save_config(args: argparse.Namespace, n_envs: int, batch_size: int) -> None:
    """Persist the Seedon training config for reproducibility."""
    os.makedirs(LOG_ROOT, exist_ok=True)
    cfg = {
        "artifacts": {
            "models_root": MODEL_ROOT,
            "logs_root": LOG_ROOT,
            "tb_root": TB_ROOT,
        },
        "scene_path": str(SCENE_PATH),
        "n_envs": n_envs,
        "batch_size": batch_size,
        "total_timesteps": _resolve_total_timesteps(args),
        "n_steps": N_STEPS,
        "n_epochs": N_EPOCHS,
        "gamma": GAMMA,
        "gae_lambda": GAE_LAMBDA,
        "learning_rate": LEARNING_RATE,
        "clip_range": CLIP_RANGE,
        "ent_coef": ENT_COEF,
        "vf_coef": VF_COEF,
        "max_grad_norm": MAX_GRAD_NORM,
        "net_arch": NET_ARCH,
        "max_episode_steps": MAX_EPISODE_STEPS,
        "reset_noise_scale": args.reset_noise_scale,
        "resume": args.resume,
        "resume_vecnorm": args.resume_vecnorm,
        "resume_action_std": args.resume_action_std,
        "action_std": args.action_std,
        "log_std_init": args.log_std_init,
        "checkpoint_freq_steps": args.checkpoint_freq_steps,
        "teacher_audit_freq_steps": args.teacher_audit_freq_steps,
        "teacher_audit_steps": args.teacher_audit_steps,
        "teacher_audit_warmup_steps": args.teacher_audit_warmup_steps,
        "pose_weight_schedule": args.pose_weight_schedule,
        "teacher_baseline_config": args.teacher_baseline_config,
        "reward_config": asdict(load_seedon_config_from_env()),
    }
    write_json(CONFIG_PATH, cfg)
    write_json(EFFECTIVE_REWARD_CONFIG_PATH, cfg["reward_config"])


def _write_manifest() -> None:
    """Write a small run manifest."""
    write_run_manifest(
        MANIFEST_PATH,
        repo_root=REPO_ROOT,
        command=sys.argv,
        models_root=MODEL_ROOT,
        logs_root=LOG_ROOT,
        tb_root=TB_ROOT,
        managed_layout=PATHS.managed_layout,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line options for Seedon PPO training."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Run a short sanity training.")
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=None,
        help="Override configured total training timesteps.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--n-envs",
        type=int,
        default=N_ENVS_DEFAULT,
        help="Number of parallel environments.",
    )
    parser.add_argument(
        "--reset-noise-scale",
        type=float,
        default=0.01,
        help="Uniform reset noise applied to actuated joint positions.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to an existing PPO zip checkpoint to resume from.",
    )
    parser.add_argument(
        "--resume-vecnorm",
        type=str,
        default=None,
        help="Path to VecNormalize stats for the resumed checkpoint.",
    )
    parser.add_argument(
        "--resume-action-std",
        type=float,
        default=0.03,
        help="Stochastic policy action std to use when fine-tuning a resumed checkpoint.",
    )
    parser.add_argument(
        "--action-std",
        type=float,
        default=None,
        help="Stochastic policy action std to use after creating or loading a model.",
    )
    parser.add_argument(
        "--log-std-init",
        type=float,
        default=None,
        help="Initial PPO policy log std for new models, e.g. log(0.1)=-2.3026.",
    )
    parser.add_argument(
        "--checkpoint-freq-steps",
        type=int,
        default=SEEDON_CONFIG.checkpoint_freq_steps,
        help="Environment timesteps between checkpoint saves.",
    )
    parser.add_argument(
        "--teacher-audit-freq-steps",
        type=int,
        default=0,
        help="Run teacher safety audit every N environment timesteps; 0 disables it.",
    )
    parser.add_argument(
        "--teacher-audit-steps",
        type=int,
        default=480,
        help="Rollout steps for each teacher checkpoint audit.",
    )
    parser.add_argument(
        "--teacher-audit-warmup-steps",
        type=int,
        default=20,
        help="Initial audit rollout steps excluded from landing/drop strict gates.",
    )
    parser.add_argument(
        "--teacher-baseline-config",
        type=str,
        default=None,
        help="Config used as the baseline for strict teacher-relative audit gates.",
    )
    parser.add_argument(
        "--pose-weight-schedule",
        type=str,
        default="",
        help="Comma-separated step:pose_weight schedule, e.g. 25000:6,50000:4.",
    )
    return parser.parse_args(argv)


def _resolve_total_timesteps(args: argparse.Namespace) -> int:
    """Return the requested number of training timesteps."""
    if args.total_timesteps is not None:
        if args.total_timesteps <= 0:
            raise ValueError("--total-timesteps must be positive.")
        return args.total_timesteps
    return SMOKE_TIMESTEPS if args.smoke else TOTAL_TIMESTEPS


def main(argv: list[str] | None = None) -> int:
    """Train the Seedon standing baseline with PPO."""
    args = parse_args(argv)
    if args.n_envs <= 0:
        raise ValueError("--n-envs must be positive.")
    if args.reset_noise_scale < 0.0:
        raise ValueError("--reset-noise-scale must be non-negative.")
    if args.resume_action_std <= 0.0:
        raise ValueError("--resume-action-std must be positive.")
    if args.action_std is not None and args.action_std <= 0.0:
        raise ValueError("--action-std must be positive.")
    if args.checkpoint_freq_steps <= 0:
        raise ValueError("--checkpoint-freq-steps must be positive.")
    if args.teacher_audit_freq_steps < 0:
        raise ValueError("--teacher-audit-freq-steps must be non-negative.")
    if args.teacher_audit_steps <= 0:
        raise ValueError("--teacher-audit-steps must be positive.")
    if args.teacher_audit_warmup_steps < 0:
        raise ValueError("--teacher-audit-warmup-steps must be non-negative.")
    pose_weight_schedule = parse_pose_weight_schedule(args.pose_weight_schedule)
    if args.teacher_baseline_config is not None and not Path(args.teacher_baseline_config).is_file():
        raise FileNotFoundError(f"--teacher-baseline-config not found: {args.teacher_baseline_config}")
    if not SCENE_PATH.is_file():
        raise FileNotFoundError(
            f"Seedon training scene not found: {SCENE_PATH}. "
            "Run `python -m tools.convert_urdf_to_mjcf` and "
            "`python -m tools.build_seedon_training_scene` first."
        )

    ensure_dirs(
        MODEL_ROOT,
        LOG_ROOT,
        BEST_MODEL_DIR,
        FORWARD_BEST_MODEL_DIR,
        STABLE_FORWARD_BEST_MODEL_DIR,
        TEACHER_AUDIT_DIR,
        TB_ROOT,
    )

    total_timesteps = _resolve_total_timesteps(args)
    batch_size = _compute_batch_size(args.n_envs)
    resume_vecnorm_path = _resolve_resume_vecnorm_path(args.resume, args.resume_vecnorm)
    base_reward_config = load_seedon_config_from_env()
    _save_config(args, args.n_envs, batch_size)
    _write_manifest()
    print(f"Artifacts: models={MODEL_ROOT} logs={LOG_ROOT} tb={TB_ROOT}")
    if args.resume:
        print(f"Resume model: {args.resume}")
        print(f"Resume VecNormalize: {resume_vecnorm_path or '(fresh stats)'}")
        print(f"Resume action std: {args.resume_action_std}")
    if args.action_std is not None:
        print(f"Action std override: {args.action_std}")

    train_env = _build_train_env(
        n_envs=args.n_envs,
        seed=args.seed,
        reset_noise_scale=args.reset_noise_scale,
        resume_vecnorm_path=resume_vecnorm_path,
    )
    eval_env = _build_eval_env(
        seed=args.seed,
        reset_noise_scale=0.0,
        train_env=train_env,
    )

    checkpoint_save_freq = max(1, args.checkpoint_freq_steps // args.n_envs)
    eval_freq = max(1, SEEDON_CONFIG.eval_freq_steps // args.n_envs)
    teacher_audit_freq = (
        max(1, args.teacher_audit_freq_steps // args.n_envs)
        if args.teacher_audit_freq_steps > 0
        else 0
    )

    callback_list: list[BaseCallback] = [
        SeedonMetricsCallback(total_timesteps=total_timesteps),
    ]
    if pose_weight_schedule:
        callback_list.append(
            SeedonPoseWeightAnnealCallback(
                schedule=pose_weight_schedule,
                base_config=base_reward_config,
                config_path=EFFECTIVE_REWARD_CONFIG_PATH,
                eval_env=eval_env,
            )
        )
    callback_list.extend(
        [
        CheckpointCallback(
            save_freq=checkpoint_save_freq,
            save_path=MODEL_ROOT,
            name_prefix="seedon_ppo",
        ),
        SeedonVecNormalizeCheckpointCallback(save_freq=checkpoint_save_freq),
        EvalCallback(
            eval_env,
            best_model_save_path=BEST_MODEL_DIR,
            log_path=LOG_ROOT,
            eval_freq=eval_freq,
            deterministic=True,
            render=False,
            n_eval_episodes=SEEDON_CONFIG.eval_episodes,
        ),
        SeedonForwardEvalCallback(
            eval_env,
            eval_freq=eval_freq,
            n_eval_episodes=SEEDON_CONFIG.eval_episodes,
        ),
        SeedonBestVecNormalizeCallback(BEST_MODEL_DIR, eval_env),
        ]
    )
    if teacher_audit_freq:
        callback_list.append(
            SeedonTeacherAuditStopCallback(
                audit_freq=teacher_audit_freq,
                config_path=EFFECTIVE_REWARD_CONFIG_PATH,
                baseline_config_path=args.teacher_baseline_config,
                steps=args.teacher_audit_steps,
                seed=args.seed,
                audit_warmup_steps=args.teacher_audit_warmup_steps,
            )
        )

    policy_kwargs = {"net_arch": NET_ARCH}
    if args.log_std_init is not None:
        policy_kwargs["log_std_init"] = args.log_std_init
    model_kwargs = dict(
        policy="MlpPolicy",
        env=train_env,
        n_steps=N_STEPS,
        batch_size=batch_size,
        n_epochs=N_EPOCHS,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        learning_rate=LEARNING_RATE,
        clip_range=CLIP_RANGE,
        ent_coef=ENT_COEF,
        vf_coef=VF_COEF,
        max_grad_norm=MAX_GRAD_NORM,
        policy_kwargs=policy_kwargs,
        tensorboard_log=TB_ROOT,
        verbose=0,
        seed=args.seed,
    )

    if args.resume:
        model = PPO.load(args.resume, env=train_env)
        if hasattr(model.policy, "log_std"):
            model.policy.log_std.data.fill_(float(np.log(args.resume_action_std)))
        model.set_random_seed(args.seed)
    else:
        model = PPO(**model_kwargs)
    if args.action_std is not None and hasattr(model.policy, "log_std"):
        model.policy.log_std.data.fill_(float(np.log(args.action_std)))

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback_list,
            tb_log_name="seedon_standing",
        )
        model.save(LATEST_MODEL_PATH)
        train_env.save(VECNORM_PATH)
    finally:
        eval_env.close()
        train_env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
