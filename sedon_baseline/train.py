"""PPO training entrypoint for the Sedon standing baseline."""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import deque
from dataclasses import asdict
from pathlib import Path

import numpy as np
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from sedon_baseline.env import (
    DEFAULT_SCENE_PATH,
    SedonStandingConfig,
    SedonStandingEnv,
)
from robot_learning.training_config import load_sedon_train_config
from robot_learning.training_paths import resolve_training_paths
from robot_learning.training_runtime import (
    compute_ppo_batch_size,
    ensure_dirs,
    write_json,
    write_run_manifest,
)

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
SEDON_CONFIG = load_sedon_train_config(REPO_ROOT)
PATHS = resolve_training_paths(
    REPO_ROOT,
    "sedon",
    legacy_model_dir=os.path.join("models", "sedon"),
    legacy_log_dir=os.path.join("logs", "sedon"),
    legacy_tb_dir=os.path.join("logs", "tb", "sedon"),
)

MODEL_ROOT = str(PATHS.models_root)
LOG_ROOT = str(PATHS.logs_root)
TB_ROOT = str(PATHS.tb_root)
VECNORM_PATH = os.path.join(MODEL_ROOT, "vecnorm.pkl")
BEST_MODEL_DIR = os.path.join(MODEL_ROOT, "best")
LATEST_MODEL_PATH = os.path.join(MODEL_ROOT, "latest_model")
CONFIG_PATH = os.path.join(LOG_ROOT, "train_config.json")
MANIFEST_PATH = os.path.join(LOG_ROOT, "run_manifest.json")

N_ENVS_DEFAULT = int(os.environ.get("SEDON_N_ENVS", str(SEDON_CONFIG.n_envs_default)))
TOTAL_TIMESTEPS = SEDON_CONFIG.total_timesteps
SMOKE_TIMESTEPS = SEDON_CONFIG.smoke_timesteps
N_STEPS = SEDON_CONFIG.n_steps
N_EPOCHS = SEDON_CONFIG.n_epochs
GAMMA = SEDON_CONFIG.gamma
GAE_LAMBDA = SEDON_CONFIG.gae_lambda
LEARNING_RATE = SEDON_CONFIG.learning_rate
CLIP_RANGE = SEDON_CONFIG.clip_range
ENT_COEF = SEDON_CONFIG.ent_coef
VF_COEF = SEDON_CONFIG.vf_coef
MAX_GRAD_NORM = SEDON_CONFIG.max_grad_norm
NET_ARCH = SEDON_CONFIG.net_arch
MAX_EPISODE_STEPS = SEDON_CONFIG.max_episode_steps


class SedonMetricsCallback(BaseCallback):
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
            f"{'BestR':>8}  {'FPS':>6}  {'ETA':>9}",
        )
        print("-" * 92)

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "base_height" in info:
                self._base_heights.append(float(info["base_height"]))
            if "upright" in info:
                self._uprights.append(float(info["upright"]))
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
                self.logger.record("sedon/base_height", float(np.mean(self._base_heights)))
            if self._uprights:
                self.logger.record("sedon/upright", float(np.mean(self._uprights)))

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
            print(
                f"{self.num_timesteps:>12,}  {self._episode_count:>6}  "
                f"{mean_reward:>9.1f}  {mean_length:>8.1f}  "
                f"{base_z:>7.3f}  {upright:>7.3f}  "
                f"{self._best_reward:>8.1f}  {fps:>6}  "
                f"{hours:02d}:{minutes:02d}:{seconds:02d}",
            )
        return True


def _compute_batch_size(n_envs: int) -> int:
    """Return a PPO batch size compatible with rollout size."""
    return compute_ppo_batch_size(n_envs, N_STEPS, minimum=128)


def _make_env(seed: int, rank: int, reset_noise_scale: float):
    """Build one monitored Sedon standing environment."""

    def _thunk():
        env = SedonStandingEnv(reset_noise_scale=reset_noise_scale)
        env = TimeLimit(env, max_episode_steps=MAX_EPISODE_STEPS)
        env.reset(seed=seed + rank)
        return Monitor(env)

    return _thunk


def _build_vec_env(n_envs: int, seed: int, reset_noise_scale: float):
    """Create a vectorized Sedon training environment."""
    env_fns = [_make_env(seed, rank, reset_noise_scale) for rank in range(n_envs)]
    if n_envs == 1:
        return DummyVecEnv(env_fns)
    return SubprocVecEnv(env_fns)


def _build_train_env(
    n_envs: int,
    seed: int,
    reset_noise_scale: float,
    resume: bool,
) -> VecNormalize:
    """Create the normalized training environment."""
    train_vec = _build_vec_env(n_envs, seed, reset_noise_scale)
    if resume and os.path.exists(VECNORM_PATH):
        train_env = VecNormalize.load(VECNORM_PATH, train_vec)
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
    """Persist the Sedon training config for reproducibility."""
    os.makedirs(LOG_ROOT, exist_ok=True)
    cfg = {
        "artifacts": {
            "models_root": MODEL_ROOT,
            "logs_root": LOG_ROOT,
            "tb_root": TB_ROOT,
        },
        "scene_path": str(DEFAULT_SCENE_PATH),
        "n_envs": n_envs,
        "batch_size": batch_size,
        "total_timesteps": SMOKE_TIMESTEPS if args.smoke else TOTAL_TIMESTEPS,
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
        "reward_config": asdict(SedonStandingConfig()),
    }
    write_json(CONFIG_PATH, cfg)


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
    """Parse command-line options for Sedon PPO training."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Run a short sanity training.")
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
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Train the Sedon standing baseline with PPO."""
    args = parse_args(argv)
    if args.n_envs <= 0:
        raise ValueError("--n-envs must be positive.")
    if args.reset_noise_scale < 0.0:
        raise ValueError("--reset-noise-scale must be non-negative.")
    if not DEFAULT_SCENE_PATH.is_file():
        raise FileNotFoundError(
            f"Sedon training scene not found: {DEFAULT_SCENE_PATH}. "
            "Run `python -m tools.convert_urdf_to_mjcf` and "
            "`python -m tools.build_sedon_training_scene` first."
        )

    ensure_dirs(MODEL_ROOT, LOG_ROOT, BEST_MODEL_DIR, TB_ROOT)

    total_timesteps = SMOKE_TIMESTEPS if args.smoke else TOTAL_TIMESTEPS
    batch_size = _compute_batch_size(args.n_envs)
    _save_config(args, args.n_envs, batch_size)
    _write_manifest()
    print(f"Artifacts: models={MODEL_ROOT} logs={LOG_ROOT} tb={TB_ROOT}")

    train_env = _build_train_env(
        n_envs=args.n_envs,
        seed=args.seed,
        reset_noise_scale=args.reset_noise_scale,
        resume=bool(args.resume),
    )
    eval_env = _build_eval_env(
        seed=args.seed,
        reset_noise_scale=0.0,
        train_env=train_env,
    )

    callback_list = [
        SedonMetricsCallback(total_timesteps=total_timesteps),
        CheckpointCallback(
            save_freq=max(1, SEDON_CONFIG.checkpoint_freq_steps // args.n_envs),
            save_path=MODEL_ROOT,
            name_prefix="sedon_ppo",
        ),
        EvalCallback(
            eval_env,
            best_model_save_path=BEST_MODEL_DIR,
            log_path=LOG_ROOT,
            eval_freq=max(1, SEDON_CONFIG.eval_freq_steps // args.n_envs),
            deterministic=True,
            render=False,
            n_eval_episodes=SEDON_CONFIG.eval_episodes,
        ),
    ]

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
        policy_kwargs={"net_arch": NET_ARCH},
        tensorboard_log=TB_ROOT,
        verbose=0,
        seed=args.seed,
    )

    if args.resume:
        model = PPO.load(args.resume, env=train_env)
        model.set_random_seed(args.seed)
    else:
        model = PPO(**model_kwargs)

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback_list,
            tb_log_name="sedon_standing",
        )
        model.save(LATEST_MODEL_PATH)
        train_env.save(VECNORM_PATH)
    finally:
        eval_env.close()
        train_env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
