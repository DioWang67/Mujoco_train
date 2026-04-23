"""PPO training entrypoint for the standalone fixed-base grasp baseline."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import deque
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from grasp_baseline.env import FixedBaseGraspEnv, GraspRewardConfig
from training_paths import resolve_training_paths

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
PATHS = resolve_training_paths(
    REPO_ROOT,
    "grasp",
    legacy_model_dir=os.path.join("models", "grasp"),
    legacy_log_dir=os.path.join("logs", "grasp"),
    legacy_tb_dir=os.path.join("logs", "tb", "grasp"),
)
MODEL_ROOT = str(PATHS.models_root)
LOG_ROOT = str(PATHS.logs_root)
TB_ROOT = str(PATHS.tb_root)
VECNORM_PATH = os.path.join(MODEL_ROOT, "vecnorm.pkl")
BEST_MODEL_DIR = os.path.join(MODEL_ROOT, "best")
LATEST_MODEL_PATH = os.path.join(MODEL_ROOT, "latest_model")
CONFIG_PATH = os.path.join(LOG_ROOT, "train_config.json")
MANIFEST_PATH = os.path.join(LOG_ROOT, "run_manifest.json")

N_ENVS_DEFAULT = int(os.environ.get("GRASP_N_ENVS", "8"))
TOTAL_TIMESTEPS = 2_000_000
SMOKE_TIMESTEPS = 20_000
N_STEPS = 512
N_EPOCHS = 10
GAMMA = 0.99
GAE_LAMBDA = 0.95
LEARNING_RATE = 3e-4
CLIP_RANGE = 0.2
ENT_COEF = 0.005
VF_COEF = 0.5
MAX_GRAD_NORM = 1.0
NET_ARCH = [256, 256]
MAX_EPISODE_STEPS = 300


def _git_commit_short() -> str:
    """Return the short git commit of the current code checkout."""
    import subprocess

    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(REPO_ROOT),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        return out or "unknown"
    except Exception:
        return "unknown"


class GraspMetricsCallback(BaseCallback):
    """Print H1-style progress plus grasp-specific success/fall rates."""

    LOG_FREQ = 4_096
    PRINT_FREQ = 20_000

    def __init__(self, total_timesteps: int):
        super().__init__(0)
        self._total_timesteps = total_timesteps
        self._ep_rewards: deque[float] = deque(maxlen=50)
        self._ep_lengths: deque[int] = deque(maxlen=50)
        self._ep_successes: deque[float] = deque(maxlen=50)
        self._ep_falls: deque[float] = deque(maxlen=50)
        self._reward_buffers: dict[str, deque[float]] = {}
        self._last_print = 0
        self._last_log = 0
        self._episode_count = 0
        self._best_reward = -np.inf
        self._started_at = 0.0

    def _on_training_start(self) -> None:
        self._started_at = time.time()
        print(
            f"\n{'Steps':>12}  {'Eps':>6}  {'MeanR':>9}  "
            f"{'MeanLen':>8}  {'Succ%':>7}  {'Fall%':>7}  "
            f"{'BestR':>8}  {'FPS':>6}  {'ETA':>9}",
        )
        print("-" * 92)

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            for key, value in info.items():
                if key.startswith("reward_"):
                    self._reward_buffers.setdefault(key, deque(maxlen=200)).append(float(value))
            episode = info.get("episode")
            if episode:
                self._ep_rewards.append(float(episode["r"]))
                self._ep_lengths.append(int(episode["l"]))
                self._ep_successes.append(float(bool(info.get("is_success", False))))
                self._ep_falls.append(float(bool(info.get("cube_fell", False))))
                self._episode_count += 1
                self._best_reward = max(self._best_reward, float(episode["r"]))

        if self.num_timesteps - self._last_log >= self.LOG_FREQ:
            self._last_log = self.num_timesteps
            for key, values in self._reward_buffers.items():
                if values:
                    self.logger.record(
                        f"reward/{key.removeprefix('reward_')}",
                        float(np.mean(values)),
                    )
            if self._ep_rewards:
                self.logger.record("episode/mean_reward", float(np.mean(self._ep_rewards)))
                self.logger.record("episode/mean_length", float(np.mean(self._ep_lengths)))
                self.logger.record("episode/success_rate", float(np.mean(self._ep_successes)))
                self.logger.record("episode/cube_fell_rate", float(np.mean(self._ep_falls)))

        if self.num_timesteps - self._last_print >= self.PRINT_FREQ:
            self._last_print = self.num_timesteps
            elapsed = time.time() - self._started_at
            fps = int(self.num_timesteps / elapsed) if elapsed > 0 else 0
            remaining = ((self._total_timesteps - self.num_timesteps) / fps) if fps > 0 else 0
            mean_reward = float(np.mean(self._ep_rewards)) if self._ep_rewards else float("nan")
            mean_length = float(np.mean(self._ep_lengths)) if self._ep_lengths else float("nan")
            success_rate = 100.0 * float(np.mean(self._ep_successes)) if self._ep_successes else float("nan")
            fall_rate = 100.0 * float(np.mean(self._ep_falls)) if self._ep_falls else float("nan")
            minutes, seconds = divmod(int(remaining), 60)
            hours, minutes = divmod(minutes, 60)
            print(
                f"{self.num_timesteps:>12,}  {self._episode_count:>6}  "
                f"{mean_reward:>9.1f}  {mean_length:>8.1f}  "
                f"{success_rate:>6.1f}%  {fall_rate:>6.1f}%  "
                f"{self._best_reward:>8.1f}  {fps:>6}  "
                f"{hours:02d}:{minutes:02d}:{seconds:02d}",
            )
        return True


def _compute_batch_size(n_envs: int) -> int:
    """Return a PPO batch size that is valid for the chosen env count."""
    rollout_size = n_envs * N_STEPS
    return max(256, min(rollout_size // 4, rollout_size))


def _make_env(
    seed: int,
    rank: int,
    task_phase: str,
    randomize_cube_pose: bool,
):
    """Build one monitored environment instance."""

    def _thunk():
        env = FixedBaseGraspEnv(
            task_phase=task_phase,
            randomize_cube_pose=randomize_cube_pose,
        )
        env = TimeLimit(env, max_episode_steps=MAX_EPISODE_STEPS)
        env.reset(seed=seed + rank)
        return Monitor(env)

    return _thunk


def _build_vec_env(
    n_envs: int,
    seed: int,
    task_phase: str,
    randomize_cube_pose: bool,
):
    """Create a vectorized training environment."""
    env_fns = [
        _make_env(seed, rank, task_phase, randomize_cube_pose)
        for rank in range(n_envs)
    ]
    if n_envs == 1:
        return DummyVecEnv(env_fns)
    return SubprocVecEnv(env_fns)


def _build_train_env(
    n_envs: int,
    seed: int,
    task_phase: str,
    randomize_cube_pose: bool,
    resume: bool,
) -> VecNormalize:
    """Create the normalized training environment, reusing VecNormalize stats."""
    train_vec = _build_vec_env(
        n_envs=n_envs,
        seed=seed,
        task_phase=task_phase,
        randomize_cube_pose=randomize_cube_pose,
    )
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
    task_phase: str,
    randomize_cube_pose: bool,
    train_env: VecNormalize,
) -> VecNormalize:
    """Create a deterministic eval env that reuses training observation stats."""
    eval_vec = DummyVecEnv(
        [_make_env(seed + 10_000, 0, task_phase, randomize_cube_pose)]
    )
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
    """Persist the grasp training config for reproducibility."""
    os.makedirs(LOG_ROOT, exist_ok=True)
    cfg = {
        "artifacts": {
            "models_root": MODEL_ROOT,
            "logs_root": LOG_ROOT,
            "tb_root": TB_ROOT,
        },
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
        "task_phase": args.phase,
        "randomize_cube_pose": not args.fixed_cube,
        "reward_config": asdict(GraspRewardConfig()),
        "max_episode_steps": MAX_EPISODE_STEPS,
    }
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, sort_keys=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line options for grasp training."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Run a short sanity training.")
    parser.add_argument(
        "--phase",
        choices=["reach", "grasp", "lift", "full"],
        default="full",
        help="Reward curriculum phase.",
    )
    parser.add_argument(
        "--fixed-cube",
        action="store_true",
        help="Disable cube XY randomization to debug scripted or early RL behavior.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to an existing PPO zip checkpoint to resume from.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--n-envs",
        type=int,
        default=N_ENVS_DEFAULT,
        help="Number of parallel environments.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Train the standalone grasp baseline with PPO."""
    args = parse_args(argv)
    if args.n_envs <= 0:
        raise ValueError("--n-envs must be positive.")

    os.makedirs(MODEL_ROOT, exist_ok=True)
    os.makedirs(LOG_ROOT, exist_ok=True)
    os.makedirs(BEST_MODEL_DIR, exist_ok=True)
    os.makedirs(TB_ROOT, exist_ok=True)

    batch_size = _compute_batch_size(args.n_envs)
    _save_config(args, args.n_envs, batch_size)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "created_at": datetime.now().isoformat(),
                "git_commit": _git_commit_short(),
                "command": " ".join(os.sys.argv),
                "models_root": MODEL_ROOT,
                "logs_root": LOG_ROOT,
                "tb_root": TB_ROOT,
                "managed_layout": PATHS.managed_layout,
            },
            f,
            indent=2,
            sort_keys=True,
        )
    print(f"Artifacts: models={MODEL_ROOT} logs={LOG_ROOT} tb={TB_ROOT}")

    train_env = _build_train_env(
        n_envs=args.n_envs,
        seed=args.seed,
        task_phase=args.phase,
        randomize_cube_pose=not args.fixed_cube,
        resume=bool(args.resume),
    )
    eval_env = _build_eval_env(
        seed=args.seed,
        task_phase=args.phase,
        randomize_cube_pose=not args.fixed_cube,
        train_env=train_env,
    )

    callback_list = [
        GraspMetricsCallback(
            total_timesteps=SMOKE_TIMESTEPS if args.smoke else TOTAL_TIMESTEPS,
        ),
        CheckpointCallback(
            save_freq=max(1, 100_000 // args.n_envs),
            save_path=MODEL_ROOT,
            name_prefix="grasp_ppo",
        ),
        EvalCallback(
            eval_env,
            best_model_save_path=BEST_MODEL_DIR,
            log_path=LOG_ROOT,
            eval_freq=max(1, 50_000 // args.n_envs),
            deterministic=True,
            render=False,
            n_eval_episodes=5,
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

    total_timesteps = SMOKE_TIMESTEPS if args.smoke else TOTAL_TIMESTEPS
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback_list,
        tb_log_name=f"grasp_{args.phase}",
    )
    model.save(LATEST_MODEL_PATH)
    train_env.save(VECNORM_PATH)

    eval_env.close()
    train_env.close()


if __name__ == "__main__":
    main()
