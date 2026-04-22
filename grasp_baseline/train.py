"""PPO training entrypoint for the standalone fixed-base grasp baseline."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from grasp_baseline.env import FixedBaseGraspEnv, GraspRewardConfig

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
MODEL_ROOT = os.path.join(REPO_ROOT, "models", "grasp")
LOG_ROOT = os.path.join(REPO_ROOT, "logs", "grasp")
VECNORM_PATH = os.path.join(MODEL_ROOT, "vecnorm.pkl")
BEST_MODEL_DIR = os.path.join(MODEL_ROOT, "best")
LATEST_MODEL_PATH = os.path.join(MODEL_ROOT, "latest_model")
CONFIG_PATH = os.path.join(LOG_ROOT, "train_config.json")

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
    }
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, sort_keys=True)


def parse_args() -> argparse.Namespace:
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
    return parser.parse_args()


def main() -> None:
    """Train the standalone grasp baseline with PPO."""
    args = parse_args()
    if args.n_envs <= 0:
        raise ValueError("--n-envs must be positive.")

    os.makedirs(MODEL_ROOT, exist_ok=True)
    os.makedirs(LOG_ROOT, exist_ok=True)
    os.makedirs(BEST_MODEL_DIR, exist_ok=True)

    batch_size = _compute_batch_size(args.n_envs)
    _save_config(args, args.n_envs, batch_size)

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
        tensorboard_log=LOG_ROOT,
        verbose=1,
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
