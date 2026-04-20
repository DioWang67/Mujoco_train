"""Optuna hyperparameter sweep for H1 walking.

Searches over reward scales and PPO hyperparameters using short
training runs (500k steps) to find promising configurations, then
the best config can be trained for the full 10M steps.

Usage:
    python sweep.py                    # run 20 trials
    python sweep.py --n-trials 50      # run 50 trials
    python sweep.py --steps 1000000    # longer per-trial budget
"""

import argparse
import json
import os
import sys

import numpy as np
import optuna
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from h1_env import H1Env, _DEFAULT_REWARD_SCALES

HERE = os.path.dirname(os.path.abspath(__file__))
SWEEP_DIR = os.path.join(HERE, "logs", "sweep")
os.makedirs(SWEEP_DIR, exist_ok=True)

# Short run defaults for sweep.
_DEFAULT_STEPS = 500_000
_N_EVAL_EPISODES = 5
_N_ENVS = 4


def make_env(rank: int = 0, seed: int = 0):
    def _init():
        env = H1Env()
        env = TimeLimit(env, max_episode_steps=1000)
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init


def make_env_with_scales(
    reward_scales: dict, rank: int = 0, seed: int = 0,
):
    def _init():
        env = H1Env(reward_scales=reward_scales)
        env = TimeLimit(env, max_episode_steps=1000)
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init


def objective(trial: optuna.Trial, total_steps: int) -> float:
    # ── Sample reward scales around the official baseline ─────────
    scales = {}
    scales["tracking_lin_vel"] = trial.suggest_float(
        "rw_tracking_lin_vel", 0.5, 2.0,
    )
    scales["tracking_ang_vel"] = trial.suggest_float(
        "rw_tracking_ang_vel", 0.1, 1.0,
    )
    scales["alive"] = trial.suggest_float(
        "rw_alive", 0.05, 0.5,
    )
    scales["contact"] = trial.suggest_float(
        "rw_contact", 0.05, 0.5,
    )
    scales["lin_vel_z"] = -trial.suggest_float(
        "rw_lin_vel_z", 0.5, 5.0,
    )
    scales["orientation"] = -trial.suggest_float(
        "rw_orientation", 0.2, 3.0,
    )
    scales["base_height"] = -trial.suggest_float(
        "rw_base_height", 2.0, 20.0,
    )
    scales["action_rate"] = -trial.suggest_float(
        "rw_action_rate", 0.001, 0.1, log=True,
    )
    scales["hip_pos"] = -trial.suggest_float(
        "rw_hip_pos", 0.2, 3.0,
    )
    # Keep these at baseline (less likely to need tuning).
    scales["ang_vel_xy"] = _DEFAULT_REWARD_SCALES["ang_vel_xy"]
    scales["dof_acc"] = _DEFAULT_REWARD_SCALES["dof_acc"]
    scales["collision"] = _DEFAULT_REWARD_SCALES["collision"]
    scales["dof_pos_limits"] = _DEFAULT_REWARD_SCALES["dof_pos_limits"]
    scales["contact_no_vel"] = _DEFAULT_REWARD_SCALES["contact_no_vel"]
    scales["feet_swing_height"] = _DEFAULT_REWARD_SCALES[
        "feet_swing_height"
    ]

    # ── Sample PPO hyperparameters ───────────────────────────────
    lr = trial.suggest_float("learning_rate", 1e-4, 1e-3, log=True)
    ent_coef = trial.suggest_float("ent_coef", 0.001, 0.02, log=True)
    gamma = trial.suggest_float("gamma", 0.95, 0.999)
    gae_lambda = trial.suggest_float("gae_lambda", 0.9, 0.99)
    clip_range = trial.suggest_float("clip_range", 0.1, 0.3)

    # ── Build environments ───────────────────────────────────────
    fns = [
        make_env_with_scales(scales, i) for i in range(_N_ENVS)
    ]
    vec_env = DummyVecEnv(fns)
    vec_env = VecNormalize(
        vec_env, norm_obs=True, norm_reward=True,
        clip_obs=10.0, gamma=gamma,
    )

    eval_fns = [make_env_with_scales(scales, 99)]
    eval_env = DummyVecEnv(eval_fns)
    eval_env = VecNormalize(
        eval_env, norm_obs=True, norm_reward=False,
        clip_obs=10.0, gamma=gamma,
    )
    eval_env.training = False

    trial_dir = os.path.join(SWEEP_DIR, f"trial_{trial.number:04d}")
    os.makedirs(trial_dir, exist_ok=True)

    # ── Train ────────────────────────────────────────────────────
    model = PPO(
        "MlpPolicy",
        vec_env,
        n_steps=2048,
        batch_size=512,
        n_epochs=10,
        learning_rate=lr,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        vf_coef=0.5,
        max_grad_norm=1.0,
        policy_kwargs={"net_arch": [256, 256]},
        verbose=0,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=trial_dir,
        log_path=trial_dir,
        eval_freq=max(total_steps // (_N_ENVS * 10), 1),
        n_eval_episodes=_N_EVAL_EPISODES,
        deterministic=True,
        verbose=0,
    )

    try:
        model.learn(
            total_timesteps=total_steps,
            callback=eval_callback,
        )
    except Exception as e:
        print(f"  Trial {trial.number} failed: {e}")
        vec_env.close()
        eval_env.close()
        return float("-inf")

    # ── Evaluate ─────────────────────────────────────────────────
    eval_path = os.path.join(trial_dir, "evaluations.npz")
    if os.path.exists(eval_path):
        data = np.load(eval_path)
        mean_rewards = data["results"].mean(axis=1)
        best_mean = float(mean_rewards.max())
    else:
        best_mean = float("-inf")

    # Save trial config.
    trial_cfg = {
        "trial": trial.number,
        "reward_scales": scales,
        "learning_rate": lr,
        "ent_coef": ent_coef,
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "clip_range": clip_range,
        "best_mean_reward": best_mean,
    }
    with open(os.path.join(trial_dir, "config.json"), "w") as f:
        json.dump(trial_cfg, f, indent=2, default=str)

    vec_env.close()
    eval_env.close()

    print(
        f"  Trial {trial.number:3d} | "
        f"best_mean_reward = {best_mean:+.1f}",
    )
    return best_mean


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n-trials", type=int, default=20,
        help="Number of Optuna trials",
    )
    parser.add_argument(
        "--steps", type=int, default=_DEFAULT_STEPS,
        help="Training steps per trial",
    )
    args = parser.parse_args()

    study = optuna.create_study(
        direction="maximize",
        study_name="h1_walking",
        storage=f"sqlite:///{SWEEP_DIR}/optuna.db",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    print(
        f"Starting sweep: {args.n_trials} trials, "
        f"{args.steps:,} steps each",
    )
    print(f"Results: {SWEEP_DIR}")
    print()

    study.optimize(
        lambda trial: objective(trial, args.steps),
        n_trials=args.n_trials,
        show_progress_bar=True,
    )

    # ── Report best ──────────────────────────────────────────────
    best = study.best_trial
    print("\n" + "=" * 60)
    print(f"Best trial: {best.number}")
    print(f"Best mean reward: {best.value:.1f}")
    print("Parameters:")
    for k, v in best.params.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.6f}")
        else:
            print(f"  {k}: {v}")

    # Save best config for full training.
    best_cfg_path = os.path.join(SWEEP_DIR, "best_config.json")
    with open(best_cfg_path, "w") as f:
        json.dump(best.params, f, indent=2)
    print(f"\nBest config saved: {best_cfg_path}")
    print("To train with these params, update train.py or pass them.")


if __name__ == "__main__":
    main()
