"""Compare BASE vs DR evaluation numerically.

Usage:
    python -m tools.compare_eval
    python -m tools.compare_eval --episodes 10 --vel 1.0
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from h1_env import H1Env

MODEL_DIR = REPO_ROOT / "models"
BEST_PATH = MODEL_DIR / "best_model.zip"
FINAL_PATH = MODEL_DIR / "h1_ppo.zip"
DR_FINAL_PATH = MODEL_DIR / "h1_ppo_dr.zip"
# DR best artifacts live in a subdirectory to avoid overwriting base best.
DR_BEST_DIR = MODEL_DIR / "dr_best"
DR_BEST_PATH = DR_BEST_DIR / "best_model.zip"
VECNORM_BEST_PATH = MODEL_DIR / "h1_vecnorm_best.pkl"
VECNORM_DR_BEST_PATH = DR_BEST_DIR / "h1_vecnorm_best.pkl"
VECNORM_DR_PATH = MODEL_DIR / "h1_vecnorm_dr.pkl"
VECNORM_PATH = MODEL_DIR / "h1_vecnorm.pkl"


def _load_vecnorm(candidates: list[Path]):
    for c in candidates:
        if c.exists():
            dummy = DummyVecEnv([lambda: H1Env()])
            vn = VecNormalize.load(str(c), dummy)
            vn.training = False
            vn.norm_reward = False
            return vn, str(c)
    return None, None


def _resolve_model() -> Path | None:
    """Pick one policy artifact to compare under base vs DR environments."""
    for model_path in [DR_BEST_PATH, DR_FINAL_PATH, BEST_PATH, FINAL_PATH]:
        if model_path.exists():
            return model_path
    return None


def _vecnorm_candidates_for(model_path: Path) -> list[Path]:
    """Use the VecNorm that matches the chosen model artifact."""
    if model_path == DR_BEST_PATH:
        return [VECNORM_DR_BEST_PATH, VECNORM_DR_PATH, VECNORM_PATH]
    if model_path == DR_FINAL_PATH:
        return [VECNORM_DR_PATH, VECNORM_DR_BEST_PATH, VECNORM_PATH]
    if model_path == BEST_PATH:
        return [VECNORM_BEST_PATH, VECNORM_PATH]
    return [VECNORM_PATH, VECNORM_BEST_PATH]


def _run_setting(model: PPO, episodes: int, dr: bool, vel: float, vec_norm, seed: int | None):
    env = TimeLimit(
        H1Env(domain_randomization=dr, target_velocity=vel, render_mode=None),
        max_episode_steps=1000,
    )
    ep_rewards = []
    ep_lens = []
    ep_xvel = []

    try:
        for ep in range(episodes):
            reset_seed = None if seed is None else seed + ep
            obs, _ = env.reset(seed=reset_seed)
            done = False
            total_r = 0.0
            steps = 0
            xv_buf = []
            while not done:
                obs_in = vec_norm.normalize_obs(obs.reshape(1, -1))[0] if vec_norm else obs
                action, _ = model.predict(obs_in, deterministic=True)
                obs, r, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                total_r += float(r)
                steps += 1
                xv_buf.append(float(info.get("x_velocity", 0.0)))
            ep_rewards.append(total_r)
            ep_lens.append(steps)
            ep_xvel.append(float(np.mean(xv_buf)) if xv_buf else 0.0)
    finally:
        env.close()

    return {
        "reward_mean": float(np.mean(ep_rewards)) if ep_rewards else float("nan"),
        "reward_std": float(np.std(ep_rewards)) if ep_rewards else float("nan"),
        "len_mean": float(np.mean(ep_lens)) if ep_lens else float("nan"),
        "len_std": float(np.std(ep_lens)) if ep_lens else float("nan"),
        "xvel_mean": float(np.mean(ep_xvel)) if ep_xvel else float("nan"),
        "xvel_std": float(np.std(ep_xvel)) if ep_xvel else float("nan"),
    }


def _print_row(name: str, m: dict):
    print(
        f"{name:<6} "
        f"R={m['reward_mean']:>8.1f}±{m['reward_std']:<6.1f} "
        f"Len={m['len_mean']:>6.1f}±{m['len_std']:<6.1f} "
        f"Vx={m['xvel_mean']:>5.2f}±{m['xvel_std']:<5.2f}",
    )


def _write_outputs(out_json: str | None, out_csv: str | None, payload: dict) -> None:
    if out_json:
        out_dir = os.path.dirname(out_json)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"[saved] json: {out_json}")
    if out_csv:
        out_dir = os.path.dirname(out_csv)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["setting", "reward_mean", "reward_std", "len_mean", "len_std", "xvel_mean", "xvel_std"],
            )
            writer.writeheader()
            writer.writerow({"setting": "BASE", **payload["base"]})
            writer.writerow({"setting": "DR", **payload["dr"]})
        print(f"[saved] csv : {out_csv}")


def run_compare(episodes: int, vel: float, seed: int | None = None) -> dict | None:
    model_path = _resolve_model()
    if model_path is None:
        print("No model found under models/. Train first.")
        return None

    print(f"Model: {model_path}")
    model = PPO.load(str(model_path), custom_objects={"learning_rate": 3e-4, "clip_range": 0.2})

    vecnorm_candidates = _vecnorm_candidates_for(model_path)
    vec_norm, vecnorm_path = _load_vecnorm(vecnorm_candidates)

    print(f"VecNorm(model): {vecnorm_path or 'None'}")
    print(f"Episodes={episodes}, target_vel={vel}")
    print("-" * 78)

    base_m = _run_setting(model, episodes, dr=False, vel=vel, vec_norm=vec_norm, seed=seed)
    dr_m = _run_setting(model, episodes, dr=True, vel=vel, vec_norm=vec_norm, seed=seed)

    _print_row("BASE", base_m)
    _print_row("DR", dr_m)

    print("-" * 78)
    print(
        f"Delta DR-BASE: "
        f"Reward={dr_m['reward_mean'] - base_m['reward_mean']:+.1f}, "
        f"Len={dr_m['len_mean'] - base_m['len_mean']:+.1f}, "
        f"Vx={dr_m['xvel_mean'] - base_m['xvel_mean']:+.2f}",
    )

    payload = {
        "model": str(model_path),
        "vecnorm": vecnorm_path,
        "episodes": episodes,
        "target_vel": vel,
        "seed": seed,
        "base": base_m,
        "dr": dr_m,
        "delta": {
            "reward_mean": dr_m["reward_mean"] - base_m["reward_mean"],
            "len_mean": dr_m["len_mean"] - base_m["len_mean"],
            "xvel_mean": dr_m["xvel_mean"] - base_m["xvel_mean"],
        },
    }
    return payload


def main(episodes: int, vel: float, out_json: str | None, out_csv: str | None, seed: int | None):
    payload = run_compare(episodes=episodes, vel=vel, seed=seed)
    if payload is None:
        return
    _write_outputs(out_json, out_csv, payload)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--vel", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=None, help="base seed for deterministic resets")
    p.add_argument("--out-json", type=str, default=None, help="save summary report as JSON")
    p.add_argument("--out-csv", type=str, default=None, help="save summary report as CSV")
    args = p.parse_args()
    main(args.episodes, args.vel, args.out_json, args.out_csv, args.seed)
