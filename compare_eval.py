"""Compare BASE vs DR evaluation numerically.

Usage:
    python compare_eval.py
    python compare_eval.py --episodes 10 --vel 1.0
"""

import argparse
import csv
import json
import os
import sys

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from h1_env import H1Env

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "models")
BEST_PATH = os.path.join(MODEL_DIR, "best_model")
FINAL_PATH = os.path.join(MODEL_DIR, "h1_ppo")
DR_FINAL_PATH = os.path.join(MODEL_DIR, "h1_ppo_dr")
# DR best artifacts live in a subdirectory to avoid overwriting base best.
DR_BEST_DIR = os.path.join(MODEL_DIR, "dr_best")
DR_BEST_PATH = os.path.join(DR_BEST_DIR, "best_model")
VECNORM_BEST_PATH = os.path.join(MODEL_DIR, "h1_vecnorm_best.pkl")
VECNORM_DR_BEST_PATH = os.path.join(DR_BEST_DIR, "h1_vecnorm_best.pkl")
VECNORM_DR_PATH = os.path.join(MODEL_DIR, "h1_vecnorm_dr.pkl")
VECNORM_PATH = os.path.join(MODEL_DIR, "h1_vecnorm.pkl")


def _resolve_model(dr: bool) -> str | None:
    if dr:
        for p in [DR_BEST_PATH, DR_FINAL_PATH, FINAL_PATH, BEST_PATH]:
            if os.path.exists(p + ".zip"):
                return p
    else:
        for p in [BEST_PATH, FINAL_PATH, DR_BEST_PATH, DR_FINAL_PATH]:
            if os.path.exists(p + ".zip"):
                return p
    return None


def _load_vecnorm(dr: bool):
    candidates = (
        [VECNORM_DR_BEST_PATH, VECNORM_DR_PATH, VECNORM_PATH]
        if dr
        else [VECNORM_BEST_PATH, VECNORM_PATH]
    )
    for c in candidates:
        if os.path.exists(c):
            dummy = DummyVecEnv([lambda: H1Env()])
            vn = VecNormalize.load(c, dummy)
            vn.training = False
            vn.norm_reward = False
            return vn, c
    return None, None


def _run_setting(model: PPO, episodes: int, dr: bool, vel: float, vec_norm, seed: int | None):
    env = H1Env(domain_randomization=dr, target_velocity=vel, render_mode=None)
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
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"[saved] json: {out_json}")
    if out_csv:
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
    model_path = _resolve_model(dr=True) or _resolve_model(dr=False)
    if model_path is None:
        print("No model found under models/. Train first.")
        return None

    print(f"Model: {model_path}.zip")
    model = PPO.load(model_path, custom_objects={"learning_rate": 3e-4, "clip_range": 0.2})

    vn_base, base_vn_path = _load_vecnorm(dr=False)
    vn_dr, dr_vn_path = _load_vecnorm(dr=True)

    print(f"VecNorm(BASE): {base_vn_path or 'None'}")
    print(f"VecNorm(DR)  : {dr_vn_path or 'None'}")
    print(f"Episodes={episodes}, target_vel={vel}")
    print("-" * 78)

    base_m = _run_setting(model, episodes, dr=False, vel=vel, vec_norm=vn_base, seed=seed)
    dr_m = _run_setting(model, episodes, dr=True, vel=vel, vec_norm=vn_dr, seed=seed)

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
        "model": model_path + ".zip",
        "vecnorm_base": base_vn_path,
        "vecnorm_dr": dr_vn_path,
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
