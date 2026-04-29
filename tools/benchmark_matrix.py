"""Run benchmark matrix scenarios and export a unified report.

Usage:
    python -m tools.benchmark_matrix
    python -m tools.benchmark_matrix --matrix configs/benchmark_matrix.json --out-json reports/benchmark_report.json
"""

import argparse
import csv
import json
import os
import sys
from statistics import mean, stdev
from pathlib import Path

import numpy as np
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from h1_baseline.env import H1Env

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


def _resolve_model() -> Path | None:
    # DR-trained models handle both base and DR scenarios best, so prefer them.
    for p in [DR_BEST_PATH, DR_FINAL_PATH, BEST_PATH, FINAL_PATH]:
        if p.exists():
            return p
    return None


def _load_vecnorm(candidates: list[Path]):
    for c in candidates:
        if c.exists():
            dummy = DummyVecEnv([lambda: H1Env()])
            vn = VecNormalize.load(str(c), dummy)
            vn.training = False
            vn.norm_reward = False
            return vn, str(c)
    return None, None


def _vecnorm_candidates_for(model_path: Path, dr: bool) -> list[Path]:
    """Pick VecNorm files that best match the chosen model artifact."""
    if dr:
        if model_path == DR_BEST_PATH:
            return [VECNORM_DR_BEST_PATH, VECNORM_DR_PATH, VECNORM_PATH]
        if model_path == DR_FINAL_PATH:
            return [VECNORM_DR_PATH, VECNORM_DR_BEST_PATH, VECNORM_PATH]
        if model_path == FINAL_PATH:
            return [VECNORM_PATH, VECNORM_BEST_PATH]
        return [VECNORM_BEST_PATH, VECNORM_PATH]

    if model_path == BEST_PATH:
        return [VECNORM_BEST_PATH, VECNORM_PATH]
    if model_path == FINAL_PATH:
        return [VECNORM_PATH, VECNORM_BEST_PATH]
    if model_path == DR_BEST_PATH:
        return [VECNORM_DR_BEST_PATH, VECNORM_DR_PATH, VECNORM_PATH]
    return [VECNORM_DR_PATH, VECNORM_DR_BEST_PATH, VECNORM_PATH]


def _run_scenario(model: PPO, model_path: Path, sc: dict) -> dict:
    dr = bool(sc.get("dr", False))
    target_vel = float(sc.get("target_vel", 1.0))
    episodes = int(sc.get("episodes", 5))
    base_seed = sc.get("seed", None)
    dr_level = sc.get("dr_level", None)

    vec_norm, vecnorm_path = _load_vecnorm(
        _vecnorm_candidates_for(model_path, dr=dr),
    )
    env = TimeLimit(
        H1Env(
            domain_randomization=dr,
            target_velocity=target_vel,
            render_mode=None,
        ),
        max_episode_steps=1000,
    )
    if dr and dr_level is not None:
        env.set_dr_level(float(np.clip(dr_level, 0.0, 1.0)))

    rewards, lengths, xvels = [], [], []
    try:
        for ep in range(episodes):
            seed = None if base_seed is None else int(base_seed) + ep
            obs, _ = env.reset(seed=seed)
            done = False
            total_r, steps = 0.0, 0
            xv_buf = []
            while not done:
                obs_in = vec_norm.normalize_obs(obs.reshape(1, -1))[0] if vec_norm else obs
                action, _ = model.predict(obs_in, deterministic=True)
                obs, r, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                total_r += float(r)
                steps += 1
                xv_buf.append(float(info.get("x_velocity", 0.0)))
            rewards.append(total_r)
            lengths.append(steps)
            xvels.append(float(np.mean(xv_buf)) if xv_buf else 0.0)
    finally:
        env.close()

    def _ms(values: list[float]) -> tuple[float, float]:
        return float(mean(values)), float(stdev(values) if len(values) > 1 else 0.0)

    reward_mean, reward_std = _ms(rewards)
    len_mean, len_std = _ms(lengths)
    xvel_mean, xvel_std = _ms(xvels)

    return {
        "name": sc.get("name", "unnamed"),
        "dr": dr,
        "target_vel": target_vel,
        "dr_level": dr_level,
        "episodes": episodes,
        "vecnorm": vecnorm_path,
        "reward_mean": reward_mean,
        "reward_std": reward_std,
        "len_mean": len_mean,
        "len_std": len_std,
        "xvel_mean": xvel_mean,
        "xvel_std": xvel_std,
    }


def main(matrix_path: str, out_json: str | None, out_csv: str | None) -> int:
    with open(matrix_path, "r", encoding="utf-8") as f:
        matrix = json.load(f)

    scenarios = matrix.get("scenarios", [])
    if not scenarios:
        print("No scenarios found in matrix.")
        return 1

    model_path = _resolve_model()
    if model_path is None:
        print("No model found in models/. Train first.")
        return 1

    print(f"Model: {model_path}")
    model = PPO.load(model_path, custom_objects={"learning_rate": 3e-4, "clip_range": 0.2})

    rows = []
    for sc in scenarios:
        print(f"\n=== Scenario: {sc.get('name', 'unnamed')} ===")
        row = _run_scenario(model, model_path, sc)
        rows.append(row)
        print(
            f"R={row['reward_mean']:.1f}±{row['reward_std']:.1f} "
            f"Len={row['len_mean']:.1f}±{row['len_std']:.1f} "
            f"Vx={row['xvel_mean']:.2f}±{row['xvel_std']:.2f}",
        )

    payload = {
        "matrix": matrix.get("description", ""),
        "matrix_path": matrix_path,
        "model": str(model_path),
        "rows": rows,
    }

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
                fieldnames=[
                    "name", "dr", "target_vel", "dr_level", "episodes", "vecnorm",
                    "reward_mean", "reward_std", "len_mean", "len_std", "xvel_mean", "xvel_std",
                ],
            )
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        print(f"[saved] csv : {out_csv}")

    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--matrix", default="configs/benchmark_matrix.json")
    p.add_argument("--out-json", default="reports/benchmark_report.json")
    p.add_argument("--out-csv", default="reports/benchmark_report.csv")
    args = p.parse_args()
    raise SystemExit(main(args.matrix, args.out_json, args.out_csv))
