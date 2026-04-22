"""Run multi-seed BASE-vs-DR comparison and aggregate statistics.

Usage:
    python -m tools.aggregate_compare --seeds 5 --episodes 5
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from statistics import mean, stdev

from tools.compare_eval import run_compare


def _ci95(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return 1.96 * (stdev(values) / (len(values) ** 0.5))


def _summary(values: list[float]) -> dict:
    if not values:
        return {"mean": float("nan"), "std": float("nan"), "ci95": float("nan")}
    return {
        "mean": mean(values),
        "std": stdev(values) if len(values) > 1 else 0.0,
        "ci95": _ci95(values),
    }


def main(
    num_seeds: int,
    seed_start: int,
    episodes: int,
    vel: float,
    out_json: str | None,
    out_csv: str | None,
) -> int:
    runs = []
    for i in range(num_seeds):
        seed = seed_start + i
        print(f"\n=== Seed {seed} ===")
        payload = run_compare(episodes=episodes, vel=vel, seed=seed)
        if payload is None:
            print("No model found; abort.")
            return 1
        runs.append(payload)

    delta_reward = [r["delta"]["reward_mean"] for r in runs]
    delta_len = [r["delta"]["len_mean"] for r in runs]
    delta_vx = [r["delta"]["xvel_mean"] for r in runs]

    agg = {
        "num_seeds": num_seeds,
        "seed_start": seed_start,
        "episodes_per_seed": episodes,
        "target_vel": vel,
        "delta_reward": _summary(delta_reward),
        "delta_len": _summary(delta_len),
        "delta_xvel": _summary(delta_vx),
        "runs": runs,
    }

    print("\n" + "=" * 78)
    print("Aggregate DR-BASE deltas")
    print(
        f"Reward mean={agg['delta_reward']['mean']:+.2f} "
        f"+/-{agg['delta_reward']['ci95']:.2f} (95% CI)",
    )
    print(
        f"Len    mean={agg['delta_len']['mean']:+.2f} "
        f"+/-{agg['delta_len']['ci95']:.2f} (95% CI)",
    )
    print(
        f"Vx     mean={agg['delta_xvel']['mean']:+.3f} "
        f"+/-{agg['delta_xvel']['ci95']:.3f} (95% CI)",
    )

    if out_json:
        out_dir = os.path.dirname(out_json)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(agg, f, indent=2, ensure_ascii=False)
        print(f"[saved] json: {out_json}")

    if out_csv:
        out_dir = os.path.dirname(out_csv)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "mean", "std", "ci95"])
            writer.writerow(
                ["delta_reward", agg["delta_reward"]["mean"], agg["delta_reward"]["std"], agg["delta_reward"]["ci95"]],
            )
            writer.writerow(
                ["delta_len", agg["delta_len"]["mean"], agg["delta_len"]["std"], agg["delta_len"]["ci95"]],
            )
            writer.writerow(
                ["delta_xvel", agg["delta_xvel"]["mean"], agg["delta_xvel"]["std"], agg["delta_xvel"]["ci95"]],
            )
        print(f"[saved] csv : {out_csv}")

    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=3, help="number of seeds")
    p.add_argument("--seed-start", type=int, default=42)
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--vel", type=float, default=1.0)
    p.add_argument("--out-json", type=str, default="reports/aggregate_compare.json")
    p.add_argument("--out-csv", type=str, default="reports/aggregate_compare.csv")
    args = p.parse_args()
    raise SystemExit(
        main(
            num_seeds=args.seeds,
            seed_start=args.seed_start,
            episodes=args.episodes,
            vel=args.vel,
            out_json=args.out_json,
            out_csv=args.out_csv,
        ),
    )
