"""Validate compare reports against release gates.

Usage:
    python -m tools.gate_check --report reports/compare_report.json --gates configs/release_gates.json
    python -m tools.gate_check --report reports/aggregate_compare.json --mode aggregate
"""

import argparse
import json


def _check_single(report: dict, gates: dict) -> dict[str, bool]:
    return {
        "dr_len_mean_min": report["dr"]["len_mean"] >= gates["dr_len_mean_min"],
        "dr_reward_mean_min": report["dr"]["reward_mean"] >= gates["dr_reward_mean_min"],
        "dr_delta_len_mean_min": report["delta"]["len_mean"] >= gates["dr_delta_len_mean_min"],
        "dr_delta_reward_mean_min": report["delta"]["reward_mean"] >= gates["dr_delta_reward_mean_min"],
        "dr_xvel_mean_min": report["dr"]["xvel_mean"] >= gates["dr_xvel_mean_min"],
    }


def _check_aggregate(report: dict, gates: dict) -> dict[str, bool]:
    return {
        "agg_delta_len_mean_min": report["delta_len"]["mean"] >= gates["agg_delta_len_mean_min"],
        "agg_delta_reward_mean_min": report["delta_reward"]["mean"] >= gates["agg_delta_reward_mean_min"],
        "agg_delta_xvel_mean_min": report["delta_xvel"]["mean"] >= gates["agg_delta_xvel_mean_min"],
        "agg_delta_len_ci95_max": report["delta_len"]["ci95"] <= gates["agg_delta_len_ci95_max"],
        "agg_delta_reward_ci95_max": report["delta_reward"]["ci95"] <= gates["agg_delta_reward_ci95_max"],
    }


def _load_rules(gates_doc: dict, profile: str | None) -> dict:
    if profile:
        profiles = gates_doc.get("profiles", {})
        if profile not in profiles:
            raise ValueError(f"Profile '{profile}' not found in gates file.")
        return profiles[profile]
    return gates_doc["rules"]


def main(report_path: str, gates_path: str, mode: str, profile: str | None) -> int:
    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)
    with open(gates_path, "r", encoding="utf-8") as f:
        gates_doc = json.load(f)
    try:
        gates = _load_rules(gates_doc, profile)
    except ValueError as e:
        print(f"[error] {e}")
        return 1

    if mode == "auto":
        mode = "aggregate" if "num_seeds" in report else "single"

    checks = _check_aggregate(report, gates) if mode == "aggregate" else _check_single(report, gates)

    print("Gate check results")
    print(f"Mode: {mode}")
    print(f"Profile: {profile or 'default-rules'}")
    print("-" * 60)
    failed = []
    for k, ok in checks.items():
        status = "PASS" if ok else "FAIL"
        print(f"{status:>4} | {k}")
        if not ok:
            failed.append(k)

    if failed:
        print("-" * 60)
        print("FAILED gates:", ", ".join(failed))
        return 1

    print("-" * 60)
    print("All gates passed.")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--report", required=True, help="JSON from compare_eval --out-json")
    p.add_argument("--gates", default="configs/release_gates.json", help="Gate file path")
    p.add_argument("--mode", choices=["auto", "single", "aggregate"], default="auto")
    p.add_argument("--profile", default=None, help="optional gate profile name")
    args = p.parse_args()
    raise SystemExit(main(args.report, args.gates, args.mode, args.profile))
