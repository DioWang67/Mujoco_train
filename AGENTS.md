# Project Agent Instructions

1. Project goal: Seedon MuJoCo / Duck-like gait diagnostics.
2. Do not repeat completed extraction, readiness, or foot prototype pipelines unless explicitly requested.
3. Do not modify `seedon_baseline/train.py` or `seedon_baseline/eval.py` without explicit instruction.
4. Do not delete or move artifacts.
5. Mark every assumption value with `source=assumption` and `confidence=low`.
6. Every new tool must pass `python -B -m py_compile`.
7. Do not claim walking success from diagnostics, scripted probes, contact checks, or PPO-adjacent reports.
8. Report `git status` after implementation or verification work.
