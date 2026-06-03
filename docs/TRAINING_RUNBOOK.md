# Training Runbook

Use this as the short operator guide. For project structure, see
`docs/PROJECT_GUIDE.md`. For Seedon-specific gait work, see
`docs/SEEDON_WORKFLOW.md`.

## Before Training

Run basic checks:

```bash
python -m tools.preflight_check
python -m pytest
```

If MuJoCo or private assets are unavailable, pure tests should still run while
MuJoCo-specific checks may fail or skip.

## H1

Smoke:

```bash
python train.py --project h1 --smoke
```

Train:

```bash
python train.py --project h1
```

Resume:

```bash
python train.py --project h1 --resume
```

Evaluate:

```bash
python eval.py --project h1 --episodes 1 --render
```

## Grasp

Smoke:

```bash
python train.py --project grasp --smoke
```

Train full phase:

```bash
python train.py --project grasp --phase full --n-envs 32
```

Evaluate:

```bash
python eval.py --project grasp --episodes 10
```

## Seedon

Build/check scene:

```bash
python -m tools.build_seedon_training_scene
python -m tools.smoke_seedon_env --steps 20
```

Smoke:

```bash
python train.py --project seedon --smoke --n-envs 1 --reset-noise-scale 0
```

Current Blue-style support gait local run:

```powershell
$env:SEEDON_CONFIG_OVERRIDES='configs\seedon\blue_dynamic_support_gait.json'
python train.py --project seedon --total-timesteps 2000000 --n-envs 4 --reset-noise-scale 0.005
```

Remote deploy, resume, and status:

```powershell
scripts\seedon_remote_deploy_and_check.bat
scripts\seedon_remote_resume_training.bat
scripts\seedon_remote_training_status.bat
```

Remote direct command from `code/current`:

```bash
export SEEDON_CONFIG_OVERRIDES=configs/seedon/blue_dynamic_support_gait.json
python -m seedon_baseline.train --total-timesteps 2000000 --n-envs 128 --reset-noise-scale 0.005
```

## Evaluation And Audit

H1 compare:

```bash
python -m tools.compare_eval --episodes 8 --vel 1.0 --out-json reports/compare_report.json --out-csv reports/compare_report.csv
python -m tools.gate_check --report reports/compare_report.json --gates configs/gate_profiles.json --profile preprod
```

Seedon viewer:

```bash
python -m tools.debug_seedon_gait_viewer --mode policy --steps 600 --speed 0.5 --out-csv artifacts/seedon_debug/policy_gait_view.csv
```

Seedon audit:

```bash
python -m tools.debug_seedon_gait_audit --mode policy --steps 600 --out-csv artifacts/seedon_debug/policy_gait_audit.csv
```

## Artifacts

Runtime outputs are intentionally ignored:

```text
models/
logs/
reports/
artifacts/
```

When evaluating a PPO model, keep model and VecNormalize files paired from the
same run. Mismatched VecNormalize stats can make a valid checkpoint look broken.

## Decision Rules

- If smoke fails, do not start a long run.
- If Seedon reward improves but gait viewer shows tiptoe, dragging, hopping, or
  base-proxy contact, treat the run as failed for gait quality.
- If Seedon still cannot unload or micro-lift after a long run, move to
  reference tracking or stronger phase terms instead of only adding timesteps.
- If remote `code/current` is not a Git repo, that is normal for release
  deployments. Inspect source files or release directory names instead.
