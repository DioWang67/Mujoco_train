# H1 MuJoCo Robot Learning

This repository contains MuJoCo reinforcement-learning experiments for:

- H1 walking with PPO
- fixed-base grasping baseline
- Sedon standing baseline from a private URDF/MJCF conversion flow
- evaluation, comparison, benchmark, and release-gate tools
- remote training/deployment helpers

The project is intentionally kept as a research/tooling codebase. Keep changes
simple, testable, and focused on training correctness before adding larger
architecture.

## Repository Layout

```text
configs/          Training, benchmark, and release-gate configs
docs/             Runbooks, remote layout notes, and project status
h1_baseline/      H1 walking environment, training, evaluation, and tests
grasp_baseline/   Fixed-base grasp environment, training, assets, and tests
sedon_baseline/   Sedon standing environment, training, and tests
robot_learning/   Shared project discovery, config, paths, and runtime helpers
scripts/          Operator wrappers for Windows/Linux remote workflows
tests/            Lightweight unit tests that should run without MuJoCo
tools/            Evaluation, deployment, benchmark, and maintenance CLIs
train.py          Unified training entrypoint, defaults to H1
eval.py           Unified evaluation entrypoint, defaults to H1
```

## Start Here

The repo now has three short orientation documents:

- `docs/PROJECT_GUIDE.md`  
  Canonical project map, active entrypoints, tool maturity, cleanup policy.
- `docs/TRAINING_RUNBOOK.md`  
  Short operator commands for H1, grasp, Sedon, eval, and remote runs.
- `docs/SEDON_WORKFLOW.md`  
  Current Sedon reverse-knee/no-tiptoe workflow, remote command, viewer/audit
  checks, and troubleshooting.

For Sedon config selection, use `configs/sedon/README.md`.

Generated runtime outputs should stay out of Git unless they are deliberate
fixtures:

```text
models/
logs/
reports/
artifacts/
private_assets/
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

MuJoCo assets are expected under `mujoco_menagerie/`. That directory is ignored
because it is large; install or sync it separately.

## Common Commands

Run quick checks:

```bash
python -m pytest
python -m tools.preflight_check
python -m tools.project_inventory
```

Create a sanitized debug workspace for AI-assisted code inspection:

```bash
python -m tools.agent_workspace --name sedon_debug --force
```

The generated workspace lives under `artifacts/agent_workspace/` and excludes
`private_assets/`, complete XML/MJCF/URDF/mesh files, local env files, models,
logs, and generated outputs. Share that workspace or its manifest with AI tools
instead of exposing the full repository.

List available Python tools:

```bash
python -m tools
```

Train H1:

```bash
python train.py --project h1 --smoke
python train.py --project h1
python train.py --project h1 --resume
python train.py --project h1 --dr
python train.py --project h1 --finetune models/best_model.zip --dr
```

Train grasp:

```bash
python train.py --project grasp --smoke
python train.py --project grasp --phase full --n-envs 32
```

Prepare and train Sedon locally:

```bash
python -m tools.convert_urdf_to_mjcf
python -m tools.build_sedon_training_scene
python -m tools.smoke_sedon_env --steps 20
python train.py --project sedon --smoke --n-envs 1
```

Current Sedon reverse-knee/no-tiptoe experiment:

```powershell
$env:SEDON_CONFIG_OVERRIDES='configs\sedon\reverse_knee_no_tiptoe_walk.json'
python train.py --project sedon --total-timesteps 10000000 --n-envs 4 --reset-noise-scale 0.01
```

Evaluate Sedon:

```bash
python eval.py --project sedon --episodes 1 --render
python eval.py --project sedon --episodes 1 --record
scripts\sedon_eval.bat
```

Add a new robot by creating `configs/<slug>/project.json` and a train module
with `main(argv)`. The shared entrypoint will then accept:

```bash
python train.py --project <slug> [project args...]
python eval.py --project <slug> [project args...]
```

Example `project.json`:

```json
{
  "slug": "quadruped",
  "display_name": "Quadruped walking",
  "train_module": "robots.quadruped.train",
  "eval_module": "robots.quadruped.eval",
  "job_name": "quadruped",
  "smoke_args": ["--smoke", "--n-envs", "1"],
  "private_asset_dir": "private_assets/quadruped"
}
```

Evaluate and gate H1 results:

```bash
python -m h1_baseline.eval
python -m tools.compare_eval --episodes 8 --vel 1.0 --out-json reports/compare_report.json --out-csv reports/compare_report.csv
python -m tools.gate_check --report reports/compare_report.json --gates configs/gate_profiles.json --profile preprod
```

Evaluate grasp:

```bash
python -m tools.eval_grasp --episodes 10 --no-render
python -m tools.grasp_sanity_check
```

## Testing

Default tests are lightweight and avoid MuJoCo runtime dependencies:

```bash
python -m pytest
```

Tests that construct MuJoCo environments are marked separately:

```bash
python -m pytest -m mujoco
```

If `mujoco` is not installed, MuJoCo-marked tests are skipped.

## Remote Training

Remote deployment uses the generic layout described in
`docs/REMOTE_LAYOUT.md`.
For wrapper script details, see `scripts/README.md`; for Python tool details,
see `tools/README.md`.

Create a clean source archive from the current commit when you need a manual
artifact:

```bash
python -m tools.deploy_release --project-slug h1
```

For normal remote work, use the env-driven deployer. It packages the current
worktree plus optional `deploy_content/` overlay, uploads it, switches
`code/current`, and smoke-checks the target project:

```bat
scripts\remote_auto_deploy.bat
```

Sedon has a one-command wrapper:

```bat
scripts\sedon_remote_deploy_and_check.bat
```

Put one-off deployment payloads under `deploy_content/` using repo-relative
paths. Example: `deploy_content/configs/sedon/foo.json` deploys as
`configs/sedon/foo.json`.

Private assets are not resent during normal deploys. Refresh them only when the
robot asset files changed:

```bat
scripts\remote_auto_deploy.bat --include-private-assets
```

## Current Cleanup Rules

- See `docs/ARCHITECTURE.md` for module boundaries and asset policy.
- Keep root entrypoints thin when possible.
- Put repeatable parameters in `configs/`, not hardcoded scripts.
- Keep private URDF/STL/CAD exports under `private_assets/`, not `configs/`.
- Keep generated outputs ignored unless they are intentional fixtures.
- Keep pure logic tests separate from simulator-dependent tests.
- Avoid adding abstractions unless a second real use case exists.

