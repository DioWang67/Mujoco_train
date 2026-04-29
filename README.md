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
```

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

Create a clean source archive from the current commit:

```bash
python -m tools.deploy_release --project-slug h1
```

Upload and switch the remote `current` release when SSH is configured:

```bash
python -m tools.deploy_release --project-slug h1 --remote-host root@10.6.243.55 --upload
```

For robots that require ignored private assets on the remote host, explicitly
include them:

```bash
python -m tools.deploy_release --project-slug sedon --include-private-assets
```

The operator wrapper for Sedon uses the same opt-in behavior:

```bat
scripts\sedon_deploy_remote.bat
```

## Current Cleanup Rules

- See `docs/ARCHITECTURE.md` for module boundaries and asset policy.
- Keep root entrypoints thin when possible.
- Put repeatable parameters in `configs/`, not hardcoded scripts.
- Keep private URDF/STL/CAD exports under `private_assets/`, not `configs/`.
- Keep generated outputs ignored unless they are intentional fixtures.
- Keep pure logic tests separate from simulator-dependent tests.
- Avoid adding abstractions unless a second real use case exists.

