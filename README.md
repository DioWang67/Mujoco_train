# H1 MuJoCo Robot Learning

This repository contains MuJoCo reinforcement-learning experiments for:

- H1 walking with PPO
- fixed-base grasping baseline
- evaluation, comparison, benchmark, and release-gate tools
- remote training/deployment helpers

The project is intentionally kept as a research/tooling codebase. Keep changes
simple, testable, and focused on training correctness before adding larger
architecture.

## Repository Layout

```text
configs/          Training, benchmark, and release-gate configs
docs/             Runbooks, remote layout notes, and project status
grasp_baseline/   Fixed-base grasp environment, training, assets, and tests
scripts/          Operator wrappers for Windows/Linux remote workflows
tests/            Lightweight unit tests that should run without MuJoCo
tools/            Evaluation, deployment, benchmark, and maintenance CLIs
train.py          Unified training entrypoint, defaults to H1
h1_train.py       H1-specific PPO training implementation
eval.py           H1 evaluation entrypoint
h1_env.py         H1 MuJoCo walking environment
```

Generated runtime outputs should stay out of Git unless they are deliberate
fixtures:

```text
models/
logs/
reports/
artifacts/
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

Add a new robot by creating `configs/<slug>/project.json` and a train module
with `main(argv)`. The shared entrypoint will then accept:

```bash
python train.py --project <slug> [project args...]
```

Example `project.json`:

```json
{
  "slug": "quadruped",
  "display_name": "Quadruped walking",
  "train_module": "robots.quadruped.train",
  "eval_module": "robots.quadruped.eval",
  "job_name": "quadruped"
}
```

Evaluate and gate H1 results:

```bash
python eval.py
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

## Current Cleanup Rules

- Keep root entrypoints thin when possible.
- Put repeatable parameters in `configs/`, not hardcoded scripts.
- Keep generated outputs ignored unless they are intentional fixtures.
- Keep pure logic tests separate from simulator-dependent tests.
- Avoid adding abstractions unless a second real use case exists.
