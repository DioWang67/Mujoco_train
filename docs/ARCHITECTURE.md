# Repository Architecture

This repository is a research/tooling codebase for multiple MuJoCo robot
training projects. Keep the structure pragmatic: shared mechanics should be
centralized, but reward logic, robot state, assets, and task-specific training
decisions should stay with the robot project.

## Current Project Model

Each trainable robot is registered by:

```text
configs/<slug>/project.json
configs/<slug>/train.json
```

`train.py` discovers these configs and dispatches to the configured
`train_module`:

```bash
python train.py --project h1 --smoke
python train.py --project grasp --smoke
python train.py --project sedon --smoke
```

`project.json` supports these fields:

```json
{
  "slug": "sedon",
  "display_name": "Sedon standing baseline",
  "train_module": "sedon_baseline.train",
  "eval_module": null,
  "job_name": "sedon",
  "smoke_args": ["--smoke", "--n-envs", "1"],
  "private_asset_dir": "private_assets/sedon"
}
```

- `smoke_args` is used by generic remote smoke verification.
- `private_asset_dir` is optional and must be repo-relative. It is ignored by
  Git and is included in release archives only when explicitly requested.

## Directory Responsibilities

```text
configs/          Committable JSON configuration only.
docs/             Runbooks, architecture notes, and review docs.
grasp_baseline/   Grasp env, train entrypoint, tests, and small committed assets.
sedon_baseline/   Sedon env, train entrypoint, and tests.
tools/            Reusable Python CLIs.
scripts/          Thin operator wrappers around Python tools or remote SSH flows.
tests/            Cross-project unit tests.
private_assets/   Ignored proprietary or large robot assets.
models/           Ignored training outputs.
logs/             Ignored logs and TensorBoard outputs.
reports/          Ignored/generated evaluation reports unless intentionally saved.
```

H1 now follows the same package pattern as newer robots:

```text
h1_baseline/
  env.py
  train.py
  eval.py
```

Do not create root-level robot files. New robot projects should use a package
directory like `sedon_baseline/`.

## What Belongs In Shared Code

Use shared modules for mechanical runtime concerns:

- project discovery
- artifact path resolution
- training run manifests
- config file parsing
- generic deployment/package tooling

Keep these project-specific:

- observation/action definitions
- reward functions
- reset logic
- termination logic
- curriculum stages
- robot-specific evaluation metrics
- proprietary asset conversion choices

## Asset Policy

`configs/` must not contain URDF, STL, export logs, or proprietary assets.
Those belong under ignored local storage:

```text
private_assets/<robot>/
```

For Sedon, the expected private layout is:

```text
private_assets/sedon/original_urdf_package/
private_assets/sedon/mjcf_source/
private_assets/sedon/training_scene.xml
```

Only commit reproducible tools and non-secret config. Do not commit company
robot geometry unless explicitly approved.

Private assets are not included in normal release archives. To deploy a robot
that requires ignored assets, use the explicit opt-in flag:

```bash
python -m tools.deploy_release --project-slug sedon --include-private-assets
```

The Windows wrapper exposes the same behavior:

```bat
scripts\sedon_deploy_remote.bat
```

## Tooling Policy

Prefer Python CLIs in `tools/`:

```bash
python -m tools
python -m tools.preflight_check
python -m tools.convert_urdf_to_mjcf
```

Use `scripts/` only for thin wrappers that help operators run common commands
on Windows or remote hosts. Avoid adding a new wrapper for every robot when the
generic command already works:

```bat
scripts\run_remote_train.bat <project-slug> [args...]
scripts\tensorboard_tunnel.bat <project> <job> <port>
```

## Refactor Direction

If the codebase grows again, the next sensible cleanup is splitting larger
H1-specific tools into the H1 package:

```text
h1_baseline/
  tools/
    compare_eval.py
    benchmark_matrix.py
    sweep.py
```

Keep that as a focused refactor because H1 still has the most evaluation,
benchmark, and documentation coupling.
