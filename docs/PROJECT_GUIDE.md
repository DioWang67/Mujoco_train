# Project Guide

This repository is a research and tooling workspace for MuJoCo robot learning.
It is not a single production application. Treat it as three robot projects
sharing common training, packaging, and evaluation infrastructure.

## Canonical Entrypoints

Use these first. Avoid running deep modules directly unless a runbook says so.

```bash
python train.py --project h1 [args...]
python train.py --project grasp [args...]
python train.py --project sedon [args...]

python eval.py --project h1 [args...]
python eval.py --project grasp [args...]
python eval.py --project sedon [args...]

python -m tools
```

For Sedon on the remote release layout, direct module execution is acceptable
when debugging dispatcher or path issues:

```bash
python -m sedon_baseline.train [args...]
```

## Repository Map

| Path | Purpose | Keep / Ignore |
|---|---|---|
| `configs/` | Static JSON configs for training, sweeps, gates, and project discovery. | Keep. |
| `docs/` | Human runbooks and architecture notes. | Keep concise and current. |
| `robot_learning/` | Shared project discovery, path resolution, config loading, and runtime helpers. | Keep small. |
| `h1_baseline/` | H1 environment, train, eval, and tests. | Active. |
| `grasp_baseline/` | Fixed-base grasp environment, train, eval, and tests. | Active. |
| `sedon_baseline/` | Sedon environment, train, eval, and tests. | Active. |
| `tools/` | Python CLIs for debug, eval, packaging, and experiments. | Active, but mixed maturity. |
| `scripts/` | Operator wrappers for local and remote work. | Active. Prefer thin wrappers. |
| `tests/` | Lightweight tests that should run without private assets where possible. | Active. |
| `private_assets/` | Ignored private robot assets. | Do not commit. |
| `models/`, `logs/`, `reports/`, `artifacts/` | Runtime outputs. | Do not commit unless intentional fixture. |
| `_verify_*` | Temporary verification output directories. | Treat as disposable. |

## Current Active Workflows

### H1

Use H1 for the older walking baseline and DR/evaluation toolchain.

```bash
python train.py --project h1 --smoke
python train.py --project h1
python eval.py --project h1 --episodes 1 --render
```

### Grasp

Use grasp for fixed-base manipulation sanity checks.

```bash
python train.py --project grasp --smoke
python train.py --project grasp --phase full --n-envs 32
python eval.py --project grasp --episodes 10
```

### Sedon

Sedon is the current locomotion focus. The canonical runbook is:

```text
scene build -> smoke -> train -> gait viewer/audit -> reward/config iteration
```

Start with:

```bash
python -m tools.build_sedon_training_scene
python -m tools.smoke_sedon_env --steps 20
python train.py --project sedon --smoke --n-envs 1
```

For the current reverse-knee/no-tiptoe experiment, see:

```text
docs/SEDON_WORKFLOW.md
configs/sedon/README.md
```

## Tool Maturity

`tools/` is intentionally broad. Do not assume every tool is part of the
standard workflow.

| Tier | Meaning | Examples |
|---|---|---|
| Standard | Safe first-choice operator commands. | `smoke_sedon_env`, `sedon_eval`, `debug_sedon_gait_viewer`, `debug_sedon_gait_audit` |
| Diagnostic | Useful for answering one specific mechanical or reward question. | `debug_sedon_*`, `preview_sedon_*`, `trace_*` |
| Sweep / Experimental | Generates comparison data; may be slow or create output dirs. | `sweep_sedon_*`, `sedon_gait_sweep` |
| Packaging | Release and deployment helpers. | `deploy_release`, `prepare_package` |

Before adding a new tool, prefer extending an existing one if the question is
the same. Add a new tool only when the output shape or experiment is genuinely
different.

## Remote Release Layout

Remote hosts use an extracted release directory, not a normal Git worktree:

```text
/root/anaconda3/mujoco-train-system/
  code/releases/<commit>/
  code/current -> releases/<commit>
  runs/<project>/
```

That means `git rev-parse` may fail under `code/current`; this is expected.
Check source content directly, or inspect the release directory name.

Deploy from local Windows:

```powershell
scripts\sedon_remote_deploy_and_check.bat
```

For one-off payloads, place files in `deploy_content/` using repo-relative
paths before running the deploy wrapper.

## Cleanup Policy

No deletion should happen just because a file looks old. Use this order:

1. Mark the canonical command in docs.
2. Move rarely used commands into "diagnostic" wording.
3. Confirm no script, README, registry entry, or runbook references the file.
4. Only then remove or archive in a separate cleanup commit.

Use the inventory command before cleanup:

```bash
python -m tools.project_inventory
```

It reports canonical directories, tool/script counts, and disposable local
runtime directories such as caches, `models/`, `logs/`, `reports/`,
`artifacts/`, and `_verify_*`. It does not delete anything.

Current known local leftovers:

- `.gitignore` has uncommitted local changes.
- `sweep_train.py` is untracked.

Those were not touched by the Sedon gait commit and should be reviewed
separately.
