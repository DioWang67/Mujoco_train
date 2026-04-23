# Remote Layout

Use a generic remote root so the training host can manage multiple robots and
projects without centering everything around H1.

## Recommended layout

```text
/root/anaconda3/mujoco-train-system/
  code/
    releases/
      05cdde4/
      0a458bf/
    current -> /root/anaconda3/mujoco-train-system/code/releases/0a458bf
  runs/
    h1/
      models/
      logs/
      reports/
    grasp/
      models/
      logs/
      reports/
  shared/
    offline/
      wheels/
      cuda_deps/
      missing_deps/
      mujoco_deps/
      cusparselt_fix/
      archives/
    incoming/
  scripts/
```

## Meaning

- `code/releases/`: immutable source snapshots extracted from a
  committed archive.
- `code/current`: symlink or agreed pointer to the active release.
- `runs/<slug>/`: training outputs for one robot / task only.
- `shared/offline/`: reusable wheels and dependency repair bundles.
- `shared/incoming/`: uploaded tarballs before extraction.
- `scripts/`: host-side helpers that are not specific to one robot.

## Current project mapping

For this repository, use:

```text
REMOTE_ROOT=/root/anaconda3/mujoco-train-system
PROJECT_SLUG=grasp
CODE_ROOT=/root/anaconda3/mujoco-train-system/code/current
RUN_ROOT=/root/anaconda3/mujoco-train-system/runs/grasp
```

That keeps the naming generic at the system level while still allowing each
robot project to have its own short slug.

## Deployment workflow

Create a clean archive from the current committed source:

```bash
python -m tools.deploy_release --project-slug h1
```

That writes a tarball into `artifacts/sync/` and prints the `scp` / `ssh`
commands needed to extract it into:

```text
/root/anaconda3/mujoco-train-system/code/releases/<commit>
```

If local SSH access is already configured, the same tool can upload and switch
`current` in one step:

```bash
python -m tools.deploy_release --project-slug h1 --remote-host root@10.6.243.55 --upload
```
