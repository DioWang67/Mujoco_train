# Remote Layout

Use a generic remote root so the training host can manage multiple robots and
projects without centering everything around H1.

## Recommended layout

```text
/root/mujoco-train-system/
  projects/
    h1/
      releases/
        05cdde4/
        0a458bf/
      current -> /root/mujoco-train-system/projects/h1/releases/0a458bf
      runs/
        models/
        logs/
        reports/
    quadruped_grasper/
      releases/
      current
      runs/
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

- `projects/<slug>/releases/`: immutable source snapshots extracted from a
  committed archive.
- `projects/<slug>/current`: symlink or agreed pointer to the active release.
- `projects/<slug>/runs/`: training outputs for that project only.
- `shared/offline/`: reusable wheels and dependency repair bundles.
- `shared/incoming/`: uploaded tarballs before extraction.
- `scripts/`: host-side helpers that are not specific to one robot.

## Current project mapping

For this repository, use:

```text
REMOTE_ROOT=/root/mujoco-train-system
PROJECT_SLUG=h1
PROJECT_ROOT=/root/mujoco-train-system/projects/h1/current
```

That keeps the naming generic at the system level while still allowing each
robot project to have its own short slug.
