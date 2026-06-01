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

## Automatic Windows deployment

For day-to-day Sedon work on Windows, use the automated wrapper instead of
hand-running `scp` and `ssh`:

```bat
scripts\sedon_remote_deploy_and_check.bat
```

It uses `.env.remote` for host, root, password, and deployment options. The
default password backend is `askpass`, which drives Windows OpenSSH with
`SSH_ASKPASS`; the password remains in the child environment instead of being
placed on the command line.

Minimal `.env.remote` shape:

```env
REMOTE_HOST=root@10.6.243.55
REMOTE_ROOT=/root/anaconda3/mujoco-train-system
REMOTE_PROJECT_SLUG=sedon
REMOTE_VERIFY_PROJECT=sedon
REMOTE_SOURCE_MODE=working-tree
REMOTE_SSH_BACKEND=askpass
REMOTE_PASSWORD=<remote password>
REMOTE_INCLUDE_PRIVATE_ASSETS=0
REMOTE_INCLUDE_EXTRA_ASSETS=0
REMOTE_SMOKE_ARGS=--smoke
```

Keep `.env.remote` local only. It is ignored by git.

Normal deploys are intentionally small. They do not resend private assets or
`mujoco_menagerie`. If the remote needs refreshed private assets, run:

```bat
scripts\remote_auto_deploy.bat --include-private-assets
```

After one successful asset deploy, later releases symlink the existing remote
asset directories into the new release.

## Deploy content overlay

The automated deployer always checks:

```text
deploy_content/
```

Files there are copied into the release archive using paths relative to
`deploy_content`. This gives a stable payload area for future deployments
without changing deployment code.

Example:

```text
deploy_content/configs/sedon/foo.json
```

becomes:

```text
/root/anaconda3/mujoco-train-system/code/releases/<release>/configs/sedon/foo.json
```

`deploy_content/` is ignored by git except for its README and placeholder. Use
it for explicit deployment payloads; use normal source files and commits for
changes that should stay in the repository.
