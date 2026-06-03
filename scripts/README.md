# Scripts Index

These files are thin operator wrappers around Python tools and remote SSH
workflows. Prefer `python -m tools...` for automation and use these scripts for
manual Windows/Linux operation.

## Local Training

- `fresh_train.bat`
  Start fresh H1 training.
- `resume_train.bat`
  Resume H1 training from the latest checkpoint.
- `dr_train.bat`
  Start H1 training with domain randomization.
- `finetune_dr.bat`
  Fine-tune H1 with DR from `models/best_model.zip`.

## Local Evaluation

- `eval.bat`
  Interactive H1, grasp, and Seedon evaluation menu.
- `seedon_eval.bat`
  Interactive Seedon eval menu; accepts extra args for `eval.py --project seedon`.
- `tensorboard.bat`
  Start local TensorBoard against `logs/tb`.

## Remote Training

- `h1_remote_train.bat`
  Start formal remote H1 training.
- `grasp_remote_train.bat`
  Start formal remote grasp training with `--phase full --n-envs 8`.
- `seedon_remote_train.bat`
  Start formal remote Seedon standing training with `--n-envs 4`.
- `run_remote_train.bat <project-slug> [args...]`
  Start foreground remote training for any configured project.
- `stop_remote_train.bat [h1|grasp]`
  Stop remote training processes for the selected target.

## Remote Evaluation

- `grasp_remote_eval.bat`
  Evaluate the remote grasp checkpoint headlessly for 10 episodes.

## Remote TensorBoard

- `h1_tensorboard.bat`
  Open the remote H1 TensorBoard tunnel on port 6006.
- `grasp_tensorboard.bat`
  Open the latest remote grasp TensorBoard run on port 6007.
- `seedon_tensorboard.bat`
  Open the remote Seedon TensorBoard tunnel on port 6008.
- `tensorboard_tunnel.bat [project] [job] [port]`
  Start remote TensorBoard and open an SSH tunnel through PowerShell. If no
  project/job is provided, the script discovers available remote runs.
- `start_remote_tensorboard.ps1`
  PowerShell implementation with project/job discovery and automatic port
  selection. Use `-LatestRun` to open the newest child run under a job.
- `start_remote_tensorboard.sh`
  Remote-side launcher used after deployment. It searches for a Python
  executable with TensorBoard installed.

## Release / Packaging

- `install_vscode_server_offline.bat`
  Upload and install the local VS Code Server archive on an offline remote host.
- `h1_deploy_remote.bat`
  Deploy and smoke-check H1 through `remote_auto_deploy.bat`.
- `grasp_deploy_remote.bat`
  Deploy and smoke-check grasp through `remote_auto_deploy.bat`.
- `deploy_release.bat`
  Wrapper for `python -m tools.deploy_release`.
- `remote_auto_deploy.bat`
  Build from the current worktree by default, upload, activate, and smoke-check
  using `.env.remote` or process environment variables. This is the preferred
  non-interactive path when using `REMOTE_PASSWORD`.
- `seedon_remote_deploy_and_check.bat`
  One-command Seedon deploy and smoke check. Uses `.env.remote`; no arguments
  needed for the normal workflow.
- `seedon_remote_check.bat`
  One-command Seedon remote health/smoke check without uploading a new release.
- `seedon_remote_resume_training.bat`
  Thin Seedon wrapper over the generic `tools.remote_training` module. Starts
  background training from the latest available checkpoint and VecNormalize
  stats.
- `seedon_remote_training_status.bat`
  Thin Seedon wrapper over the generic `tools.remote_training` module. Shows
  training processes and tails the latest project log.
- `prepare_package.bat`
  Wrapper for `python -m tools.prepare_package`.
- `check_remote.sh`
  Small remote shell sanity check.

## Configuration

Most remote scripts accept these environment variables before falling back to
defaults:

```bat
set REMOTE_HOST=root@10.6.243.55
set REMOTE_ROOT=/root/anaconda3/mujoco-train-system
```

For password-based non-interactive deploys, copy `.env.remote.example` to
`.env.remote` and set `REMOTE_PASSWORD`. `.env.remote` is ignored by git.
On Windows, use `REMOTE_SSH_BACKEND=askpass` so the tool can drive
`C:\Windows\System32\OpenSSH\ssh.exe` and `scp.exe` while reading the password
from `.env.remote`. SSH-key mode can use `REMOTE_SSH_BACKEND=openssh` with no
password.

```bat
scripts\remote_auto_deploy.bat
```

Use `--source-mode git-ref` when deploying only committed content. The default
`working-tree` mode includes tracked and untracked non-ignored files, which is
better for fast Seedon experiment iteration.

Keep `REMOTE_INCLUDE_PRIVATE_ASSETS=0` after the first successful asset deploy;
including Seedon private assets makes each archive several hundred MB.

For repeatable ad-hoc content deploys, put files under `deploy_content/` using
repo-relative paths, then run `scripts\seedon_remote_deploy_and_check.bat`.
Example: `deploy_content/configs/seedon/foo.json` deploys to
`configs/seedon/foo.json` in the remote release.

The normal operator path is:

```bat
scripts\seedon_remote_deploy_and_check.bat
scripts\seedon_remote_resume_training.bat
scripts\seedon_remote_training_status.bat
```

`seedon_remote_resume_training.bat` starts training in the background through
the generic `tools.remote_training` module. It uses these optional `.env.remote`
values:

```env
SEEDON_RESUME_CONFIG=configs/seedon/blue_dynamic_support_gait.json
SEEDON_RESUME_TOTAL_TIMESTEPS=2000000
SEEDON_RESUME_N_ENVS=128
SEEDON_RESUME_RESET_NOISE_SCALE=0.005
```

For Seedon release directories, `code/current` is not a Git worktree. If the
generic dispatcher is confusing a debug session, run the Seedon module directly
from the remote release:

```bash
export SEEDON_CONFIG_OVERRIDES=configs/seedon/reverse_knee_no_tiptoe_walk.json
python -m seedon_baseline.train --total-timesteps 10000000 --n-envs 128 --reset-noise-scale 0.01
```

PowerShell scripts also expose equivalent parameters such as `-RemoteHost` and
`-RemoteRoot`.

## Adding a New Robot

Add `configs/<slug>/project.json` with a `train_module`, then run:

```bat
scripts\run_remote_train.bat <slug> --smoke
```

The train module must be importable from the deployed code root and expose:

```python
def main(argv: list[str] | None = None) -> int | None:
    ...
```

Use the generic `run_remote_train.bat` entrypoint for new robots.
Do not add a new per-robot wrapper unless it captures a real operator workflow
that the generic script cannot express.

If a robot needs ignored assets on the remote host, define `private_asset_dir`
in `configs/<slug>/project.json` and deploy with:

```bat
scripts\remote_auto_deploy.bat --project-slug <slug> --verify-project <slug> --include-private-assets
```
