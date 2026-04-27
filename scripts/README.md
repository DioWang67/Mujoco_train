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
  Interactive H1 evaluation, compare, gate, aggregate, and benchmark menu.
- `tensorboard.bat`  
  Start local TensorBoard against `logs/tb`.

## Remote Training

- `run_remote_train.bat <project-slug> [args...]`  
  Start foreground remote training for any configured project.
- `run_remote_h1_train.bat [args...]`  
  Start foreground H1 remote training.
- `run_remote_grasp_train.bat [args...]`  
  Start foreground grasp remote training.
- `stop_remote_train.bat [h1|grasp]`  
  Stop remote training processes for the selected target.

## Remote TensorBoard

- `tensorboard_tunnel.bat [project] [job] [port]`  
  Start remote TensorBoard and open an SSH tunnel through PowerShell. If no
  project/job is provided, the script discovers available remote runs.
- `start_remote_tensorboard.ps1`  
  PowerShell implementation with project/job discovery and automatic port
  selection.
- `start_remote_tensorboard.sh`  
  Remote-side launcher used after deployment. It searches for a Python
  executable with TensorBoard installed.

## Release / Packaging

- `deploy_release.bat`  
  Wrapper for `python -m tools.deploy_release`.
- `deploy_remote_release.bat`  
  Wrapper for `deploy_remote_release.ps1`.
- `deploy_remote_release.ps1`  
  Build, upload, extract, activate, and optionally smoke-test a remote release.
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

The target-specific H1/grasp wrappers remain for convenience, but new robots
should use the generic `run_remote_train.bat` entrypoint.
