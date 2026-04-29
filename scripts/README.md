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
  Interactive H1, grasp, and Sedon evaluation menu.
- `sedon_eval.bat`
  Interactive Sedon eval menu; accepts extra args for `eval.py --project sedon`.
- `tensorboard.bat`  
  Start local TensorBoard against `logs/tb`.

## Remote Training

- `h1_remote_train.bat`  
  Start formal remote H1 training.
- `grasp_remote_train.bat`  
  Start formal remote grasp training with `--phase full --n-envs 8`.
- `sedon_remote_train.bat`  
  Start formal remote Sedon standing training with `--n-envs 4`.
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
- `sedon_tensorboard.bat`  
  Open the remote Sedon TensorBoard tunnel on port 6008.
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
  Deploy the current committed release for H1 and run H1 smoke verify.
- `grasp_deploy_remote.bat`  
  Deploy the current committed release for grasp and run grasp smoke verify.
- `sedon_deploy_remote.bat`  
  Deploy the current committed release for Sedon, include private assets, and
  run Sedon smoke verify.
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

Use the generic `run_remote_train.bat` entrypoint for new robots.
Do not add a new per-robot wrapper unless it captures a real operator workflow
that the generic script cannot express.

If a robot needs ignored assets on the remote host, define `private_asset_dir`
in `configs/<slug>/project.json` and deploy with:

```bat
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\deploy_remote_release.ps1 -ProjectSlug <slug> -VerifyProject <slug> -IncludePrivateAssets
```
