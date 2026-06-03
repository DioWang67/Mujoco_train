# Seedon Workflow

Seedon is currently in a locomotion research phase. The practical goal is not
"high reward"; it is a gait that visually and mechanically matches the intended
reverse-knee, low-clearance, no-tiptoe style.

## Current Canonical Flow

```text
build scene
-> smoke test
-> train with one named config
-> view rollout
-> audit contacts / gait phases
-> adjust config or reference
```

Do not judge a run only by reward. Always inspect contact, foot clearance,
knee pitch phase, base height, and uprightness.

## Local Preparation

```powershell
python -m tools.build_seedon_training_scene
python -m tools.smoke_seedon_env --steps 20
python train.py --project seedon --smoke --n-envs 1 --reset-noise-scale 0
```

## Current Blue-Style Support Gait Experiment

Config:

```text
configs/seedon/blue_dynamic_support_gait.json
```

Intent:

- train a Blue/BDX-style low-clearance support gait, not H1-like humanoid walking
- reward support-side load transfer and swing-foot unload
- keep clearance low and controlled instead of encouraging hopping
- preserve longer double-support recovery windows

Local command:

```powershell
$env:SEEDON_CONFIG_OVERRIDES='configs\seedon\blue_dynamic_support_gait.json'
python train.py --project seedon --total-timesteps 2000000 --n-envs 4 --reset-noise-scale 0.005
```

If the local Windows environment blocks `SubprocVecEnv`, use `--n-envs 1`.

## Remote Training

From local Windows, preferred deployment and smoke-check wrapper:

```powershell
scripts\seedon_remote_deploy_and_check.bat
```

This wrapper reads `.env.remote`, uploads a small release archive, activates
`code/current`, and runs a remote Seedon smoke check. Password-based deployment
uses Windows OpenSSH with `SSH_ASKPASS`, so `REMOTE_PASSWORD` is read from
`.env.remote` without typing it each time.

Use this when you only want to check the active remote release:

```powershell
scripts\seedon_remote_check.bat
```

Use this to resume yesterday's Seedon training from the newest remote checkpoint:

```powershell
scripts\seedon_remote_resume_training.bat
```

Use this to inspect the running remote training process and tail the latest log:

```powershell
scripts\seedon_remote_training_status.bat
```

The resume wrapper currently targets:

```text
configs/seedon/blue_dynamic_support_gait.json
```

and resumes from:

```text
/root/anaconda3/mujoco-train-system/runs/seedon/models/seedon/latest_model.zip
/root/anaconda3/mujoco-train-system/runs/seedon/models/seedon/vecnorm.pkl
```

If you need to run manually from inside the remote release directory:

```bash
cd /root/anaconda3/mujoco-train-system/code/current
mkdir -p /root/anaconda3/mujoco-train-system/runs/seedon/logs/seedon

export SEEDON_CONFIG_OVERRIDES=configs/seedon/blue_dynamic_support_gait.json
export MUJOCO_TRAIN_LAYOUT_ROOT=/root/anaconda3/mujoco-train-system
export MUJOCO_TRAIN_PROJECT_SLUG=seedon
export MKL_THREADING_LAYER=GNU
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python -m seedon_baseline.train --total-timesteps 2000000 --n-envs 128 --reset-noise-scale 0.005 2>&1 | tee /root/anaconda3/mujoco-train-system/runs/seedon/logs/seedon/blue_dynamic_support_gait_2m.log
```

Use `python -m seedon_baseline.train` on the remote if the generic dispatcher
behaves unexpectedly. It removes one layer of indirection and calls the Seedon
training module directly.

## Deploy Content Overlay

For repeatable one-off deployments, put files under:

```text
deploy_content/
```

using repository-relative paths. The deploy tool overlays that directory onto
the release root before uploading. Examples:

```text
deploy_content/configs/seedon/blue_dynamic_support_gait.json
deploy_content/tools/custom_debug_tool.py
deploy_content/seedon_baseline/env.py
```

will deploy as:

```text
configs/seedon/blue_dynamic_support_gait.json
tools/custom_debug_tool.py
seedon_baseline/env.py
```

Then run:

```powershell
scripts\seedon_remote_deploy_and_check.bat
```

This is the preferred path when the deployment content is not meant to become
part of the repo yet. Keep normal source changes in the worktree or commits;
use `deploy_content/` for explicit overlay payloads.

Start with a short remote sanity run before a long run:

```bash
python -m seedon_baseline.train --total-timesteps 20000 --n-envs 32 --reset-noise-scale 0.01
```

If it starts cleanly, increase `--n-envs`. If the process is killed or stalls,
reduce to `64` or `32`.

## Monitoring

Training log:

```bash
tail -f /root/anaconda3/mujoco-train-system/runs/seedon/logs/seedon/resume_blue_dynamic_support_gait_*.log
```

TensorBoard from local Windows:

```powershell
scripts\seedon_tensorboard.bat
```

Important training columns:

- `MeanLen`: should rise and avoid short fall episodes.
- `BaseZ`: should stay near the configured target height.
- `Upright`: should remain high.
- `FwdV`: should become positive but not rely on jumping or tiptoe artifacts.

## Post-Training Evaluation

Use viewer first:

```powershell
python -m tools.debug_seedon_gait_viewer --mode policy --checkpoint-path models\seedon\latest_model.zip --vecnorm-path models\seedon\vecnorm.pkl --steps 600 --speed 0.5 --out-csv artifacts\seedon_debug\policy_gait_view.csv
```

Use audit next:

```powershell
python -m tools.debug_seedon_gait_audit --mode policy --steps 600 --out-csv artifacts\seedon_debug\policy_gait_audit.csv
```

Judge these first:

- swing foot unloads before any lift
- foot clearance is low and real, not base bounce
- knee pitch follows the intended reverse-knee phase
- COM remains stable
- no foot-foot collision
- no base proxy floor contact
- no persistent tiptoe or hopping

## Pose / Reference Tools

Use these only when designing or inspecting reference motion. They are not
required for every PPO run.

```powershell
python -m tools.debug_seedon_pose_editor --scene private_assets/seedon/training_scene.xml
python -m tools.debug_seedon_gait_viewer --mode scripted --gait-seed-path artifacts/seedon_debug/seedon_reference_gait_seed.json --steps 400
```

Current limitation: hand-authored poses can easily be visually plausible but
physically invalid. Use `Settle pose`, `Preview sequence`, and gait audit before
training from a reference.

## Common Failure Modes

`unrecognized arguments: --total-timesteps`

- The Python module being executed is old, or the dispatcher is loading a
  different module than expected.
- On remote, run:

```bash
python - <<'PY'
import inspect
import seedon_baseline.train as st
print(st.__file__)
print("--total-timesteps" in inspect.getsource(st.parse_args))
PY
```

If it prints `True`, run direct module training:

```bash
python -m seedon_baseline.train --total-timesteps 10000000 --n-envs 32 --reset-noise-scale 0.01
```

Remote `git rev-parse` fails

- This is expected in deployed release directories. `code/current` is an
  extracted release, not a Git worktree.

Remote smoke fails with `Geom 'R_foot_collision' not found`

- The code expects the newer Seedon training scene, but the active release is
  using older private assets.
- Run one deploy with private assets included, then return to fast deploys:

```powershell
scripts\remote_auto_deploy.bat --include-private-assets
```

- After that, normal deploys symlink the already uploaded private assets and do
  not resend several hundred MB each time.

Training still learns tiptoe

- Treat the run as a diagnostic result. Do not blindly add more timesteps.
- Next step should be reference tracking or stronger gait-phase terms, not only
  larger PPO runs.
