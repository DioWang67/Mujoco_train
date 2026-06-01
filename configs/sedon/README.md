# Sedon Configs

Sedon configs are JSON overrides consumed through `SEDON_CONFIG_OVERRIDES`.
They do not replace `configs/sedon/train.json`; they override reward, gait,
termination, and controller settings inside `SedonStandingConfig`.

## Active Config

Use this for the current reverse-knee/no-tiptoe 10M experiment:

```text
reverse_knee_no_tiptoe_walk.json
```

Run locally:

```powershell
$env:SEDON_CONFIG_OVERRIDES='configs\sedon\reverse_knee_no_tiptoe_walk.json'
python train.py --project sedon --total-timesteps 10000000 --n-envs 4 --reset-noise-scale 0.01
```

Run on remote release:

```bash
export SEDON_CONFIG_OVERRIDES=configs/sedon/reverse_knee_no_tiptoe_walk.json
python -m sedon_baseline.train --total-timesteps 10000000 --n-envs 128 --reset-noise-scale 0.01
```

## Config Groups

| Config | Status | Purpose |
|---|---|---|
| `train.json` | Canonical base training hyperparameters | PPO defaults, episode length, eval/checkpoint frequency. |
| `project.json` | Canonical project metadata | Shared dispatcher and deploy discovery. |
| `blue_dynamic_support_gait.json` | Active | Blue/BDX-style low-clearance support-phase gait objective. |
| `reverse_knee_no_tiptoe_walk.json` | Experimental | Slow reverse-knee walk with anti-tiptoe bias. |
| `reverse_knee_walk*.json` | Experimental | Earlier reverse-knee walk reward/gait variants. |
| `reverse_knee_short*.json` | Experimental | Short-horizon reverse-knee stability checks. |
| `reverse_knee_lift_recovery.json` | Experimental | Recovery/lift-focused variant. |
| `blue_*.json`, `nv_blue_style_walk.json` | Experimental | NVIDIA Blue / BDX style low-clearance experiments. |
| `com_shift*.json` | Diagnostic curriculum | COM shift and unload exploration before walking. |
| `sweep_*.json` | Sweep inputs | Used by search/sweep scripts, not direct default training. |
| `zero_action_safe_stand.json` | Diagnostic | Static standing seed verification. |

## How To Choose

For Blue/BDX-style Sedon gait training right now:

```text
blue_dynamic_support_gait.json
```

For the older reverse-knee/no-tiptoe branch:

```text
reverse_knee_no_tiptoe_walk.json
```

For debugging why Sedon cannot unload a foot:

```text
com_shift_curriculum.json
com_shift_micro_explore_*.json
zero_action_safe_stand.json
```

For comparing alternative gait priors:

```text
reverse_knee_walk_forward_balanced.json
reverse_knee_walk_forward_push.json
blue_support_phase_walk.json
nv_blue_style_walk.json
```

## Rules

- Put repeatable reward/gait settings here, not in shell scripts.
- Keep one config per experiment question.
- Do not keep tuning by editing `train.json` unless changing global PPO defaults.
- Name active configs by intent, not by date.
- Archive or remove configs only after checking scripts, docs, and experiment logs.
