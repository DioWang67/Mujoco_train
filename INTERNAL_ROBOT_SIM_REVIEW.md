# Internal Robot Simulation Integration Review

Date: 2026-04-20

## Executive summary
This repo is a strong prototype for humanoid locomotion training, but it is still in a
"single-task research stack" shape. For internal company use (multiple robots, reproducibility,
validation, CI/CD, and deployment handoff), the highest-priority gaps are:

1. **Experiment reproducibility/traceability hardening** (artifact naming, seed/metadata lineage).
2. **Task/generalization coverage** (currently mainly flat-floor walking with physics randomization).
3. **Validation protocol standardization** (numeric BASE vs DR comparison should be first-class).
4. **Production interfaces** (robot-agnostic config, model export contract, safety gates).

## Current strengths
- Clean environment abstraction (`H1Env`) with readable reward decomposition and info logging.
- DR hooks exist (mass/friction/noise/command randomization, curriculum ramp).
- `VecNormalize` handling is considered in train/eval.
- Scripts are simple enough for quick iteration.

## Key risks for company-internal scaling

### R1. Scenario coverage is still narrow
- Training/eval are primarily flat floor + dynamics randomization.
- "DR" here is not equivalent to terrain/domain suite coverage (ramps, steps, uneven patches,
  contact edge cases, actuator lag profiles, sensor dropouts).

**Impact**: Can overfit to one contact geometry while appearing robust in dashboard metrics.

### R2. Validation depends too much on visual inspection
- Watching animation is useful but weak as acceptance criteria.
- Need deterministic scorecards (mean/std over seeds, stress buckets, fail-mode labels).

**Impact**: Hard to make pass/fail decisions for release candidates.

### R3. Artifact contract was previously ambiguous
- DR vs BASE model/stat naming mismatch already caused confusion.
- A formal artifact convention and manifest is still recommended.

**Impact**: Wrong model/stat pairing risk during internal handoff.

### R4. Missing integration contract for "sim -> real"
- No explicit interface doc for expected observation transform, action scaling, timing,
  fallback behavior, and runtime safety checks.

**Impact**: Deployment team has to reverse-engineer training assumptions.

## Recommendations (priority order)

### P0 (do now, 1-2 weeks)
1. Introduce a **numeric acceptance report** in every experiment run:
   - BASE eval mean/std (reward, episode length, tracking velocity)
   - DR eval mean/std under same checkpoint
   - PASS thresholds versioned in repo
2. Save a `manifest.json` per run:
   - git commit, CLI args, model path, vecnorm path, seed, DR settings
3. Establish naming policy:
   - `h1_ppo.zip` (base), `h1_ppo_dr.zip` (dr), paired vecnorm files mandatory.

### P1 (next)
1. Add scenario suites:
   - terrain buckets (flat, mild uneven, steps/ramp), disturbance buckets (push/noise/latency)
2. Add multi-seed aggregate runner and confidence intervals.
3. Define release gates:
   - e.g., DR score must exceed base score under perturbation set X.

### P2 (deployment-facing)
1. Document real-time contract:
   - control rate, delay budget, action clipping, emergency stop policy
2. Add export and verification path:
   - ONNX/TorchScript consistency check + latency benchmark
3. Add hardware shadow-mode replay for logged trajectories.

## Suggested immediate workflow for your team
1. Train/fine-tune candidate model.
2. Run numeric compare (`compare_eval.py`) for BASE vs DR.
3. Keep TensorBoard + numeric report + manifest as one release bundle.
4. Only promote model when bundle meets gate thresholds.

## What DR means here (important clarification)
- Current DR = parameter randomization (mass/friction/noise/command),
  not full terrain generation.
- If you need visible terrain differences in animation, you must add terrain assets
  (heightfield/meshes/scene variants) and route train/eval to those scenes.

## Action items checklist
- [ ] Add run manifest writer.
- [ ] Add scenario suite runner.
- [ ] Add terrain variant XML(s).
- [ ] Add release gate config file.
- [ ] Add deployment interface doc.
