# Sedon Blue-Like Dynamic Gait Spec

## Part 1: Architecture And Design Summary

### Task Classification

Class C: Script / Tool / Experiment.

This document defines an experimental locomotion target and audit criteria for Sedon MuJoCo work. It is not a production API contract and should stay practical: precise enough to prevent false positives, but not so abstract that it slows mechanical diagnosis.

### Requirement Summary

The target is a Blue / BDX-like dynamic small-step gait, not merely static balance and not a grounded shuffle. A successful gait should preserve forward momentum, let the body move in a controlled falling pattern, use hip roll and hip pitch to move load and swing direction, exploit the rounded sole for passive rollover, and place the next foot where it can catch the falling body.

Primary inputs:

- MuJoCo rollout timeline with base pose/velocity, foot contacts, foot forces, and contact region data.
- Optional controller debug timeline with phase, support/swing side, targets, and actual foot placement.

Primary outputs:

- Pass/fail classification for staged diagnostics.
- Timeline evidence for forward momentum, support/swing force split, center-to-toe rollover, touchdown behavior, and left/right alternation.

Key hidden constraint:

- `base_x` increasing alone is insufficient. Forward displacement caused by both-foot dragging, external push without capture, or long sliding contact is not walking.

### Solution Strategy

Use staged criteria. Phase 0 accepts the current slow grounded baseline only as a baseline. Phase 1 checks controlled falling and rollover before demanding visible foot clearance. Later phases can add stronger swing unload, capture touchdown, and true single-support behavior.

This spec intentionally avoids defining success only through reward terms. The gait must be judged from physical diagnostics: contact force, contact region ordering, support alternation, touchdown placement, and stability.

### Edge Cases

- Empty or truncated rollout: fail; no gait evidence exists.
- Forward displacement with both feet dragging for most of the run: fail as dynamic gait, even if `base_x` increases.
- Toe contact without earlier center contact: not a center-to-toe handoff.
- No-contact bursts or jumps: fail for Phase 1 and later.
- One foot permanently carries all load: fail alternation criteria.
- Touchdown behind COM/base: fail capture-step placement for later phases.
- High touchdown force spike: fail controlled impact criteria.

### Trade-Off

The staged criteria deliberately accept a grounded intermediate Phase 1 so we can diagnose rollover and controlled falling before solving full swing clearance. The cost is that Phase 1 pass is not walking; it is only evidence that the mechanical bridge from shuffle to dynamic stepping exists.

### Concurrency / State Safety

The diagnostics are deterministic single-process MuJoCo rollouts. They do not share mutable state across workers and do not introduce concurrency risk.

## Target Gait Definition

Sedon should gradually approach this gait pattern:

1. The base / COM maintains positive forward velocity.
2. The body stays upright enough but moves with controlled falling / forward momentum.
3. Hip roll and pelvis lean shift load toward the support side.
4. Hip pitch creates a forward or side-forward swing tendency.
5. The support foot naturally rolls through the rounded sole.
6. The support contact should show center-to-toe rollover under forward momentum.
7. The swing foot moves quickly forward or side-forward.
8. The next touchdown lands in front of, or side-front of, the base / COM.
9. The touchdown catches the falling body without a large uncontrolled impact.
10. The support side alternates left/right across steps.

## Current Project Position

Current status is:

```text
Phase 0: Blue-like grounded / slow forward shuffle baseline
```

This is not final walking. Phase 0 is useful because it proves Sedon can stay upright, produce low-speed forward displacement, and expose load/contact diagnostics. It must not be reported as Blue-like dynamic gait.

Known current conclusions:

- `v5_a` foot geometry is the best current sole candidate.
- `v5_a` has center-first standing.
- `v5_a` can produce center-to-toe handoff under dynamic push.
- PPO smoke tests did not learn dynamic gait.
- Teacher imitation is valid as a pipeline but remains grounded shuffle.
- Visible stepping currently fails mainly because swing-foot unload authority is insufficient and contact constraints block lift.
- Larger lift targets, landing interpolation sweeps, and reference phase sweeps have low ROI right now.

## Required Dynamic-Gait Evidence

A rollout must be evaluated against these indicators:

1. Base / COM forward velocity must be positive.
2. The body should preserve controlled falling / forward momentum.
3. Support-foot force must be clearly larger than swing-foot force.
4. Swing-foot force must decrease before swing-foot lift.
5. The support foot should show center-to-toe rollover.
6. Swing-foot touchdown should land in front of or side-front of the base / COM.
7. Touchdown impact must be controlled.
8. Jumping is not allowed.
9. Long no-contact intervals are not allowed.
10. Left/right support phases must alternate.
11. Both-foot dragging must not be accepted as dynamic gait.

## Phase Definitions

### Phase 0: Grounded / Slow Forward Shuffle Baseline

Purpose:

- Establish safe standing, contact continuity, and repeatable debug timelines.
- Allow low-speed forward movement only as a diagnostic baseline.

Required evidence:

- Stable upright posture.
- No jump.
- No long no-contact burst.
- Contact and force measurements are usable.

Not accepted as final walking:

- Both feet remain planted while the base drifts forward.
- There is no clear support/swing force separation.
- There is no capture-style touchdown.

### Phase 1: Controlled Falling / Rollover Diagnostic

Purpose:

- Verify that Sedon can use forward momentum and the rounded sole without losing stability.
- Confirm center-to-toe handoff and emerging support alternation before demanding visible clearance.

Phase 1 pass conditions:

```text
contact_none_ratio == 0
jump_count == 0
min_upright >= 0.98
forward_displacement > 0
mean_forward_velocity > 0
toe_handoff_detected == true
left_right_phase_switch_count >= 1
```

Phase 1 does not require:

- Visible foot clearance.
- True single support.
- Full capture stepping.

### Phase 2: Capture-Step Skeleton

Purpose:

- Change controller logic from "lift the foot" to "place the next foot to catch the forward-falling body."

Expected evidence:

- FSM phases are explicit.
- Swing target is computed from base position and base velocity.
- Hip roll / pelvis lean changes load transfer.
- Hip pitch creates forward swing tendency.
- Toe rocker / rollover is observed from contact diagnostics, not hard-coded.

### Phase 3: Dynamic Step Candidate

Purpose:

- Demonstrate repeated side-alternating capture steps with controlled touchdown.

Required evidence:

- Positive forward displacement.
- Support/swing force separation.
- Swing foot unload before touchdown.
- Touchdown in front or side-front of base / COM.
- Impact below configured limit.
- No jump and no long airborne interval.

## Explicit Non-Goals

Do not treat these as success:

- A long PPO run that only learns to slide.
- A reference pose that moves `base_x` while both feet drag.
- A single rendered clip with subjective visual success.
- A rollout that has forward displacement but no support alternation.
- A rollout that obtains speed only by reward pressure without physical stepping evidence.

## Part 3: Testing And Verification

Verification should use timeline CSV plus JSON summaries. A valid report must include forward velocity, base roll/pitch/upright, side forces, support/swing ratios, center/toe contact, contact-none ratio, jump count, support phase switches, and pass/fail reasons.

At least these cases should be checked when changing tools:

- Stable zero-action or low-action standing does not falsely pass Phase 1 if there is no forward momentum.
- A dynamic-push rollout with center-to-toe handoff can pass the rollover part only when contact continuity and upright constraints hold.
- Any no-contact burst fails Phase 1.

## Part 4: Performance And Risk Analysis

The diagnostics are linear in rollout length and number of active contacts. For a 600-step rollout, the runtime bottleneck is MuJoCo stepping, not CSV/JSON writing.

If rollout length grows 10x, disk output and render capture become the first practical issue. If it grows 100x, aggregate streaming should replace keeping all rows in memory. Current scale does not justify that complexity.

Main risk:

- Contact labels and force thresholds can mislead if foot geometry names change. Tools should fail clearly or record unknown contacts instead of silently declaring gait success.

## Part 5: Usage Notes

Use this spec as the acceptance reference for:

- `tools/sedon_blue_like_phase1_rollover_diagnostic.py`
- `tools/sedon_capture_step_controller_v1.py`

Do not report Phase 0 or Phase 1 as final walking. The project direction remains Blue / BDX-like dynamic gait.
