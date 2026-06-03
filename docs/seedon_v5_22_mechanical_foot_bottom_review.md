# Seedon v5_22 Mechanical Foot Bottom Review

Task class: Class C mechanical review note. This document translates simulation diagnostics into mechanical foot bottom requirements. It does not define final mechanical design, run PPO, or claim walking success.

## Summary

Seedon v5_22 is mechanically loadable and stable under nominal PD hold, but the current foot contact model is not sufficient for toe handoff or rollover diagnostics. The central issue is not just ankle torque. Toe loading can be produced, and ankle boost reduces saturation, but contact persistence remains poor.

The next decision should be mechanical foot bottom review before more controller or PPO work.

## Why Current Foot Contact Is Insufficient

- `current_box_baseline` cannot classify center/toe/heel contact regions.
- Discrete toe/center/heel patches are classifiable, but they did not produce a toe handoff candidate.
- The best toe handoff targeted probe created high toe loading but still had `contact_none_rate=0.85`.
- The continuous foot bottom prototype produced local-x progression toward toe, but best contact persistence was still insufficient.
- Ankle boost reduced saturation but did not fix contact persistence.

## Evidence From Experiments

| Diagnostic | Key result | Interpretation |
|---|---|---|
| v5_22 sanity gate | `PARTIAL_PASS` | Model can load and PD hold is stable, but not a complete controller/motor baseline. |
| v5_22 actuator envelope | `PARTIAL_ACTUATOR_ENVELOPE` | Torque data is bounded diagnostic only; torque side remains unknown. |
| foot x actuator sensitivity | ankle boost reduces saturation | Authority helps saturation but does not create toe handoff. |
| toe handoff targeted probe | best toe ratio `0.928829`, contact_none_rate `0.85` | Toe loading is possible but intermittent. |
| Duck foot reference | active candidates are `foot_bottom_tpu` mesh collisions | Duck does not support blind toe/center/heel ratio copying. |
| continuous foot bottom probe | best overall contact_none_rate `0.82`; best continuous `0.83` | Continuous direction is slightly better than discrete, but still fails persistence. |

## Duck Reference Interpretation

Open Duck Mini XML does not expose explicit toe/center/heel primitive contact patches. The visible active foot contact candidates are:

- `left_foot_bottom_tpu`
- `right_foot_bottom_tpu`

This suggests Duck is closer to a continuous TPU-like bottom contact surface than a set of separated toe/center/heel primitives. Therefore Seedon should not continue blind patch-ratio tuning from Duck. The useful lesson is the contact concept: continuous bottom support with diagnostic regions inferred from contact point location.

## Required Foot Bottom Behavior

- Stable center contact: neutral stance should maintain persistent support near the center/bottom region.
- Continuous rollover: contact point should progress from center/rear toward toe under forward lean or toe-down posture.
- Persistent toe contact: toe-region contact should persist over time, not appear only as intermittent spikes.
- No toe/heel bridge: toe and heel should not both bypass center in a separated bridge-like support pattern.
- Low contact_none_rate: prototype threshold target is `contact_none_rate < 0.25`; current best is about `0.82`.

## Mechanical Questions For Review

- What is the actual foot bottom shape in CAD/STEP?
- Does the forefoot have a toe rocker, radius, chamfer, or curved underside?
- Is there a TPU/rubber contact surface, and where is it relative to the rigid foot?
- Where does the ankle pitch axis project relative to the foot bottom, toe, and heel?
- What are the measured foot length, toe length, heel length, and bottom width?
- What friction should the contact material use in MuJoCo?
- Can CAD/STEP/mesh provide a simplified bottom collision geometry?
- Should the MuJoCo foot collision be a continuous bottom mesh/convex hull rather than separated primitive patches?

## Recommended Design Directions

- Use a continuous rocker-like bottom instead of separated tiny patches.
- Represent a TPU-like bottom surface if the real mechanism has one.
- Provide enough toe contact area to keep toe support persistent.
- Avoid relying on a small isolated toe patch as the main handoff mechanism.
- Use diagnostic heel/center/toe regions for measurement, but keep the collision surface continuous.

## What Simulation Can And Cannot Claim

Simulation can claim:

- The current Seedon foot contact model is insufficient for rollover diagnostics.
- Toe loading is possible but not persistent under current prototypes.
- Continuous bottom prototypes are directionally better than discrete patches but still insufficient.
- Ankle authority alone does not solve contact persistence.

Simulation cannot claim:

- Walking success.
- Sim2real readiness.
- Final mechanical design validity.
- Verified material friction.
- PPO readiness.

## Next Decision

1. Wait for mechanical foot bottom review.
2. Update MuJoCo foot collision using reviewed bottom geometry.
3. Rerun the contact persistence / continuous bottom probe.
4. Only after contact persistence improves, consider scripted rollover sequencing.
5. Do not PPO before the foot contact model passes persistence diagnostics.
