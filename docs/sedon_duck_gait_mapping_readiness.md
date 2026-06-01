# Sedon Duck Gait Mapping Readiness

## Summary

**Status: NOT READY**

Sedon and Open Duck Mini now have enough extracted morphology, actuator, foot-geometry, and semantic leg-joint data to start read-only gait reference analysis. They are not ready for direct Duck-like gait mapping into Sedon control, reward, PPO, or training configs.

The main blocker is not joint-name coverage. The semantic leg mapping covers the expected 10 Sedon leg joints, but Duck joint axes are not explicit in the selected XML, Sedon actuator envelope fields are incomplete, foot contact models differ strongly, and no Duck gait period, clearance, or reference motion source has been identified.

## Available Data

Sources:

- Sedon parameters: `configs/sedon/sedon_robot_parameters.yaml`
- Duck parameters: `references/open_duck_mini/duck_robot_parameters.yaml`
- Comparison report: `docs/sedon_duck_comparison.md`
- Mapping notes: `docs/sedon_duck_mapping_notes.md`
- Semantic mapping data: `references/open_duck_mini/sedon_duck_joint_mapping.yaml`

### Sedon

| Field | Value |
|---|---:|
| bodies | 11 |
| joints | 11 |
| geoms | 14 |
| actuators | 10 |
| foot-related geoms | 4 |
| explicit body mass | 10.045411478 |

Sedon exposes 10 hinge leg joints plus one floating base. The selected Sedon XML has explicit joint axes and ranges. Sedon actuators are `motor` actuators with `ctrlrange=[-100, 100]`; explicit actuator `kp` and `forcerange` are not present in the extracted XML snapshot.

Sedon foot-related geoms from the XML-only heuristic:

- `L_foot_collision`
- `L_link_ankle_pitch`
- `R_foot_collision`
- `R_link_ankle_pitch`

### Open Duck Mini

| Field | Value |
|---|---:|
| bodies | 16 |
| joints | 15 |
| geoms | 46 |
| actuators | 14 |
| foot-related geoms | 28 |
| explicit body mass | 2.1071407 |

Duck exposes 14 hinge joints plus one floating base. The leg joints have mechanical ranges in the XML, but their axes are not explicit in the selected source. Duck actuators are `position` actuators using default class `sts3215`, with extracted reference values `kp=13.37` and `forcerange=[-3.23, 3.23]`.

Duck foot-related geoms are mesh based and numerous. They were selected by name heuristic, not by validated contact-patch semantics.

### Semantic Leg Joint Mapping Coverage

| Coverage item | Value |
|---|---:|
| Sedon leg joints expected | 10 |
| Sedon leg joints mapped | 10 |
| Duck leg joints mapped | 10 |
| Mapping coverage | 100% semantic leg coverage |
| Mapping confidence | medium, sign validation required |
| Duck joints excluded | `neck_pitch`, `head_pitch`, `head_yaw`, `head_roll` |

Mapped semantic joints:

- right hip yaw, roll, pitch
- right knee pitch
- right ankle pitch
- left hip yaw, roll, pitch
- left knee pitch
- left ankle pitch

This coverage is sufficient for naming alignment, but not sufficient for motion transfer.

## Blocking Gaps

- **Duck joint axis unknown/null:** Duck hinge joint axes are not explicit in the selected XML. Sign-sensitive gait mapping requires simulation or compiled-model validation.
- **Sedon actuator kp unknown:** Sedon source exposes `motor` actuators, not position gains. Duck `kp=13.37` is not compatible with direct Sedon transfer.
- **Sedon actuator forcerange unknown:** Sedon extracted actuators do not provide explicit force limits. Duck `forcerange=[-3.23, 3.23]` cannot be used as a Sedon envelope.
- **Sedon leg length/foot scale not inferred:** XML-only comparison does not infer world-frame leg length, segment lengths, or normalized foot scale.
- **Duck gait period/clearance not available from XML:** The Duck XML provides robot structure, not dynamic gait timing, step clearance, stance ratio, or phase trajectory.
- **Sedon foot contact model too coarse compared with Duck:** Sedon has explicit box foot collision geoms; Duck has many mesh foot geoms. Contact patch semantics cannot be matched from names alone.

## Safe Reference Fields

Safe to use as reference:

- Semantic leg joint names and left/right grouping.
- Duck mechanical joint ranges as qualitative envelope references.
- Duck actuator type and default class as metadata describing the Duck source, not Sedon settings.
- Duck explicit mass and Sedon explicit mass as rough morphology context.
- Foot-related geom names as a checklist for later contact classification.
- The Sedon/Duck mass ratio from `docs/sedon_duck_comparison.md` as a morphology note, not a control multiplier.

Not safe to directly apply:

- Duck `kp` values.
- Duck `forcerange` values.
- Duck joint positions or ranges as Sedon action limits.
- Duck leg-joint signs before axis/sign validation.
- Duck foot mesh geometry as Sedon contact-patch geometry.
- Any Duck gait timing, clearance, or support phase unless it comes from a separate validated gait/reference-motion source.

## Required Next Diagnostics

1. **Joint axis/sign validation**
   - Load both models or compiled model metadata.
   - Apply small positive and negative perturbations per mapped joint.
   - Record whether semantic motion direction matches.

2. **Sedon actuator envelope clarification**
   - Identify whether Sedon control is torque-like, position-like, or otherwise transformed by environment code.
   - Derive safe action and force envelopes from Sedon-specific control semantics only.

3. **Sedon foot contact patch classification**
   - Classify Sedon left/right support contact geoms by actual contact behavior.
   - Separate sole contact, ankle visual/collision, and non-support contacts.

4. **Duck gait/reference motion source identification**
   - Find an actual Duck gait trajectory, controller log, motion capture, scripted controller, or published gait profile.
   - Extract gait period, stance/swing timing, clearance, and task-space foot path.

5. **Sedon scale mapping**
   - Build a normalized Sedon/Duck geometry scale table.
   - Use segment lengths and foot contact dimensions when available.
   - Mark fields unknown instead of inferring from mesh names.

## Recommended Next Step

Do not start PPO from this data.

Recommended sequence:

1. Build a Sedon/Duck normalized geometry scale table.
2. Identify or extract a Duck-like task-space gait reference: phase, stance ratio, foot path, clearance, and body motion targets.
3. Convert that reference into a Sedon-specific task-space target using only validated scale and sign mappings.
4. Run a scripted smoke test outside training.
5. Only after smoke-test evidence, consider whether training configs or rewards need changes.

## Do Not Do

- Do not directly apply Duck `kp` to Sedon.
- Do not directly apply Duck `forcerange` to Sedon.
- Do not directly apply Duck joint trajectories to Sedon.
- Do not use Duck XML joint ranges as Sedon action ranges.
- Do not treat semantic joint coverage as sign validation.
- Do not claim walking success from this report.
- Do not add PPO, reward, or training code based on this report alone.

## Readiness Decision

Duck-like gait mapping is **not ready** for Sedon control or training.

The project is ready for the next read-only preparation step: a normalized geometry scale table plus joint axis/sign diagnostics. After those are complete, a Duck-like task-space gait reference can be prepared for scripted smoke testing.
