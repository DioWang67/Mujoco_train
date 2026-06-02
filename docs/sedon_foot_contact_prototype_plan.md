# Sedon Foot Contact Prototype Plan

Task class: Class C experiment workflow. This plan creates artifact-only contact prototypes for MuJoCo diagnostics.

## Safety Constraints

- Does not modify `private_assets/sedon/training_scene.xml`.
- Does not modify `sedon_baseline/train.py` or `sedon_baseline/eval.py`.
- Does not delete or move artifacts.
- Does not enter PPO and does not claim walking success.
- Open Duck Mini foot geometry is concept/reference only, not Sedon verified geometry.
- Every added patch is tagged `source=assumption/prototype`, `confidence=low`, `valid_for=simulation_prototype_only`.

## Variants

| variant | status | original foot collision | added patches | purpose |
|---|---|---|---:|---|
| `current_box_baseline` | `generated` | kept | 0 | Copy the current Sedon training scene without adding contact patches. |
| `simple_toe_center_heel` | `generated` | non-contact in generated XML | 6 | Replace each original foot collision with heel, center, and toe box patches estimated from Sedon foot length. |
| `duck_like_multi_patch` | `generated` | non-contact in generated XML | 10 | Use the Open Duck Mini multi-patch contact concept as a low-confidence Sedon simulation prototype. |

## Diagnostics

The comparison workflow evaluates neutral contact, forward pitch contact, backward pitch contact, left/right support symmetry, raw contact pairs, and contact patch classification.

Required metrics: `can_classify_center_toe_heel`, `neutral_center_first_left/right`, `forward_pitch_toe_handoff_candidate_left/right`, `toe_heel_bridge_contact_detected_left/right`, `left_right_symmetry`, `contact_model_sufficient_for_rollover_analysis`, and `recommendation_to_mechanical_team`.

## Trade-off

The prototype disables original foot collision only inside generated variant XMLs so patch-level contacts can be classified. This improves diagnostic clarity, but it means the variants are not physical Sedon geometry and must stay simulation-prototype-only.
