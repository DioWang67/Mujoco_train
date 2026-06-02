# Sedon Foot Contact Variant Comparison

Task class: Class C contact prototype diagnostic. This report does not modify the source model, enter PPO, or claim walking success.

- manifest: `D:\Git\robotlearning\h1_mujoco\artifacts\sedon_debug\foot_contact_variants\manifest.json`
- Duck foot geometry usage: concept/reference only, not Sedon verified geometry.
- Added patch metadata: `source=assumption/prototype`, `confidence=low`, `valid_for=simulation_prototype_only`.

## Metrics

| variant | classify C/T/H | neutral center L | neutral center R | forward toe L | forward toe R | bridge L | bridge R | symmetry | rollover-analysis sufficient |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `current_box_baseline` | False | True | True | False | False | False | False | False | False |
| `simple_toe_center_heel` | True | False | True | True | True | True | True | True | False |
| `duck_like_multi_patch` | True | False | False | True | True | True | True | True | False |

## Recommendations

- `current_box_baseline`: Not sufficient: add or repair named center/toe/heel patches before rollover analysis.
- `simple_toe_center_heel`: Not sufficient: toe/heel bridge contact appears; adjust patch spacing or height.
- `duck_like_multi_patch`: Not sufficient: toe/heel bridge contact appears; adjust patch spacing or height.

## Safety Notes

- `contact_model_sufficient_for_rollover_analysis` means only that the contact model may be useful for scripted contact diagnostics.
- It is not walking success, not mechanical validation, and not PPO readiness.
- Variants remain simulation prototypes until a mechanical owner validates geometry and contact behavior.
