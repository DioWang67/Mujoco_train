# Sedon v5_22 Duck-Guided Foot Tuning Plan

Task class: Class C planning artifact. This plan does not run tuning, modify source XML, run PPO, or claim walking success.

## Summary

- Source: `duck_reference_guided_prototype`
- Confidence: `low`
- Valid for: `contact_persistence_tuning_only`
- Invalid for: `['final_mechanical_design_claim', 'walking_success_claim', 'sim2real_claim']`
- Duck reference status: `PARTIAL_REFERENCE`

## Search Space

| axis | center | values | status |
|---|---:|---|---|
| `toe_x_ratio` | `None` | `[]` | `manual_required` |
| `toe_z_relative_to_center` | `None` | `[]` | `manual_required` |
| `toe_patch_scale` | `None` | `[]` | `manual_required` |
| `center_z_offset` | `0.0` | `[-0.004, -0.002, 0.0, 0.002, 0.004]` | `READY` |
| `heel_x_ratio` | `None` | `[]` | `manual_required` |
| `heel_z_relative_to_center` | `None` | `[]` | `manual_required` |

## Variant Metadata

- `source`: `duck_reference_guided_prototype`
- `confidence`: `low/medium`
- `valid_for`: `contact_persistence_tuning_only`
- `invalid_for`: `final_mechanical_design_claim`

## Execution Policy

- `run_tuning_now`: `False`
- `requires_manual_duck_patch_review`: `True`
- `do_not_modify_source_xml`: `True`
- `do_not_run_ppo`: `True`

## Recommendation

- Duck XML does not expose explicit toe/center/heel primitive patches, so manual Duck foot contact review is required before running Duck-guided Sedon tuning.
