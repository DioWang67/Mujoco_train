# Open Duck Foot Contact Geometry Reference

Task class: Class C local XML reference extraction. No network was used and Duck geometry is not directly applied to Seedon.

## Summary

- Source XML: `references/open_duck_mini/source/playground_open_duck_mini_v2/xmls/open_duck_mini_v2.xml`
- Foot-related geom count: `24`
- Visible active contact candidates: `2`
- Candidate names: `['left_foot_bottom_tpu', 'right_foot_bottom_tpu']`
- Normalized reference status: `PARTIAL_REFERENCE`

## Normalized Geometry Reference

| field | value | source type | confidence |
|---|---:|---|---|
| `foot_length_estimate` | `None` | `manual_required` | `low` |
| `toe_x_ratio` | `None` | `manual_required` | `low` |
| `center_x_ratio` | `None` | `manual_required` | `low` |
| `heel_x_ratio` | `None` | `manual_required` | `low` |
| `toe_z_relative_to_center` | `None` | `manual_required` | `low` |
| `heel_z_relative_to_center` | `None` | `manual_required` | `low` |
| `toe_patch_size_ratio` | `None` | `manual_required` | `low` |
| `heel_patch_size_ratio` | `None` | `manual_required` | `low` |
| `inner_outer_width_ratio` | `None` | `manual_required` | `low` |

## Foot-Related Geoms

| name | parent body | side | category | type | pos | size | class/default | contype | conaffinity | active? |
|---|---|---|---|---|---|---|---|---|---|---|
| `<unnamed:knee_and_ankle_assembly:0>` | `knee_and_ankle_assembly` | `left` | `unknown` | `mesh` | `[0.01606, 0.065, 0.10915]` | `None` | `visual` | `unknown` | `unknown` | `False` |
| `<unnamed:knee_and_ankle_assembly:1>` | `knee_and_ankle_assembly` | `left` | `unknown` | `mesh` | `[0.01606, 0.14365, 0.10925]` | `None` | `visual` | `unknown` | `unknown` | `False` |
| `<unnamed:knee_and_ankle_assembly:2>` | `knee_and_ankle_assembly` | `left` | `unknown` | `mesh` | `[0.01606, 0.14365, 0.10915]` | `None` | `visual` | `unknown` | `unknown` | `False` |
| `<unnamed:knee_and_ankle_assembly:3>` | `knee_and_ankle_assembly` | `left` | `unknown` | `mesh` | `[0.01606, 0.14365, 0.10915]` | `None` | `visual` | `unknown` | `unknown` | `False` |
| `<unnamed:knee_and_ankle_assembly_2:4>` | `knee_and_ankle_assembly_2` | `left` | `unknown` | `mesh` | `[0.01606, 0.14365, 0.10925]` | `None` | `visual` | `unknown` | `unknown` | `False` |
| `<unnamed:knee_and_ankle_assembly_2:5>` | `knee_and_ankle_assembly_2` | `left` | `unknown` | `mesh` | `[0.01606, 0.14365, 0.10915]` | `None` | `visual` | `unknown` | `unknown` | `False` |
| `<unnamed:knee_and_ankle_assembly_2:6>` | `knee_and_ankle_assembly_2` | `left` | `unknown` | `mesh` | `[0.01606, 0.14365, 0.10915]` | `None` | `visual` | `unknown` | `unknown` | `False` |
| `<unnamed:foot_assembly:7>` | `foot_assembly` | `left` | `foot` | `mesh` | `[0.01606, 0.2223, 0.10905]` | `None` | `visual` | `unknown` | `unknown` | `False` |
| `<unnamed:foot_assembly:8>` | `foot_assembly` | `left` | `sole` | `mesh` | `[0.01656, 0.2228, 0.10955]` | `None` | `visual` | `unknown` | `unknown` | `False` |
| `left_foot_bottom_tpu` | `foot_assembly` | `left` | `sole` | `mesh` | `[0.01656, 0.2228, 0.10955]` | `None` | `collision` | `unknown` | `unknown` | `True` |
| `<unnamed:foot_assembly:10>` | `foot_assembly` | `left` | `sole` | `mesh` | `[0.01656, 0.2228, 0.10955]` | `None` | `visual` | `unknown` | `unknown` | `False` |
| `<unnamed:foot_assembly:11>` | `foot_assembly` | `left` | `foot` | `mesh` | `[0.01606, 0.2223, 0.10905]` | `None` | `visual` | `unknown` | `unknown` | `False` |
| `<unnamed:knee_and_ankle_assembly_3:12>` | `knee_and_ankle_assembly_3` | `right` | `unknown` | `mesh` | `[0.01606, -0.065, 0.1092]` | `None` | `visual` | `unknown` | `unknown` | `False` |
| `<unnamed:knee_and_ankle_assembly_3:13>` | `knee_and_ankle_assembly_3` | `right` | `unknown` | `mesh` | `[0.01606, -0.14365, -0.07215]` | `None` | `visual` | `unknown` | `unknown` | `False` |
| `<unnamed:knee_and_ankle_assembly_3:14>` | `knee_and_ankle_assembly_3` | `right` | `unknown` | `mesh` | `[0.01606, -0.14365, -0.07205]` | `None` | `visual` | `unknown` | `unknown` | `False` |
| `<unnamed:knee_and_ankle_assembly_3:15>` | `knee_and_ankle_assembly_3` | `right` | `unknown` | `mesh` | `[0.01606, -0.14365, -0.07205]` | `None` | `visual` | `unknown` | `unknown` | `False` |
| `<unnamed:knee_and_ankle_assembly_4:16>` | `knee_and_ankle_assembly_4` | `right` | `unknown` | `mesh` | `[0.01606, 0.14365, 0.10925]` | `None` | `visual` | `unknown` | `unknown` | `False` |
| `<unnamed:knee_and_ankle_assembly_4:17>` | `knee_and_ankle_assembly_4` | `right` | `unknown` | `mesh` | `[0.01606, 0.14365, 0.10915]` | `None` | `visual` | `unknown` | `unknown` | `False` |
| `<unnamed:knee_and_ankle_assembly_4:18>` | `knee_and_ankle_assembly_4` | `right` | `unknown` | `mesh` | `[0.01606, 0.14365, 0.10915]` | `None` | `visual` | `unknown` | `unknown` | `False` |
| `<unnamed:foot_assembly_2:19>` | `foot_assembly_2` | `right` | `foot` | `mesh` | `[0.01606, 0.2223, 0.10905]` | `None` | `visual` | `unknown` | `unknown` | `False` |
| `<unnamed:foot_assembly_2:20>` | `foot_assembly_2` | `right` | `sole` | `mesh` | `[0.01656, 0.2228, 0.10955]` | `None` | `visual` | `unknown` | `unknown` | `False` |
| `right_foot_bottom_tpu` | `foot_assembly_2` | `right` | `sole` | `mesh` | `[0.01656, 0.2228, 0.10955]` | `None` | `collision` | `unknown` | `unknown` | `True` |
| `<unnamed:foot_assembly_2:22>` | `foot_assembly_2` | `right` | `sole` | `mesh` | `[0.01656, 0.2228, 0.10955]` | `None` | `visual` | `unknown` | `unknown` | `False` |
| `<unnamed:foot_assembly_2:23>` | `foot_assembly_2` | `right` | `foot` | `mesh` | `[0.01606, 0.2223, 0.10905]` | `None` | `visual` | `unknown` | `unknown` | `False` |

## Limitations

- Foot-related geoms use name and ancestor-body heuristics.
- Mesh geoms are listed but their dimensions are not inferred from mesh files.
- Visible active contact candidates are based on explicit XML class/contype/conaffinity only.
- Duck reference is for Seedon prototype guidance only, not final mechanical design.
