# Sedon v5_22 Foot STL vs Collision Review

Task class: Class C read-only geometry diagnostic. This review does not modify source XML/URDF/train/eval/env, does not run PPO, does not use STL as final collision, and does not claim walking success.

## Summary

- Status: `FOOT_STL_VS_COLLISION_REVIEW_COMPLETE`
- STL candidates found: `8`
- Collision over simplified: `True`
- Physical foot may support rollover: `unknown`
- MuJoCo collision needs update: `unknown`
- Mechanical redesign needed: `unknown`
- Next step: `request_cad_step_or_bottom_specific_mesh`

## Found Foot-Related STL

| side | role | path | bottom shape | length x | width y | height z |
|---|---|---|---|---:|---:|---:|
| `left` | `ankle_pitch_visual_candidate` | `private_assets/sedon/mjcf_source/L_link_ankle_pitch.STL` | `flat` | 0.199268 | 0.0739498 | 0.14486 |
| `right` | `ankle_pitch_visual_candidate` | `private_assets/sedon/mjcf_source/R_link_ankle_pitch.STL` | `flat` | 0.199268 | 0.0739498 | 0.14486 |
| `left` | `ankle_pitch_visual_candidate` | `private_assets/sedon/original_urdf_package/urdf/meshes/L_link_ankle_pitch.STL` | `flat` | 0.199268 | 0.0739498 | 0.14486 |
| `right` | `ankle_pitch_visual_candidate` | `private_assets/sedon/original_urdf_package/urdf/meshes/R_link_ankle_pitch.STL` | `flat` | 0.199268 | 0.0739498 | 0.14486 |
| `left` | `ankle_pitch_visual_candidate` | `private_assets/sedon_v5_22/mjcf_source/L_link_ankle_pitch.STL` | `flat` | 0.199268 | 0.0739498 | 0.14486 |
| `right` | `ankle_pitch_visual_candidate` | `private_assets/sedon_v5_22/mjcf_source/R_link_ankle_pitch.STL` | `flat` | 0.199268 | 0.0739498 | 0.14486 |
| `left` | `ankle_pitch_visual_candidate` | `private_assets/SEEDON_URDF_5_22/meshes/L_link_ankle_pitch.STL` | `flat` | 0.199268 | 0.0739498 | 0.14486 |
| `right` | `ankle_pitch_visual_candidate` | `private_assets/SEEDON_URDF_5_22/meshes/R_link_ankle_pitch.STL` | `flat` | 0.199268 | 0.0739498 | 0.14486 |

## MuJoCo Collision Comparison

| side | collision geom | type | compared STL | mismatch score | over simplified |
|---|---|---|---|---:|---|
| `left` | `L_foot_collision` | `box` | `private_assets/sedon_v5_22/mjcf_source/L_link_ankle_pitch.STL` | `0.2642173911566153` | `True` |
| `right` | `R_foot_collision` | `box` | `private_assets/sedon_v5_22/mjcf_source/R_link_ankle_pitch.STL` | `0.26421740202749033` | `True` |

## Foot Bottom Interpretation

- `flat_bottom_candidate`, `rocker_like_candidate`, `continuous_bottom_candidate`, `discrete_patch_candidate`, `toe_contact_area_candidate`, and `heel_contact_area_candidate` are vertex-distribution heuristics.
- Any threshold value marked with `source=assumption` uses `confidence=low` and is valid for this review only.
- `ankle_pitch_visual_candidate` means the mesh is attached to the same ankle-pitch link as the current foot collision, not that it is a bottom-specific CAD surface.

## Recommendations

- Request CAD/STEP or a bottom-specific mesh because the discovered STL files are ankle-pitch visual candidates, not explicit sole/bottom geometry.
- Do not directly use the STL mesh as final MuJoCo collision.
- Use mechanical review to decide whether to update collision or redesign the foot bottom.

## Artifacts

- `foot_stl_summary.yaml`: `artifacts/sedon_debug/v5_22_foot_stl_vs_collision_review/foot_stl_summary.yaml`
- `foot_stl_bbox.csv`: `artifacts/sedon_debug/v5_22_foot_stl_vs_collision_review/foot_stl_bbox.csv`
- `collision_comparison.yaml`: `artifacts/sedon_debug/v5_22_foot_stl_vs_collision_review/collision_comparison.yaml`
