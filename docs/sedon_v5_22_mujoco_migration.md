# Sedon v5_22 MuJoCo Migration

Task class: Class C prototype migration. This keeps the older Sedon MuJoCo
assets as archived reference and makes the updated `SEEDON_URDF_5_22` package
the active mechanical source for new Sedon diagnostics.

## Active Source

- Version: `v5_22`
- Source URDF: `private_assets/SEEDON_URDF_5_22/urdf/SEEDON_URDF_5_21.urdf`
- Generated scene: `private_assets/sedon_v5_22/training_scene.xml`
- Generator: `tools/sedon/geometry/generate_sedon_v5_22_mjcf_scene.py`
- Version registry: `configs/sedon/sedon_model_versions.yaml`
- Default Sedon env scene: `private_assets/sedon_v5_22/training_scene.xml`

## What Changed

- Uses the updated v5_22 STL visual meshes directly.
- Uses v5_22 mass, COM, inertia, joint origins, joint axes, and joint ranges.
- Keeps the old `private_assets/sedon/` scene as archived reference only.
- Adds a versioned output root so future versions can use
  `private_assets/sedon_<version>/` without changing the Sedon baseline code.

## Prototype Assumptions

All assumptions are valid for `simulation_prototype_only`.

- `motor_ctrlrange=-100 100`
  - source: `assumption`
  - confidence: `low`
  - reason: no verified Sedon motor/controller spec was found.
- `R_joint_hip_yaw` effort fallback to `300`
  - source: `assumption`
  - confidence: `low`
  - reason: source URDF has `effort=0` and `velocity=0`.
- right-leg joint name adapter:
  - `R_joint_knee` -> `R_joint_knee_pitch`
  - `R_joint_knee_pitch` -> `R_joint_ankle_pitch`
  - source: `assumption`
  - confidence: `low`
  - reason: existing Sedon env expects knee/ankle pitch names.
- invisible `base_proxy`, `R_foot_collision`, and `L_foot_collision`
  - source: `assumption`
  - confidence: `low`
  - reason: existing eval/contact semantics require these geoms.

## Validation

Run:

```powershell
python -B -m py_compile tools\sedon\geometry\generate_sedon_v5_22_mjcf_scene.py
python -B tools\sedon\geometry\generate_sedon_v5_22_mjcf_scene.py
python -B -m sedon_baseline.eval --episodes 1 --scene-path private_assets\sedon_v5_22\training_scene.xml --record
```

When evaluating old checkpoints, use `--ignore-train-config` if the saved
`train_config.json` points at the archived legacy scene. This preserves old-run
reproducibility while keeping new default Sedon work on `v5_22`.

Current v5_22 eval smoke result:

- model load: passed
- required joint names: found
- required foot/base/floor geoms: found
- actuators: `10`
- visual fidelity: original v5_22 STL meshes are used directly
- walking success: not claimed

The existing checkpoint falls after 40 steps on this scene. Treat this as a
compatibility/visualization result only, not a policy success.
