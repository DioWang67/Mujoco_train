# Sedon Parameter Workflow

This workflow captures robot parameters for Sedon and Open Duck Mini without changing training logic, experiments, or artifacts.

## 1. Extract Sedon Parameters

```powershell
python -m tools.sedon.extractors.extract_sedon_parameters
```

Default inputs and outputs:

| Item | Path |
|---|---|
| source scene | `private_assets/sedon/training_scene.xml` |
| YAML snapshot | `configs/sedon/sedon_robot_parameters.yaml` |
| report | `docs/sedon_parameter_index.md` |

The extractor parses explicit MJCF XML fields only. Missing attributes are written as `null`.

## 2. Extract Duck Parameters

```powershell
python -m tools.sedon.extractors.extract_duck_parameters --duck-xml <path-to-duck-mujoco.xml>
```

Default outputs:

| Item | Path |
|---|---|
| YAML snapshot | `references/open_duck_mini/duck_robot_parameters.yaml` |
| report | `references/open_duck_mini/duck_extraction_report.md` |
| source manifest | `references/open_duck_mini/source_manifest.yaml` |

If the Duck XML path is missing, the tool reports the missing path cleanly and exits without a traceback.

## 3. Compare Sedon vs Duck

```powershell
python -m tools.sedon.extractors.compare_sedon_duck_parameters
```

Default inputs and output:

| Item | Path |
|---|---|
| Sedon YAML | `configs/sedon/sedon_robot_parameters.yaml` |
| Duck YAML | `references/open_duck_mini/duck_robot_parameters.yaml` |
| comparison report | `docs/sedon_duck_comparison.md` |

Run this only after both YAML snapshots exist.

## 4. Fields That Are Reference-Only

These fields must not be directly copied from Duck into Sedon:

- joint names, because naming and side conventions may differ;
- joint axis and range, because sign conventions and morphology differ;
- actuator `kp`, `ctrlrange`, and `forcerange`, because actuator semantics differ by model;
- foot geom size/position, because contact patch design and solver behavior differ;
- gait period, clearance, and velocity targets, because they need Sedon-specific stability validation.

Use Duck values as reference targets for analysis, not as direct Sedon controls or PPO reward constants.

## 5. Semantic Mapping

The first semantic joint mapping lives at:

```text
references/open_duck_mini/sedon_duck_joint_mapping.yaml
```

Notes and transfer limits are documented in:

```text
docs/sedon_duck_mapping_notes.md
```

This mapping aligns leg joints by semantic role only. It does not validate sign, gain, amplitude, or policy action compatibility.

## 6. Known Unknowns

- XML-only extraction does not infer MuJoCo defaults or compiled inertia.
- Leg length and world-frame foot scale are not computed in this first pass.
- Foot-related geoms are selected with a conservative name heuristic.
- Contact semantics still require simulation diagnostics before training decisions.
