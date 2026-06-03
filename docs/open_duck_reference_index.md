# Open Duck Mini Reference Index

This directory stores Open Duck Mini reference parameter snapshots for Seedon gait research.

## Files

| Path | Purpose |
|---|---|
| `references/open_duck_mini/source_manifest.yaml` | Records the Duck XML source path, extraction time, tool version, and notes. |
| `references/open_duck_mini/duck_robot_parameters.yaml` | Generated Duck parameter snapshot. Created only after a valid Duck XML is supplied. |
| `references/open_duck_mini/duck_extraction_report.md` | Generated human-readable extraction report. Created only after a valid Duck XML is supplied. |
| `references/open_duck_mini/source/playground_open_duck_mini_v2/xmls/` | Copied Open Duck Mini v2 MuJoCo XML package and lightweight mesh assets. |
| `references/open_duck_mini/seedon_duck_joint_mapping.yaml` | Semantic Seedon/Duck leg-joint mapping for reference analysis. |
| `docs/seedon_duck_mapping_notes.md` | Human-readable notes and transfer limits for the semantic mapping. |

## Current State

Open Duck Mini v2 XML was copied from:

```text
C:\Users\diowang\open_duck_mini_ws\Open_Duck_Playground\playground\open_duck_mini_v2\xmls
```

Current extracted source:

```text
references/open_duck_mini/source/playground_open_duck_mini_v2/xmls/open_duck_mini_v2.xml
```

## Extraction Command

```powershell
python -m tools.seedon.extractors.extract_duck_parameters --duck-xml <path-to-duck-mujoco.xml>
```

The extractor prints a clear error and returns a non-zero exit code if the XML path does not exist.

## Use Limits

Duck data is reference material only. Do not directly copy Duck joint targets, action scales, actuator gains, or contact assumptions into Seedon without a separate validation phase.
