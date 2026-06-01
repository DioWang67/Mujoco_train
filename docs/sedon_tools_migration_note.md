# Sedon Tools Migration Note

## 2026-06-01 - Phase 1 Contact Diagnostic Move

Task class: Class C, tooling organization. This phase only moved active/stable contact diagnostic implementations and kept legacy wrappers. It did not change training logic, rewards, scenes, or artifacts.

## Moved Implementations

| Legacy command/import path | New implementation path | Compatibility |
|---|---|---|
| `tools.debug_sedon_contacts` | `tools.sedon.diagnostics.contact.debug_sedon_contacts` | old module remains as wrapper |
| `tools.sedon_phase2c_contact_constrained_foot_mapping` | `tools.sedon.diagnostics.contact.phase2c_contact_constrained_foot_mapping` | old module remains as wrapper |
| `tools.sedon_phase_g1_raw_contact_pair_diagnostic` | `tools.sedon.diagnostics.contact.phase_g1_raw_contact_pair_diagnostic` | old module remains as wrapper |

## Wrapper Policy

The old top-level modules must remain available for at least one migration cycle because they are referenced by existing docs, progress logs, registry examples, and ad-hoc commands:

```text
python -m tools.debug_sedon_contacts
python -m tools.sedon_phase2c_contact_constrained_foot_mapping
python -m tools.sedon_phase_g1_raw_contact_pair_diagnostic
```

New code may import the implementation modules directly:

```text
tools.sedon.diagnostics.contact.debug_sedon_contacts
tools.sedon.diagnostics.contact.phase2c_contact_constrained_foot_mapping
tools.sedon.diagnostics.contact.phase_g1_raw_contact_pair_diagnostic
```

## Scope Exclusions

This migration intentionally did not move:

- experimental contact sweeps;
- geometry generation tools;
- load-transfer diagnostics;
- archived phase tools;
- artifacts under `artifacts/sedon_debug/`;
- any training/evaluation logic under `sedon_baseline/`.

## Validation

Required smoke check for this phase:

```text
import tools.debug_sedon_contacts
import tools.sedon_phase2c_contact_constrained_foot_mapping
import tools.sedon_phase_g1_raw_contact_pair_diagnostic
import tools.sedon.diagnostics.contact.debug_sedon_contacts
import tools.sedon.diagnostics.contact.phase2c_contact_constrained_foot_mapping
import tools.sedon.diagnostics.contact.phase_g1_raw_contact_pair_diagnostic
```
