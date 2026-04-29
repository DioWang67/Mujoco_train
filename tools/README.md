# Tools Index

Run this to print the same index from the command line:

```bash
python -m tools
```

## Checks

- `python -m tools.preflight_check`  
  Check local runtime prerequisites before training.

## Evaluation

- `python -m tools.compare_eval`  
  Compare base and DR H1 policies.
- `python -m tools.aggregate_compare`  
  Run multi-seed H1 comparison and 95% confidence intervals.
- `python -m tools.benchmark_matrix`  
  Run configured H1 benchmark scenarios.
- `python -m tools.gate_check`  
  Validate reports against gate profiles.
- `python -m tools.plot_eval`  
  Plot evaluation CSV files.

## Grasp

- `python -m tools.eval_grasp`  
  Evaluate a trained fixed-base grasp checkpoint.
- `python -m tools.grasp_sanity_check`  
  Run a scripted grasp rollout to verify reset/controller setup.

## Sedon

- `python -m tools.convert_urdf_to_mjcf`  
  Convert the private Sedon URDF/STL package into a MuJoCo MJCF scene.
- `python -m tools.build_sedon_training_scene`  
  Build the floating-base Sedon training scene from converted MJCF.
- `python -m tools.smoke_sedon_env --steps 20`  
  Run a short Sedon standing environment smoke test.

## Release

- `python -m tools.deploy_release`  
  Create and optionally upload a clean source release archive.
- `python -m tools.prepare_package`  
  Build an offline dependency/source bundle for a remote host.

## Maintenance

- `python -m tools.download_cuda_deps`  
  Download CUDA runtime Python packages for remote repair.
- `python -m tools.download_missing`  
  Download missing Python packages for offline install.
- `python -m tools.fix_cusparselt`  
  Prepare cuSPARSELt repair instructions/assets.

## Experiments

- `python -m tools.sweep`  
  Run Optuna sweeps for H1 training parameters.

## Rule of Thumb

- Put reusable Python CLIs here.
- Keep wrapper-only commands in `scripts/`.
- Avoid importing simulator/training dependencies at module import time unless
  the command truly needs them immediately.
