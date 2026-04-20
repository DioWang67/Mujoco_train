# Training Runbook (Recommended Order)

Date: 2026-04-20

## 0) Preflight
```bash
python preflight_check.py
```
Expected: `Preflight PASSED`.

## 1) Smoke test (always first)
```bash
python train.py --smoke
python train.py --smoke --dr --dr-start-level 0.0 --dr-ramp-end 0.35
```
Purpose: quickly verify code/env before long runs.

## 2) Base training (mainline)
```bash
python train.py
```
Default is 40M steps.

## 3) Base eval + sanity metrics
```bash
python eval.py --episodes 3 --no-render
python compare_eval.py --episodes 8 --out-json compare_report.json --out-csv compare_report.csv
python gate_check.py --report compare_report.json --gates gate_profiles.json --mode single --profile preprod
```

## 4) DR fine-tune from best base checkpoint
```bash
python train.py --finetune models/best_model.zip --dr --dr-start-level 0.05 --dr-ramp-end 0.7
```

## 5) DR aggregate validation (multi-seed)
```bash
python aggregate_compare.py --seeds 3 --seed-start 42 --episodes 5 --out-json aggregate_compare.json --out-csv aggregate_compare.csv
python gate_check.py --report aggregate_compare.json --gates gate_profiles.json --mode aggregate --profile preprod
```

## 6) Benchmark matrix report
```bash
python benchmark_matrix.py --matrix benchmark_matrix.json --out-json benchmark_report.json --out-csv benchmark_report.csv
```

## 7) Continue training strategy
- If gates pass and curve still rising slowly:
  - Continue DR finetune with `--resume --dr`.
- If gates fail on stability (CI too large):
  - increase seeds/episodes and reduce DR ramp aggressiveness.

Example continue command:
```bash
python train.py --resume --dr --dr-start-level 0.05 --dr-ramp-end 0.7
```

## Windows .bat shortcuts
- `fresh_train.bat`: base training
- `dr_train.bat`: DR training/ramp defaults
- `finetune_dr.bat`: DR finetune from best base
- `eval.bat`: eval/compare/aggregate/gate/benchmark menu
