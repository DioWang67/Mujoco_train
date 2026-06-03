@echo off
setlocal

cd /d "%~dp0\.."
set "SEEDON_CONFIG_OVERRIDES=configs\seedon\reference_teacher_pose_1_4_imitation.json"

.venv\Scripts\python.exe -m seedon_baseline.train ^
  --total-timesteps 25000 ^
  --n-envs 1 ^
  --reset-noise-scale 0 ^
  --resume models\seedon\teacher_safe_baseline\model.zip ^
  --resume-vecnorm models\seedon\teacher_safe_baseline\vecnorm.pkl ^
  --resume-action-std 0.05 ^
  --action-std 0.05 ^
  --checkpoint-freq-steps 25000 ^
  --teacher-audit-freq-steps 25000 ^
  --teacher-audit-steps 480 ^
  --teacher-audit-warmup-steps 20 ^
  --teacher-baseline-config configs\seedon\reference_teacher_pose_1_4_imitation.json

if errorlevel 1 exit /b %errorlevel%

.venv\Scripts\python.exe -m tools.autonomy_stage1_probe_report ^
  --teacher-config configs\seedon\reference_teacher_pose_1_4_imitation.json ^
  --probe-config configs\seedon\reference_teacher_pose_1_4_imitation.json ^
  --checkpoint models\seedon\latest_model.zip ^
  --vecnorm-path models\seedon\vecnorm.pkl ^
  --steps 480 ^
  --seed 42 ^
  --audit-warmup-steps 20 ^
  --out-csv artifacts\seedon_debug\autonomy_probe_v2_25k_report.csv

endlocal
