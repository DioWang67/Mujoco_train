@echo off
setlocal

cd /d "%~dp0\.."
set "SEEDON_CONFIG_OVERRIDES=configs\seedon\reference_teacher_pose_1_4_imitation.json"

.venv\Scripts\python.exe -m seedon_baseline.train ^
  --total-timesteps 100000 ^
  --n-envs 1 ^
  --reset-noise-scale 0 ^
  --resume models\seedon\teacher_audit\teacher_audit_100000_steps.zip ^
  --resume-vecnorm models\seedon\teacher_audit\teacher_audit_100000_steps_vecnorm.pkl ^
  --resume-action-std 0.05 ^
  --action-std 0.05 ^
  --checkpoint-freq-steps 25000 ^
  --teacher-audit-freq-steps 25000 ^
  --teacher-audit-steps 480

endlocal
