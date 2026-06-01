@echo off
setlocal

cd /d "%~dp0\.."
set "SEDON_CONFIG_OVERRIDES=configs\sedon\teacher_imitation_impact_tuned.json"

.venv\Scripts\python.exe -m sedon_baseline.train ^
  --total-timesteps 100000 ^
  --n-envs 1 ^
  --reset-noise-scale 0 ^
  --resume models\sedon\teacher_audit\teacher_audit_25000_steps.zip ^
  --resume-vecnorm models\sedon\teacher_audit\teacher_audit_25000_steps_vecnorm.pkl ^
  --resume-action-std 0.05 ^
  --action-std 0.05 ^
  --checkpoint-freq-steps 25000 ^
  --teacher-audit-freq-steps 25000 ^
  --teacher-audit-steps 480

endlocal
