@echo off
setlocal

cd /d "%~dp0\.."
set "SEDON_CONFIG_OVERRIDES=configs\sedon\autonomy_stage1_teacher_curriculum.json"

.venv\Scripts\python.exe -m sedon_baseline.train ^
  --total-timesteps 100000 ^
  --n-envs 1 ^
  --reset-noise-scale 0 ^
  --resume models\sedon\latest_model.zip ^
  --resume-vecnorm models\sedon\vecnorm.pkl ^
  --resume-action-std 0.1 ^
  --action-std 0.1 ^
  --checkpoint-freq-steps 25000 ^
  --teacher-audit-freq-steps 25000 ^
  --teacher-audit-steps 480 ^
  --teacher-baseline-config configs\sedon\reference_teacher_pose_1_4_imitation.json ^
  --pose-weight-schedule 25000:6,50000:4

endlocal
