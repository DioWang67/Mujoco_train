@echo off
setlocal

if "%REMOTE_HOST%"=="" set "REMOTE_HOST=root@10.6.243.55"
if "%REMOTE_ROOT%"=="" set "REMOTE_ROOT=/root/anaconda3/mujoco-train-system"

set "PROJECT_SLUG=sedon"
set "RUN_ROOT=%REMOTE_ROOT%/runs/%PROJECT_SLUG%"
set "LOG_FILE=%RUN_ROOT%/logs/%PROJECT_SLUG%/reverse_knee_no_tiptoe_10m.log"
set "OVERRIDE_PATH=configs/sedon/reverse_knee_no_tiptoe_walk.json"

echo Starting remote Sedon training: reverse-knee no-tiptoe walk, 10M steps
echo Host: %REMOTE_HOST%
echo Root: %REMOTE_ROOT%
echo Override: %OVERRIDE_PATH%
echo Log file: %LOG_FILE%
echo.

ssh -t %REMOTE_HOST% "bash -lc 'mkdir -p %RUN_ROOT%/logs/%PROJECT_SLUG% && cd %REMOTE_ROOT%/code/current && export MUJOCO_TRAIN_LAYOUT_ROOT=%REMOTE_ROOT% MUJOCO_TRAIN_PROJECT_SLUG=%PROJECT_SLUG% SEDON_CONFIG_OVERRIDES=%OVERRIDE_PATH% MKL_THREADING_LAYER=GNU OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 && /root/anaconda3/bin/python -u train.py --project %PROJECT_SLUG% --total-timesteps 10000000 --n-envs 4 --reset-noise-scale 0.01 2>&1 | tee %LOG_FILE%'"
exit /b %errorlevel%
