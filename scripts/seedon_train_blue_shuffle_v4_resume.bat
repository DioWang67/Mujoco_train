@echo off
setlocal
cd /d "%~dp0.."

set "PYTHON=python"
if exist ".venv\Scripts\python.exe" set "PYTHON=.venv\Scripts\python.exe"

set "SEEDON_TRAIN_MODE=resume"
set "SEEDON_RESUME_CONFIG=configs/seedon/reference_march_pose_1_4_blue_shuffle_v4.json"
set "SEEDON_RESUME_TOTAL_TIMESTEPS=900000"
set "SEEDON_RESUME_N_ENVS=128"
set "SEEDON_RESUME_RESET_NOISE_SCALE=0.004"
set "SEEDON_RESUME_EXTRA_ARGS=--resume-action-std 0.04"

echo === Seedon Blue-like shuffle v4 remote resume training ===
echo Config    : %SEEDON_RESUME_CONFIG%
echo Timesteps : %SEEDON_RESUME_TOTAL_TIMESTEPS%
echo Envs      : %SEEDON_RESUME_N_ENVS%
echo.

%PYTHON% -m tools.remote_training --project seedon
if errorlevel 1 (
  echo.
  echo Seedon Blue-like shuffle v4 resume training launch failed.
  pause
  exit /b 1
)

echo.
echo Seedon Blue-like shuffle v4 resume training launch completed.
pause
