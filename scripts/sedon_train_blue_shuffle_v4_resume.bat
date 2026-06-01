@echo off
setlocal
cd /d "%~dp0.."

set "PYTHON=python"
if exist ".venv\Scripts\python.exe" set "PYTHON=.venv\Scripts\python.exe"

set "SEDON_TRAIN_MODE=resume"
set "SEDON_RESUME_CONFIG=configs/sedon/reference_march_pose_1_4_blue_shuffle_v4.json"
set "SEDON_RESUME_TOTAL_TIMESTEPS=900000"
set "SEDON_RESUME_N_ENVS=128"
set "SEDON_RESUME_RESET_NOISE_SCALE=0.004"
set "SEDON_RESUME_EXTRA_ARGS=--resume-action-std 0.04"

echo === Sedon Blue-like shuffle v4 remote resume training ===
echo Config    : %SEDON_RESUME_CONFIG%
echo Timesteps : %SEDON_RESUME_TOTAL_TIMESTEPS%
echo Envs      : %SEDON_RESUME_N_ENVS%
echo.

%PYTHON% -m tools.remote_training --project sedon
if errorlevel 1 (
  echo.
  echo Sedon Blue-like shuffle v4 resume training launch failed.
  pause
  exit /b 1
)

echo.
echo Sedon Blue-like shuffle v4 resume training launch completed.
pause
