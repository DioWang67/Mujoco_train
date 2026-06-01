@echo off
setlocal
cd /d "%~dp0.."

set "PYTHON=python"
if exist ".venv\Scripts\python.exe" set "PYTHON=.venv\Scripts\python.exe"

set "SEDON_TRAIN_MODE=fresh"
set "SEDON_RESUME_CONFIG=configs/sedon/reference_march_pose_1_4_blue_shuffle_v2.json"
set "SEDON_RESUME_TOTAL_TIMESTEPS=1200000"
set "SEDON_RESUME_N_ENVS=128"
set "SEDON_RESUME_RESET_NOISE_SCALE=0.008"
set "SEDON_RESUME_EXTRA_ARGS=--action-std 0.08"

echo === Sedon Blue-like shuffle v2 remote training ===
echo Config    : %SEDON_RESUME_CONFIG%
echo Timesteps : %SEDON_RESUME_TOTAL_TIMESTEPS%
echo Envs      : %SEDON_RESUME_N_ENVS%
echo.

%PYTHON% -m tools.remote_training --project sedon
if errorlevel 1 (
  echo.
  echo Sedon Blue-like shuffle v2 training launch failed.
  pause
  exit /b 1
)

echo.
echo Sedon Blue-like shuffle v2 training launch completed.
pause
