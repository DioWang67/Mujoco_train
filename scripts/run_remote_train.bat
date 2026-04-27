@echo off
setlocal
set "TARGET=%~1"
if "%TARGET%"=="" (
  echo Usage: scripts\run_remote_train.bat ^<project-slug^> [train args...]
  exit /b 1
)
if "%REMOTE_HOST%"=="" set "REMOTE_HOST=root@10.6.243.55"
if "%REMOTE_ROOT%"=="" set "REMOTE_ROOT=/root/anaconda3/mujoco-train-system"
set "PROJECT_SLUG=%TARGET%"
set "RUN_ROOT=%REMOTE_ROOT%/runs/%PROJECT_SLUG%"
set "LOG_FILE=%RUN_ROOT%/logs/%PROJECT_SLUG%/foreground_train.log"

set "FORWARDED_ARGS="
shift /1
:collect_args
if "%~1"=="" goto run_target
set FORWARDED_ARGS=%FORWARDED_ARGS% %1
shift /1
goto collect_args

:run_target
echo Starting remote training for project=%PROJECT_SLUG% in foreground...
echo Log file: %LOG_FILE%
echo.
ssh -t %REMOTE_HOST% "bash -lc 'mkdir -p %RUN_ROOT%/logs/%PROJECT_SLUG% && cd %REMOTE_ROOT%/code/current && MUJOCO_TRAIN_LAYOUT_ROOT=%REMOTE_ROOT% MUJOCO_TRAIN_PROJECT_SLUG=%PROJECT_SLUG% /root/anaconda3/bin/python -u train.py --project %PROJECT_SLUG% %FORWARDED_ARGS% 2>&1 | tee %LOG_FILE%'"
exit /b %errorlevel%
