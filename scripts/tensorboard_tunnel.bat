@echo off
setlocal
set "REMOTE_HOST=root@10.6.243.55"
set "REMOTE_ROOT=/root/anaconda3/mujoco-train-system"
set "PROJECT_SLUG=%~1"
if "%PROJECT_SLUG%"=="" set "PROJECT_SLUG=h1"
set "JOB_NAME=%~2"
if "%JOB_NAME%"=="" set "JOB_NAME=%PROJECT_SLUG%"
set "PORT=%~3"
if "%PORT%"=="" (
  if /I "%PROJECT_SLUG%"=="grasp" (
    set "PORT=6007"
  ) else (
    set "PORT=6006"
  )
)
set "REMOTE_LOG=/tmp/tb_%PROJECT_SLUG%_%JOB_NAME%_%PORT%.log"
echo Starting TensorBoard on remote for project=%PROJECT_SLUG% job=%JOB_NAME%...
ssh %REMOTE_HOST% "bash %REMOTE_ROOT%/code/current/scripts/start_remote_tensorboard.sh %REMOTE_ROOT% %PROJECT_SLUG% %JOB_NAME% %PORT%"
if errorlevel 1 (
  echo Remote TensorBoard failed to start.
  pause
  exit /b 1
)
echo Opening tunnel: localhost:%PORT% -^> remote:%PORT%
echo Then open: http://localhost:%PORT%
echo Remote log: %REMOTE_LOG%
echo.
ssh -N -L %PORT%:127.0.0.1:%PORT% %REMOTE_HOST%
pause
