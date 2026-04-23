@echo off
set "REMOTE_HOST=root@10.6.243.55"
set "REMOTE_ROOT=/root/anaconda3/mujoco-train-system"
set "PROJECT_SLUG=h1"
set "JOB_NAME=%~1"
if "%JOB_NAME%"=="" set "JOB_NAME=h1"
set "PORT=%~2"
if "%PORT%"=="" set "PORT=6006"
set "LOGDIR=%REMOTE_ROOT%/projects/%PROJECT_SLUG%/runs/logs/tb/%JOB_NAME%"
set "REMOTE_LOG=/tmp/tb_%PROJECT_SLUG%_%JOB_NAME%_%PORT%.log"
echo Starting TensorBoard on remote for %JOB_NAME%...
ssh %REMOTE_HOST% "nohup /root/anaconda3/bin/tensorboard --logdir %LOGDIR% --port %PORT% --host 127.0.0.1 > %REMOTE_LOG% 2>&1 &"
echo Opening tunnel: localhost:%PORT% -^> remote:%PORT%
echo Then open: http://localhost:%PORT%
echo Remote log: %REMOTE_LOG%
echo.
ssh -L %PORT%:127.0.0.1:%PORT% %REMOTE_HOST%
pause
