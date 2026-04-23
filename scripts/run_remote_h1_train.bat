@echo off
setlocal
set "REMOTE_HOST=root@10.6.243.55"
set "REMOTE_ROOT=/root/anaconda3/mujoco-train-system"
set "PROJECT_SLUG=h1"
set "LOG_FILE=%REMOTE_ROOT%/projects/%PROJECT_SLUG%/runs/logs/h1/foreground_train.log"

echo Starting remote H1 training in foreground...
echo Log file: %LOG_FILE%
echo.
ssh -t %REMOTE_HOST% "bash -lc 'mkdir -p %REMOTE_ROOT%/projects/%PROJECT_SLUG%/runs/logs/h1 && cd %REMOTE_ROOT%/projects/%PROJECT_SLUG%/code/current && /root/anaconda3/bin/python -u train.py --h1 %* 2>&1 | tee %LOG_FILE%'"
