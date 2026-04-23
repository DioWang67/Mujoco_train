@echo off
setlocal
set "REMOTE_HOST=root@10.6.243.55"
set "REMOTE_ROOT=/root/anaconda3/mujoco-train-system"
set "PROJECT_SLUG=grasp"
set "LOG_FILE=%REMOTE_ROOT%/projects/%PROJECT_SLUG%/runs/logs/grasp/foreground_train.log"

echo Starting remote grasp training in foreground...
echo Log file: %LOG_FILE%
echo.
ssh -t %REMOTE_HOST% "bash -lc 'mkdir -p %REMOTE_ROOT%/projects/%PROJECT_SLUG%/runs/logs/grasp && cd %REMOTE_ROOT%/projects/%PROJECT_SLUG%/current && /root/anaconda3/bin/python -u train.py --grasp %* 2>&1 | tee %LOG_FILE%'"
