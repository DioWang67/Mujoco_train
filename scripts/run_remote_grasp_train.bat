@echo off
setlocal
set "REMOTE_HOST=root@10.6.243.55"
set "REMOTE_ROOT=/root/anaconda3/mujoco-train-system"
set "PROJECT_SLUG=grasp"
set "RUN_ROOT=%REMOTE_ROOT%/runs/%PROJECT_SLUG%"
set "LOG_FILE=%RUN_ROOT%/logs/grasp/foreground_train.log"

echo Starting remote grasp training in foreground...
echo Log file: %LOG_FILE%
echo.
ssh -t %REMOTE_HOST% "bash -lc 'mkdir -p %RUN_ROOT%/logs/grasp && cd %REMOTE_ROOT%/code/current && MUJOCO_TRAIN_LAYOUT_ROOT=%REMOTE_ROOT% MUJOCO_TRAIN_PROJECT_SLUG=%PROJECT_SLUG% /root/anaconda3/bin/python -u train.py --grasp %* 2>&1 | tee %RUN_ROOT%/logs/grasp/foreground_train.log'"
