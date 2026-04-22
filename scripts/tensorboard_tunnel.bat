@echo off
set "REMOTE_HOST=root@10.6.243.55"
set "REMOTE_ROOT=/root/mujoco-train-system"
set "PROJECT_SLUG=h1"
echo Starting TensorBoard on remote...
ssh %REMOTE_HOST% "nohup /root/anaconda3/bin/tensorboard --logdir %REMOTE_ROOT%/projects/%PROJECT_SLUG%/current/logs/tb --port 6006 --host 127.0.0.1 > /tmp/tb.log 2>&1 &"
echo Opening tunnel: localhost:6006 -^> remote:6006
echo Then open: http://localhost:6006
echo.
ssh -L 6006:localhost:6006 %REMOTE_HOST%
pause
