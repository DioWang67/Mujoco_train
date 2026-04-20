@echo off
echo Starting TensorBoard on remote...
ssh root@10.6.243.55 "nohup /root/anaconda3/bin/tensorboard --logdir /root/anaconda3/h1_package/code/logs/tb --port 6006 --host 127.0.0.1 > /tmp/tb.log 2>&1 &"
echo Opening tunnel: localhost:6006 -^> remote:6006
echo Then open: http://localhost:6006
echo.
ssh -L 6006:localhost:6006 root@10.6.243.55
pause
