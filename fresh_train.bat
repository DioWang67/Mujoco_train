@echo off
cd /d %~dp0
echo Starting fresh H1 training (20M steps)...
echo.
python train.py
pause
