@echo off
cd /d "%~dp0.."
echo Starting H1 training (resume from latest checkpoint)...
python train.py --resume
pause
