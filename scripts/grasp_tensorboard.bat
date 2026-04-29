@echo off
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_remote_tensorboard.ps1" -ProjectSlug grasp -JobName grasp -Port 6007 -LatestRun
pause
