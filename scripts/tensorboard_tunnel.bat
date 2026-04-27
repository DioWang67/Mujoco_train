@echo off
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_remote_tensorboard.ps1" -ProjectSlug "%~1" -JobName "%~2" -Port "%~3"
pause
