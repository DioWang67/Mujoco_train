@echo off
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install_vscode_server_offline.ps1" %*
pause
