@echo off
cd /d "%~dp0.."
python -m tools.prepare_package
pause
