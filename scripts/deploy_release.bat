@echo off
cd /d "%~dp0.."
python -m tools.deploy_release %*
