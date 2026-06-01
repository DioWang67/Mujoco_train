@echo off
setlocal
call "%~dp0remote_auto_deploy.bat" --project-slug grasp --verify-project grasp
pause
