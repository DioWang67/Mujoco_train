@echo off
setlocal
call "%~dp0remote_auto_deploy.bat" --project-slug h1 --verify-project h1
pause
