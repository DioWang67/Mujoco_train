@echo off
setlocal
call "%~dp0run_remote_train.bat" grasp --phase full --n-envs 8
pause
