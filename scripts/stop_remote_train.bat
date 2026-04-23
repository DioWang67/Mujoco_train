@echo off
setlocal
set "REMOTE_HOST=root@10.6.243.55"
set "TARGET=%~1"
if "%TARGET%"=="" set "TARGET=grasp"

if /I "%TARGET%"=="grasp" (
  set "PATTERN=train.py --project grasp"
) else (
  set "PATTERN=train.py --project h1"
)

echo Stopping remote training processes matching: %PATTERN%
ssh %REMOTE_HOST% "pkill -9 -f '%PATTERN%' || true"
