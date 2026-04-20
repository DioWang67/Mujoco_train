@echo off
cd /d %~dp0
echo H1 Eval Options:
echo   1. Normal eval (base policy)
echo   2. DR eval (domain randomization)
echo   3. Record video (base policy)
echo   4. Record video (DR)
echo.
set /p choice="Select (1-4): "

if "%choice%"=="1" python eval.py
if "%choice%"=="2" python eval.py --dr
if "%choice%"=="3" python eval.py --record
if "%choice%"=="4" python eval.py --record --dr
pause
