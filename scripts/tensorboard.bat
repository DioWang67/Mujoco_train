@echo off
cd /d "%~dp0.."
echo Starting TensorBoard...
echo Open browser: http://localhost:6006
echo Press Ctrl+C to stop.
echo.
python -m tensorboard.main --logdir logs\tb
pause
