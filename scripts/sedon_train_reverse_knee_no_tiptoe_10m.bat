@echo off
cd /d "%~dp0.."
set "SEDON_CONFIG_OVERRIDES=configs\sedon\reverse_knee_no_tiptoe_walk.json"
echo Starting Sedon reverse-knee no-tiptoe walk training (10M steps)...
echo Config: %SEDON_CONFIG_OVERRIDES%
echo.
.\.venv\Scripts\python.exe train.py --project sedon --total-timesteps 10000000 --n-envs 4 --reset-noise-scale 0.01
pause
