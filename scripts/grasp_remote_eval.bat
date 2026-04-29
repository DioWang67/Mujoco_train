@echo off
setlocal
if "%REMOTE_HOST%"=="" set "REMOTE_HOST=root@10.6.243.55"
if "%REMOTE_ROOT%"=="" set "REMOTE_ROOT=/root/anaconda3/mujoco-train-system"
ssh -t %REMOTE_HOST% "bash -lc 'cd %REMOTE_ROOT%/code/current && export MUJOCO_TRAIN_LAYOUT_ROOT=%REMOTE_ROOT% MUJOCO_TRAIN_PROJECT_SLUG=grasp MKL_THREADING_LAYER=GNU OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 && /root/anaconda3/bin/python -m tools.eval_grasp --episodes 10 --no-render'"
pause
