@echo off
cd /d %~dp0
echo Fine-tuning with Domain Randomization from pre-trained model...
echo   Source model : models\best_model.zip  (peak performance checkpoint)
echo   VecNormalize : fresh (DR changes observation distribution)
echo   DR enabled   : friction, mass, noise, delay, command randomization
echo   Curriculum   : 0.3 m/s -> 1.0 m/s
echo   LR           : 1e-4 -> 0 (linear decay, lower than base to protect base policy)
echo   ent_coef     : 0.02 (higher than base for DR adaptation)
echo.
echo Note: Policy weights are reused, VecNormalize starts fresh.
echo       Expect faster convergence than training from scratch.
echo.
python train.py --finetune models\best_model.zip --dr
pause
