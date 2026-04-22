# Env Count Tuning & Next-Phase Meaning

Date: 2026-04-20

## Q1: 要不要設 `H1_N_ENVS=128`？
**不一定。** `128` 只在 CPU 核心數、RAM、進程調度都足夠時才有機會更快；
如果資源不夠，反而會讓 FPS 下降、context switch 增加、訓練變慢。

### 建議起手式
1. 先用 `H1_N_ENVS=32` 跑 10~20 分鐘，記錄 FPS。
2. 依序測 `64`、`96`、`128`，每次同條件比較 FPS 與穩定性。
3. 選 **FPS 高且最穩定** 的值，不是盲目選最大值。

### 參考區間（經驗值）
- 16~32 vCPU：`H1_N_ENVS=16~48`
- 48~64 vCPU：`H1_N_ENVS=48~96`
- 96+ vCPU：`H1_N_ENVS=96~160`

## Q2: 下個階段是什麼意思？
這裡的「下個階段」是從**現在可訓練/可評估的 preprod 流程**，走到**可持續發布**。

### Phase A（你現在）— Preprod baseline
- 可做：preflight、base/DR 訓練、compare、aggregate、gate、benchmark。
- 目標：把模型品質變成可量化與可重現。

### Phase B（下一步）— Gate-enforced CI
- 把 `tools/gate_check.py` 接進 CI：gate fail 就不允許合併/發布。
- 目標：流程自動化，減少人工判斷誤差。

### Phase C（再下一步）— Terrain matrix + Sim2Real contract
- 新增 terrain variants（heightfield/ramp/step）到 benchmark matrix。
- 定義 sim-to-real 介面契約（obs/action schema、時序與安全策略）。
- 目標：讓結果可對接內部機器人部署團隊。

## 常用指令
### Linux/macOS
```bash
export H1_N_ENVS=64
python train.py --dr --dr-start-level 0.05 --dr-ramp-end 0.7
```

### Windows (PowerShell)
```powershell
$env:H1_N_ENVS = "64"
python train.py --dr --dr-start-level 0.05 --dr-ramp-end 0.7
```

### Windows (cmd)
```cmd
set H1_N_ENVS=64
python train.py --dr --dr-start-level 0.05 --dr-ramp-end 0.7
```
