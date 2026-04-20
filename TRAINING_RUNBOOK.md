# H1 MuJoCo 訓練操作手冊

## 目錄
1. [環境確認](#1-環境確認)
2. [標準訓練流程](#2-標準訓練流程)
3. [Domain Randomization 流程](#3-domain-randomization-流程)
4. [評估與品質驗證](#4-評估與品質驗證)
5. [TensorBoard 監控](#5-tensorboard-監控)
6. [模型檔案說明](#6-模型檔案說明)
7. [常見問題](#7-常見問題)

---

## 1. 環境確認

訓練前先執行預檢，確認 GPU、套件、模型檔案都就緒：

```bash
python preflight_check.py
```

全部顯示 `[OK]` 才繼續。有 `[FAIL]` 先排除再訓練。

---

## 2. 標準訓練流程

### 2.1 全新訓練（40M steps，non-DR）

```bash
python train.py
```

- 使用 32 個並行環境（可用 `H1_N_ENVS=4` 環境變數覆蓋，適合資源少的機器）
- 訓練完成輸出：
  - `models/h1_ppo.zip`（最終模型）
  - `models/h1_vecnorm.pkl`（VecNormalize 統計）
  - `models/best_model.zip`（EvalCallback 選出的最佳 policy）
  - `models/h1_vecnorm_best.pkl`（對應 best_model 的 VecNorm）

### 2.2 快速驗證（不燒完整資源）

```bash
python train.py --smoke   # 5k steps，確認程式能跑
python train.py --quick   # 300k steps，4 個環境，觀察 reward 趨勢
```

`--smoke` 不觸發 EvalCallback，也不會產生 `best_model.zip`（設計如此）。

### 2.3 從斷點續訓

```bash
python train.py --resume         # 接 non-DR 最新 step checkpoint
python train.py --resume --dr    # 接 DR 最新 step checkpoint
```

Resume 邏輯（[train.py:_find_latest_checkpoint](train.py)）：
1. 優先找符合模式（DR/base）的最大 step 數 checkpoint（`h1_ppo_XXXXX_steps.zip` 或 `h1_ppo_dr_XXXXX_steps.zip`）
2. 找不到 → fallback 到對應模式的 final 檔（`h1_ppo.zip` 或 `h1_ppo_dr.zip`）
3. 再找不到 → 才用 `best_model.zip`（會印警告，因為這是 EvalCallback 選的不一定是最新的）

---

## 3. Domain Randomization 流程

### 推薦順序：先跑基礎訓練，再做 DR finetune

```
Step 1: 基礎訓練（non-DR）→ 產生 best_model.zip
Step 2: DR Finetune（從 best_model.zip 出發）
```

### Step 1：基礎訓練

```bash
python train.py
```

完成後確認 `models/best_model.zip` 存在。

### Step 2：DR Finetune

```bash
python train.py --finetune models/best_model.zip --dr
```

- 自動啟用 Curriculum Learning（target velocity：0.2 → 0.5 → 0.8 m/s）
- 自動啟用 DR Ramp（DR 強度從 start_level 漸進到 1.0）
- 使用較小 LR（1e-4 → 1e-5）與較高 ent_coef（0.02）保護 base policy
- 輸出：
  - `models/h1_ppo_dr.zip`（DR finetune 最終模型，**注意是 `_dr` 後綴**）
  - `models/h1_vecnorm_dr.pkl`（DR-adapted VecNorm 統計）

> **注意**：DR finetune 必須跑完（40M steps），中途 Ctrl+C 雖然現在會存下 partial model，但 policy 在 DR eval 下仍容易跌倒。建議跑完整流程。

### 從零開始 DR 訓練（不推薦，收斂較慢）

```bash
python train.py --dr
```

輸出一樣是 `models/h1_ppo_dr.zip` + `models/h1_vecnorm_dr.pkl`（model 和 VecNorm 配對一致）。

### DR 強度調整（進階）

```bash
python train.py --finetune models/best_model.zip --dr \
    --dr-start-level 0.05 \
    --dr-ramp-end 0.7
```

| 參數 | 預設 | 說明 |
|------|------|------|
| `--dr-start-level` | `0.0` | 起始 DR 強度（0=幾乎無擾動，1=滿強度）|
| `--dr-ramp-end` | `0.35` | progress 到多少時 DR 到達滿強度 |

建議值：
- **Finetune**（已有 base policy）：`--dr-start-level 0.05 --dr-ramp-end 0.7`，強度緩升保護 base
- **Fresh DR**（從零）：`--dr-start-level 0.0 --dr-ramp-end 0.35`，較早達到滿強度

---

## 4. 評估與品質驗證

### 4.1 基本 Eval

```bash
# 看即時畫面（base policy）
python eval.py

# DR 環境下評估（優先順序：dr_best/best_model.zip → h1_ppo_dr.zip → 其他 fallback）
python eval.py --dr

# 自動偵測：有 DR artifact 就自動開 DR 模式
python eval.py --auto-dr

# 記錄影片
python eval.py --record
python eval.py --record --dr

# 多回合 + 存 CSV（含每步 joint pos/vel/torque）
python eval.py --episodes 5 --log

# 指定目標速度
python eval.py --vel 0.8
```

### 4.2 Base vs DR 數值比較

```bash
python compare_eval.py --episodes 8 --vel 1.0 \
    --out-json compare_report.json \
    --out-csv compare_report.csv
```

輸出 base/DR 的 reward、episode length、x 方向速度統計，以及兩者差異（delta）。

### 4.3 品質門檻驗證（Gate Check）

Gate check 有兩種模式：

**A. 單一規則（`release_gates.json`）**

```bash
python gate_check.py --report compare_report.json --gates release_gates.json
```

**B. Profile 模式（`gate_profiles.json`，分三層嚴格度）**

```bash
# 研究階段（最寬鬆）
python gate_check.py --report compare_report.json \
    --gates gate_profiles.json --profile research

# 預上線（標準）
python gate_check.py --report compare_report.json \
    --gates gate_profiles.json --profile preprod

# 正式發布（最嚴格）
python gate_check.py --report compare_report.json \
    --gates gate_profiles.json --profile release
```

門檻意義（`release_gates.json` 預設）：

| 指標 | 門檻 | 意義 |
|------|------|------|
| `dr_len_mean_min` | ≥ 300 | DR 環境下平均撐多少 step 才不算跌倒 |
| `dr_reward_mean_min` | ≥ 150 | DR reward 平均下限 |
| `dr_delta_reward_mean_min` | ≥ -120 | base→DR reward 衰退不超過 120 |
| `dr_xvel_mean_min` | ≥ 0.6 | 前進速度至少達成 60% 目標 |

### 4.4 多 seed 聚合（帶 95% 信賴區間）

```bash
python aggregate_compare.py --seeds 3 --seed-start 42 --episodes 5 --vel 1.0 \
    --out-json aggregate_compare.json \
    --out-csv aggregate_compare.csv

# 對聚合結果跑 gate check
python gate_check.py --report aggregate_compare.json \
    --gates gate_profiles.json --profile preprod --mode aggregate
```

> `--mode` 可省略（預設 `auto`，會依 report 內容自動判斷 single / aggregate）

### 4.5 Benchmark Matrix（多場景壓測）

```bash
python benchmark_matrix.py \
    --matrix benchmark_matrix.json \
    --out-json benchmark_report.json \
    --out-csv benchmark_report.csv
```

`benchmark_matrix.json` 預設場景：base v1.0、DR v1.0、stress DR_level=0.7 v0.8、stress DR_level=1.0 v1.2。可自行加更多場景。

---

## 5. TensorBoard 監控

### SSH 端啟動

```bash
tensorboard --logdir ~/anaconda3/h1_package/code/logs/tb --port 6006 --host 0.0.0.0
```

### 本地開 Tunnel 連進去

```bash
ssh -L 6006:localhost:6006 root@10.6.243.55
```

瀏覽器開 `http://localhost:6006`

### 背景執行（不佔 terminal）

```bash
nohup tensorboard --logdir ~/anaconda3/h1_package/code/logs/tb \
    --port 6006 --host 127.0.0.1 > /tmp/tb.log 2>&1 &
```

---

## 6. 模型檔案說明

```
models/
├── best_model.zip              # EvalCallback 挑出的最佳 policy（base 訓練專用）
├── h1_vecnorm_best.pkl         # 對應 best_model.zip 的 VecNorm
│
├── h1_ppo.zip                  # base 訓練的最終 policy
├── h1_vecnorm.pkl              # base 訓練的 VecNorm
├── h1_ppo_XXXXX_steps.zip      # base 定期 checkpoint（保留最近 5 個）
│
├── h1_ppo_dr.zip               # DR 訓練（fresh / finetune 皆同名）的最終 policy
├── h1_vecnorm_dr.pkl           # DR 訓練的 VecNorm
├── h1_ppo_dr_XXXXX_steps.zip   # DR 定期 checkpoint
│
└── dr_best/                    # DR 專屬 best artifact 隔離目錄（避免覆蓋 base best）
    ├── best_model.zip          # DR EvalCallback 挑出的最佳 policy
    └── h1_vecnorm_best.pkl     # 對應 dr_best/best_model.zip 的 VecNorm
```

> **為什麼 DR best 放在子目錄？** SB3 的 `EvalCallback` 會把最佳模型固定存成 `best_model.zip`，檔名寫死無法客製。如果 DR 訓練直接寫到 `models/`，會把 base 訓練的 `best_model.zip` / `h1_vecnorm_best.pkl` 覆蓋掉。改成讓 DR 訓練寫到 `models/dr_best/` 子目錄，即可保留兩套獨立的 best artifact。

### Model ↔ VecNorm 配對規則

| 使用情境 | Model | VecNorm |
|----------|-------|---------|
| Base eval | `best_model.zip` | `h1_vecnorm_best.pkl` |
| DR eval | `dr_best/best_model.zip`（最優先）或 `h1_ppo_dr.zip` | `dr_best/h1_vecnorm_best.pkl` 或 `h1_vecnorm_dr.pkl` |
| Base resume | `h1_ppo_*_steps.zip`（最大 step）| `h1_vecnorm.pkl` |
| DR resume | `h1_ppo_dr_*_steps.zip`（最大 step）| `h1_vecnorm_dr.pkl` |

> **配對必須嚴格遵守**：policy 是在特定 VecNorm 分佈下訓練的，用錯 VecNorm 會導致觀測分佈漂移，policy 立刻跌倒。`eval.py --dr` 已內建以下優先順序：`dr_best/best_model.zip` → `h1_ppo_dr.zip` → `h1_ppo.zip` →（最後備援）`best_model.zip`，對應 VecNorm 也同順序搜尋。

---

## 7. 常見問題

**Q: eval 時 policy 一直跌倒？**
1. 確認 model / VecNorm 配對正確（見 Section 6）
2. DR eval 必須用 DR artifacts：優先 `dr_best/best_model.zip` + `dr_best/h1_vecnorm_best.pkl`，否則 `h1_ppo_dr.zip` + `h1_vecnorm_dr.pkl`；絕對不要用 base `best_model.zip`（會觀測分佈漂移）
3. 確認 DR finetune 沒有中斷（看 `logs/run_XXX/manifest.json` 的 `status` 欄位）
4. 執行 `gate_check.py` 確認品質門檻

**Q: `models/` 裡沒有 `best_model.zip`？**
- `--smoke` 模式不觸發 EvalCallback（設計如此）
- 一般訓練的第一次 eval 要到 `SAVE_FREQ // n_envs` 步後才觸發，訓練初期正常

**Q: 多機器換環境數量？**

Linux / SSH（bash / zsh）：
```bash
H1_N_ENVS=8 python train.py     # 本地較弱機器
H1_N_ENVS=32 python train.py    # SSH 高效能機
```

Windows PowerShell：
```powershell
$env:H1_N_ENVS="8"; python train.py
$env:H1_N_ENVS="32"; python train.py
```

Windows cmd.exe：
```cmd
set H1_N_ENVS=8 && python train.py
```

**Q: 訓練中途中斷怎麼辦？**
- `train.py` 會在 finally 區塊 best-effort 存最後狀態（model + VecNorm + manifest）
- 可用 `--resume` 接回最新 step checkpoint 繼續
- manifest 的 `status` 會記 `interrupted_by_user` 或 `failed:ExceptionName`

**Q: Fresh DR 跟 DR Finetune 輸出檔名一樣，會不會互相覆蓋？**
- 會。兩種模式都輸出到 `h1_ppo_dr.zip` / `h1_vecnorm_dr.pkl`，以及 `models/dr_best/` 下的 best artifact。跑 fresh DR 前先備份或改名這幾個檔案，避免覆蓋 finetune 成果。

**Q: DR 訓練會不會覆蓋 `best_model.zip` / `h1_vecnorm_best.pkl`？**
- 不會。DR 的 `EvalCallback` 被導向 `models/dr_best/` 子目錄，base best 與 DR best 完全隔離。`eval.py --dr` / `compare_eval.py` / `benchmark_matrix.py` 都已更新會優先讀 `dr_best/best_model.zip`。

**Q: 想做 ablation 研究，關掉某個 reward term？**
```bash
python train.py --ablate contact   # 把 contact reward 歸零
```
可用的 reward term 名稱見 `h1_env.py:_DEFAULT_REWARD_SCALES`。

**Q: checkpoint 存的頻率對嗎？**
- `SAVE_FREQ`（預設 500k timesteps）已在 [train.py](train.py) 自動除以 `n_envs` 換算成 callback-call 數，不需手動調整
