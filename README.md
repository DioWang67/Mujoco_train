# H1 MuJoCo Walking

使用 PPO 訓練 Unitree H1 人形機器人在 MuJoCo 模擬環境中行走。

Reward 結構移植自 [unitree_rl_gym](https://github.com/unitreerobotics/unitree_rl_gym) 官方 H1 配置（legged_gym 框架）。

---

## 這個專案在做什麼？

這個專案用**強化學習（Reinforcement Learning, RL）** 教一個虛擬人形機器人學會走路。

### 強化學習是什麼？

簡單說：讓機器人在模擬環境裡不斷嘗試，做對的事情給獎勵，做錯的事情給懲罰，久了它就學會怎麼走路。

```
機器人做動作 → 環境給 reward（分數）→ 演算法更新策略 → 再試一次
                                                    ↑ 重複幾百萬次
```

### 關鍵名詞

| 名詞 | 白話解釋 |
|------|---------|
| **Policy** | 機器人的「大腦」，輸入感測器數據，輸出要怎麼動 |
| **Reward** | 每個時間步給的分數，正的是獎勵、負的是懲罰 |
| **Episode** | 一次完整的嘗試，從站立到跌倒（或跑滿時間限制） |
| **Step** | 一個時間步，約 0.02 秒模擬時間 |
| **PPO** | 我們用的 RL 演算法，全名 Proximal Policy Optimization |
| **Checkpoint** | 訓練中途儲存的模型快照，可以從這裡繼續 |

---

## 安裝

```cmd
cd d:\Git\robotlearning\h1_mujoco
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

---

## 快速開始

### 第零步：先做 preflight（推薦）

```cmd
python preflight_check.py
```

若顯示 `Preflight PASSED` 再開始訓練，可避免路徑/依賴/環境問題卡在中途。


### 第一步：開始訓練

```cmd
fresh_train.bat
```

畫面會出現類似這樣的輸出：

```
       Steps     Eps      MeanR   MeanLen     BestR     FPS        ETA
------------------------------------------------------------------------
   1,000,000      42       85.3     120.5     210.1   6200  02:10:00
   2,000,000     105      210.5     280.3     450.2   6100  01:58:00
```

| 欄位 | 說明 | 代表什麼 |
|------|------|---------|
| Steps | 總訓練步數 | 訓練進度 |
| Eps | 完成的 episode 數 | - |
| MeanR | 最近 50 集的平均分數 | 越高越好 |
| MeanLen | 平均 episode 長度（步數） | 越長代表越不容易跌倒 |
| BestR | 歷史最高分 | - |
| FPS | 每秒模擬步數 | 速度指標 |
| ETA | 預計完成時間 | - |

### 第二步：看機器人走路

訓練過程中或結束後，開新的終端機：

```cmd
python eval.py
```

### 第三步：監控訓練（可選）

```cmd
tensorboard --logdir logs\tb
```

然後開瀏覽器進 `http://localhost:6006`

---

## 使用方式

```cmd
fresh_train.bat              # 全新訓練 (40M steps)
resume_train.bat             # 從上次中斷點繼續

python train.py --smoke      # 快速測試（5k steps，確認環境沒壞）
python train.py --dr         # 啟用 Domain Randomization + Curriculum Learning
python train.py --ablate contact  # Ablation: 把 contact reward 設成 0 觀察影響
python eval.py               # 看機器人走路
python eval.py --log         # 走一集並記錄每步數據到 CSV
python eval.py --episodes 10 # 跑 10 集評估
python plot_eval.py --save   # 把 eval 數據畫成圖
python compare_eval.py --episodes 8 --out-json compare_report.json --out-csv compare_report.csv
python aggregate_compare.py --seeds 3 --episodes 5 --out-json aggregate_compare.json --out-csv aggregate_compare.csv
python gate_check.py --report aggregate_compare.json --gates gate_profiles.json --mode aggregate --profile preprod
python benchmark_matrix.py --matrix benchmark_matrix.json --out-json benchmark_report.json --out-csv benchmark_report.csv
python sweep.py              # Optuna 超參數搜索（進階）
```

---

## 怎麼看訓練是否正常？

### 用終端機數字判斷

| 訓練階段 | Steps | MeanLen | MeanR |
|---------|-------|---------|-------|
| 一直跌倒 | < 1M | < 100 | < 50 |
| 開始站穩 | ~1M | 100~300 | 50~200 |
| 開始走路 | ~3M | 300~600 | 200~600 |
| 走路穩定 | ~8M | 600~900 | 600~1500 |
| 收斂 | ~18M | 接近 1000 | > 1500 |

**MeanLen = 1000** 代表機器人在 1000 步內完全沒跌倒（跑滿時間限制），這是最理想的狀態。

### 用 TensorBoard 判斷

開啟後重點看以下曲線（都應該隨時間往好的方向走）：

| Tag | 好的數值 | 代表意義 |
|-----|---------|---------|
| `reward/tracking_lin_vel` | > 1.5 | 機器人有沒有照指令速度走 |
| `reward/tracking_ang_vel` | > 0.6 | 機器人有沒有走直線不偏轉 |
| `reward/contact_no_vel` | 接近 0 | 腳著地時有沒有在拖步 |
| `reward/orientation` | 接近 0 | 身體有沒有傾斜 |
| `reward/hip_pos` | 接近 0 | 髖關節有沒有偏掉（繞環步） |
| `rollout/ep_len_mean` | 越高越好 | 平均存活多久 |

---

## 什麼時候該等、什麼時候該動手？

這是最重要的判斷，新手最常犯的錯誤是**太早動手**。

### 等待（不要動參數）

- 訓練剛開始 < 3M steps，什麼問題都正常
- 改完參數後 < 2M steps，policy 還在適應
- 曲線在上升，即使上升很慢
- 偶爾退步但整體趨勢向上

### 考慮動手

- 某個問題在 5M steps 後完全沒有改善
- 曲線完全平坦超過 3M steps（卡住了）
- 出現影像上明顯的問題（走歪、繞環、不對稱）

### 動手原則

**一次只改一個參數，然後從頭重新訓練。**

不要邊跑邊改，這樣很難判斷哪個改動有效。

---

## 進階功能

### Domain Randomization（`--dr`）

訓練時加上 `--dr` 旗標會啟用以下隨機化，讓 policy 更 robust，為 sim-to-real 做準備：

| 隨機化項目 | 範圍 | 目的 |
|-----------|------|------|
| 地面摩擦力 | baseline × 0.5~1.5 | 適應不同地板材質 |
| 肢體質量 | baseline × 0.9~1.1 | 適應製造公差 |
| 觀測噪聲 | Gaussian σ=0.02 | 適應感測器噪聲 |
| 馬達延遲 | 預設關閉（0 步） | 可自行擴充做延遲壓力測試 |
| 速度指令 | vx=[0.3,1.5], vy=[-0.3,0.3], vyaw=[-0.3,0.3] | 學會多種速度和方向 |

`--dr` 同時啟用 **Curriculum Learning**：目標速度從 0.3 m/s 漸進到 1.0 m/s，讓機器人先學會慢走再加速。

### Ablation Study（`--ablate`）

用來分析各 reward 項目的影響。指定一個 reward term 名稱，訓練時會把它的 scale 設成 0：

```cmd
python train.py --quick --ablate contact       # 不給步態節拍獎勵會怎樣？
python train.py --quick --ablate hip_pos        # 不懲罰髖關節偏移會怎樣？
python train.py --quick --ablate feet_swing_height  # 不要求抬腳高度會怎樣？
```

可用的 reward term 名稱：`tracking_lin_vel`, `tracking_ang_vel`, `alive`, `contact`, `lin_vel_z`, `ang_vel_xy`, `orientation`, `base_height`, `dof_acc`, `action_rate`, `collision`, `dof_pos_limits`, `hip_pos`, `contact_no_vel`, `feet_swing_height`

### 模擬數據輸出

```cmd
python eval.py --log         # 產出 eval_ep1.csv
python plot_eval.py --save   # 產出 eval_ep1_plot.png
```

CSV 包含每步的骨盆位置、速度、各關節角度/角速度/扭矩。詳見 [HOW_TO_READ_RESULTS.md](HOW_TO_READ_RESULTS.md)。

---

## 專案結構

```
h1_mujoco/
├── h1_env.py            # 環境定義（reward、DR、command randomization）
├── train.py             # PPO 訓練（含 Curriculum、Ablation）
├── eval.py              # 評估 + CSV 數據輸出
├── plot_eval.py         # 視覺化 eval 數據（6 面板圖）
├── sweep.py             # Optuna 超參數自動搜索
├── preflight_check.py   # 訓練前環境健康檢查
├── compare_eval.py      # BASE vs DR 單次數值比較
├── aggregate_compare.py # 多 seed 比較 + 95% CI
├── benchmark_matrix.py  # Benchmark matrix 統一報告
├── gate_check.py        # 依 gate 規則自動 PASS/FAIL
├── gate_profiles.json   # research/preprod/release 門檻
├── fresh_train.bat      # 全新訓練捷徑
├── resume_train.bat     # 繼續訓練捷徑
├── HOW_TO_READ_RESULTS.md  # 給機構/馬達工程師的結果判讀指南
├── requirements.txt
├── mujoco_menagerie/    # H1 機器人 3D 模型檔
├── models/              # 儲存的模型
│   ├── h1_ppo.zip       # 最終模型
│   ├── h1_ppo_dr.zip    # DR 最終模型
│   ├── best_model.zip   # eval 分數最高的模型
│   └── h1_ppo_*_steps.zip  # 中途 checkpoint
└── logs/
    ├── tb/              # TensorBoard 資料
    └── run_<hash>/      # 每次訓練的設定記錄（含 DR/ablation 設定）
```

---

## Reward 設計（白話說明）

Reward 就是告訴機器人「什麼是好行為、什麼是壞行為」的分數系統。每個時間步都會計算一次，加總起來就是這個 episode 的總分。

### 正向獎勵（做到這些給加分）

| 項目 | Scale | 白話說明 |
|------|-------|---------|
| tracking_lin_vel | +2.0 | 照指定速度往前走，走越準分越高 |
| tracking_ang_vel | +0.5 | 照指定方向走（不偏轉），走越準分越高 |
| alive | +0.5 | 每一步只要沒跌倒就加分 |
| contact | +0.18 | 腳的接觸時機跟步態節拍吻合就加分 |

### 懲罰（做這些會扣分）

| 項目 | Scale | 白話說明 | 懲罰什麼行為 |
|------|-------|---------|------------|
| lin_vel_z | -0.5 | 身體不能上下彈跳 | 防止用跳的代替走 |
| ang_vel_xy | -0.01 | 身體不能前後左右搖晃 | 防止重心不穩 |
| orientation | -1.0 | 身體不能傾斜 | 防止走路身體歪一邊 |
| base_height | -2.0 | 骨盆要維持在 1.05m 高 | 防止蹲著走 |
| dof_acc | -2.5e-7 | 關節不能急速加速 | 讓動作更平滑自然 |
| action_rate | -0.01 | 相鄰動作不能差太多 | 防止抖動 |
| collision | -1.0 | 髖/膝不能互相碰撞 | 防止奇怪姿勢 |
| dof_pos_limits | -5.0 | 關節不能頂到極限角度 | 防止關節鎖死 |
| hip_pos | -0.5 | 髖關節要保持在中立位置 | 防止腳往外繞半圈走路 |
| contact_no_vel | -0.2 | 腳著地的時候不能還在移動 | 防止拖步 |
| feet_swing_height | -5.0 | 擺動的腳要抬到 8cm 高 | 確保腳有抬起來 |

### Scale 的意義

Scale 的**絕對值越大**，這個項目對機器人的影響越大。

例如：`tracking_lin_vel = +2.0` 遠大於 `contact = +0.18`，代表「往前走」比「踩準節拍」重要 10 倍。

---

## 控制方式（技術細節）

**PD 位置控制**（非直接力矩），與官方 legged_gym 一致：

```
action ∈ [-1, 1]^19          ← policy 輸出 19 個關節的目標
target_pos = default_pos + action × 0.25 rad
torque = Kp × (target - current) − Kd × velocity
```

---

## 觀測空間 (dim = 73)

機器人每個時間步「看到」的資訊：

| 欄位 | 維度 | 說明 |
|------|------|------|
| projected gravity | 3 | 重力方向（判斷自己有沒有傾斜） |
| base linear velocity | 3 | 現在的移動速度 |
| base angular velocity | 3 | 現在的旋轉速度 |
| velocity command | 3 | 被指定要走的速度和方向 |
| joint pos − default | 19 | 各關節目前偏離預設位置多少 |
| joint velocities | 19 | 各關節現在轉動速度 |
| previous action | 19 | 上一步做的動作 |
| gait phase | 4 | 步態節拍（告訴機器人現在該哪隻腳動） |

---

## PPO 超參數

| 參數 | 值 | 說明 |
|------|----|------|
| 並行環境數 | 4 | 同時跑 4 個模擬加快收集資料 |
| 總訓練步數 | 20,000,000 | 預計跑 20M steps |
| 最大 episode 長度 | 1,000 steps | 不跌倒最多跑 1000 步（約 20 秒） |
| Rollout steps | 2,048 | 每次更新前收集多少資料 |
| Batch size | 512 | 每次梯度更新用多少樣本 |
| Learning rate | 3e-4 | 每次更新的步伐大小 |
| Network | MLP [256, 256] | 神經網路架構（兩層各 256 個神經元） |

---

## 訓練進度里程碑

| 里程碑 | 預期 steps | 觀察指標 |
|--------|-----------|---------|
| 開始站穩 | ~1M | `ep_len_mean` > 100 |
| 開始往前走 | ~3M | `tracking_lin_vel` > 0.5 |
| 走路穩定 | ~8M | `ep_len_mean` > 700 |
| 走路方向正確 | ~12M | `tracking_ang_vel` > 0.5 |
| 收斂 | ~18M | `ep_len_mean` 接近 1000 |

---

## 常見問題排查

### 走歪 / 緩慢轉圈

**症狀**：機器人走著走著慢慢偏轉方向，不走直線

**診斷**：TensorBoard 看 `reward/tracking_ang_vel`，如果 < 0.3 就是這個問題

**原因**：`tracking_ang_vel` 的 scale 太小，policy 不在乎方向偏轉

**解法**：在 `h1_env.py` 調高 `tracking_ang_vel` scale（目前 1.0，原本 0.5 不夠）

---

### 身體傾斜

**症狀**：走路時身體明顯歪向一側，像喝醉了一樣

**診斷**：看 `reward/ang_vel_xy` 絕對值是否 > 0.05

**原因**：`ang_vel_xy` scale 太小，roll/pitch 不穩沒有被有效懲罰

**解法**：調高 `ang_vel_xy` 絕對值（從 -0.01 調到 -0.05）

---

### 拖步 / 不抬腳

**症狀**：走路像在拖著腳走，腳不離地

**診斷**：`reward/contact_no_vel` 絕對值持續 > 0.3

**原因**：著地時腳還在移動，`contact_no_vel` 懲罰不夠重

**解法**：調高 `contact_no_vel` 絕對值（從 -0.2 調到 -0.4）

---

### 繞環步（腳往外劃半圈）

**症狀**：擺動的腳不是直接往前，而是向外繞一圈再踏下去，不自然

**診斷**：`reward/hip_pos` 絕對值持續 > 0.05

**原因**：髖關節側偏沒有被有效懲罰，機器人用「繞」代替「抬」來淨空地面

**解法**：調高 `hip_pos` 絕對值（從 -0.5 調到 -2.0）

---

### 左右腳步伐不對稱

**症狀**：一腳大步、另一腳小步，或明顯的一腳主導

**原因**：每次 episode 都從相同的步態相位（phase = 0）開始，
policy 學到偏袒「先動的那隻腳」

**解法**：`h1_env.py` 的 `reset_model()` 改成隨機初始化步態相位

```python
# 改這一行
self._gait_phase = float(self.np_random.uniform(0.0, 1.0))
```

---

### Eval 卡住不動（畫面停著不更新）

**症狀**：訓練輸出停止更新好幾分鐘，看不出有沒有在跑

**原因**：機器人學會走路後不跌倒，但環境沒有設最長時間，
EvalCallback 的 episode 永遠不結束

**解法**：`train.py` 的 `make_env()` 加入 `TimeLimit` wrapper

```python
from gymnasium.wrappers import TimeLimit
env = TimeLimit(env, max_episode_steps=1000)
```

---

### 記憶體不足（MemoryError: bad allocation）

**症狀**：啟動訓練時，多個 subprocess 報錯後 crash

**原因**：並行環境數太多，每個 MuJoCo 環境需要約 500MB RAM

**解法**：`train.py` 中將 `N_ENVS` 從 8 降到 4

---

### 訓練速度突然驟降

**症狀**：FPS 從幾千突然掉到幾百，持續一段時間後恢復

**原因**：`EvalCallback` 觸發，正在跑 evaluation episode（單環境、確定性）

**這是正常現象**，等它跑完會自動恢復。
如果覺得太頻繁，調高 `SAVE_FREQ`（目前 500,000）

---

## Sim-to-Real 建議流程

這個模擬的最終目標是部署到真實機器人。建議的訓練順序：

### 第一階段：確認 policy 能走路（無 DR）

```cmd
fresh_train.bat   # 標準訓練，不加 --dr
```

目標：eval reward > 1500，ep_len 接近 1000。先確認 policy 在理想模擬條件下能走好，這是後續所有步驟的基礎。不要跳過這步就加 DR——先有能走的 policy，才有「讓它走得更 robust」的空間。

### 第二階段：加入 Domain Randomization（提升真實世界適應性）

```cmd
python train.py --dr
```

DR 會讓機器人在各種物理條件下都能走，縮小模擬與現實的差距（Sim-to-Real Gap）。預期現象：
- 訓練初期分數下降（環境變難了）
- 需要更多 steps 才能收斂（建議 25~30M）
- 最終走路品質可能比無 DR 稍差，但對真機更穩

### 第三階段：評估並輸出數據

```cmd
python eval.py --log
python plot_eval.py --save
```

把輸出的圖表（`eval_ep1_plot.png`）給機構/馬達工程師看，對照 [HOW_TO_READ_RESULTS.md](HOW_TO_READ_RESULTS.md)。

### Sim-to-Real Gap 主要來源

| 差異項目 | 這個模擬的處理方式 | 備註 |
|---------|-----------------|------|
| 地面摩擦力 | DR 隨機化 ×0.5~1.5 | 真機要先在平整地面測試 |
| 機體質量 | DR 隨機化 ±10% | 實際測重後可調整範圍 |
| 馬達延遲 | DR 模擬 0~40ms | 真機通訊延遲要量測 |
| 感測器噪聲 | DR 加入 Gaussian 噪聲 | IMU 噪聲特性依型號不同 |
| 地面不平整 | 未處理 | 進階：可加 terrain randomization |
| 關節彈性形變 | 未處理 | 真機需要實測驗證 |

---

## Reward Tuning 原則

1. **一次只改一個參數**，給足 2~3M steps 再判斷有沒有效
2. **先重新訓練，不要只 resume**，帶著舊行為繼續往往學得很慢
3. **先看 TensorBoard 數字，再改**，不要憑感覺亂調
4. **Eval reward 是主要指標**，rollout 的 MeanR 受 VecNormalize 影響可能失真
5. **改完後退步是正常的**，policy 需要 1~2M steps 適應新規則才會回升
6. **用 ablation 找關鍵 reward**：懷疑某項 reward 沒有貢獻，先用 `--ablate` 快速驗證，不要直接改 scale
