# H1 MuJoCo 完整訓練手冊

這份文件的目標很直接：

1. 從零開始把環境跑起來
2. 完成 base training
3. 正確評估與判讀結果
4. 在 base 穩定後再進入 DR finetune
5. 避免常見的模型 / VecNorm / artifact 混用

如果你只想要最短版本，先看「快速路線」。如果你要可重複的完整流程，從頭照這份走。

---

## 1. 快速路線

### 1.1 從零到 base 成功

```bash
python -m tools.preflight_check
python train.py --smoke
python train.py
```

### 1.2 base 跑完後驗證

```bash
python eval.py
python -m tools.compare_eval --episodes 8 --vel 1.0 --out-json reports/compare_report.json --out-csv reports/compare_report.csv
python -m tools.gate_check --report reports/compare_report.json --gates configs/gate_profiles.json --profile preprod
```

### 1.3 base 成功後做 DR finetune

```bash
python train.py --finetune models/best_model.zip --dr
```

### 1.4 DR 跑完後驗證

```bash
python eval.py --dr
python -m tools.compare_eval --episodes 8 --vel 1.0 --out-json reports/compare_report.json --out-csv reports/compare_report.csv
python -m tools.gate_check --report reports/compare_report.json --gates configs/gate_profiles.json --profile preprod
```

---

## 2. 先理解你在跑什麼

這個 repo 目前有兩條主要訓練路線：

- `base training`
  先在 non-DR 環境把 walking policy 練穩
- `DR finetune`
  從 base best model 出發，在 domain randomization 下增加 robustness

正確順序是：

```text
preflight -> smoke -> base training -> base eval/compare -> DR finetune -> DR eval/compare
```

不建議的順序：

- base 還沒穩就直接 `--dr`
- compare report 很差還硬切 DR
- model / VecNorm 沒對齊就拿數字做結論

---

## 3. 環境與前置檢查

### 3.1 安裝

如果你還沒建環境：

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 3.2 preflight

正式訓練前先跑：

```bash
python -m tools.preflight_check
```

你要看到整體是通過的狀態，再往下跑。  
如果 import、DLL、GPU、MuJoCo、Torch 這裡就失敗，後面不要硬跑。

### 3.3 smoke

```bash
python train.py --smoke
```

用途：

- 確認程式沒壞
- 確認訓練流程能啟動
- 確認 callback / env / MuJoCo / SB3 沒有立即炸掉

注意：

- `--smoke` 不會產生 `best_model.zip`
- 它只是流程檢查，不是性能驗證

---

## 4. 訓練指令

### 4.1 base training

```bash
python train.py
```

預設：

- 40M timesteps
- non-DR
- 32 個並行 env（可由 `H1_N_ENVS` 覆蓋）

Windows PowerShell 設 env 範例：

```powershell
$env:H1_N_ENVS="8"; python train.py
```

Linux / bash 範例：

```bash
H1_N_ENVS=8 python train.py
```

### 4.2 續訓

```bash
python train.py --resume
python train.py --resume --dr
```

`--resume` 會依模式選最近的 checkpoint：

- base 模式優先 `h1_ppo_*_steps.zip`
- DR 模式優先 `h1_ppo_dr_*_steps.zip`
- 找不到才 fallback 到 final model
- 再找不到才最後 fallback 到 `best_model.zip`

### 4.3 DR finetune

```bash
python train.py --finetune models/best_model.zip --dr
```

這是推薦路線。  
意義是：

- 載入 base 最佳模型
- 用較保守的 finetune hyperparameters
- 進入 DR + curriculum + DR ramp

### 4.4 從零開始 DR

```bash
python train.py --dr
```

可以跑，但不是首選。  
通常收斂會慢，而且更容易把 base 本來可以穩定走的能力一起打掉。

---

## 5. 訓練輸出檔案

### 5.1 base artifacts

```text
models/
  best_model.zip
  h1_vecnorm_best.pkl
  h1_ppo.zip
  h1_vecnorm.pkl
  h1_ppo_XXXXX_steps.zip
```

### 5.2 DR artifacts

```text
models/
  h1_ppo_dr.zip
  h1_vecnorm_dr.pkl
  h1_ppo_dr_XXXXX_steps.zip
  dr_best/
    best_model.zip
    h1_vecnorm_best.pkl
```

### 5.3 為什麼 DR best 要放在 `dr_best/`

因為 SB3 的 `EvalCallback` 會固定把最佳模型寫成 `best_model.zip`。  
如果 DR 訓練也直接寫進 `models/`，就會覆蓋 base 的：

- `best_model.zip`
- `h1_vecnorm_best.pkl`

現在的設計是：

- base best 留在 `models/`
- DR best 寫到 `models/dr_best/`

這樣兩套最佳模型彼此隔離。

---

## 6. model / VecNorm 配對規則

這件事很重要。配錯時，policy 可能立刻跌倒，或者數值報告完全失真。

### 6.1 正確配對

| 使用情境 | Model | VecNorm |
|---|---|---|
| Base eval | `models/best_model.zip` | `models/h1_vecnorm_best.pkl` |
| Base final eval | `models/h1_ppo.zip` | `models/h1_vecnorm.pkl` |
| DR eval | `models/dr_best/best_model.zip` | `models/dr_best/h1_vecnorm_best.pkl` |
| DR final eval | `models/h1_ppo_dr.zip` | `models/h1_vecnorm_dr.pkl` |
| Base resume | `h1_ppo_*_steps.zip` | `h1_vecnorm.pkl` |
| DR resume | `h1_ppo_dr_*_steps.zip` | `h1_vecnorm_dr.pkl` |

### 6.2 compare_eval 現在怎麼做

`tools.compare_eval` 現在是：

- 先選一個要比較的 model artifact
- 再用與該 model 對應的單一 VecNorm
- 用同一個 model / VecNorm 去跑：
  - base env
  - DR env

所以 `Delta DR-BASE` 的意思是：

- 同一個 policy
- 在 base 與 DR 兩種環境中的差異

這樣才有比較價值。

---

## 7. 評估指令

### 7.1 直接 eval

```bash
python eval.py
python eval.py --dr
python eval.py --auto-dr
```

說明：

- `python eval.py`
  base 模式
- `python eval.py --dr`
  DR 模式，優先讀 `dr_best`
- `python eval.py --auto-dr`
  若偵測到 DR artifacts，自動切 DR 模式

### 7.2 錄影 / CSV

```bash
python eval.py --record
python eval.py --record --dr
python eval.py --episodes 5 --log
python eval.py --vel 0.8
```

### 7.3 compare report

```bash
python -m tools.compare_eval --episodes 8 --vel 1.0 --out-json reports/compare_report.json --out-csv reports/compare_report.csv
```

### 7.4 gate check

```bash
python -m tools.gate_check --report reports/compare_report.json --gates configs/gate_profiles.json --profile preprod
```

### 7.5 aggregate compare

```bash
python -m tools.aggregate_compare --seeds 3 --seed-start 42 --episodes 5 --vel 1.0 --out-json reports/aggregate_compare.json --out-csv reports/aggregate_compare.csv
python -m tools.gate_check --report reports/aggregate_compare.json --gates configs/gate_profiles.json --profile preprod --mode aggregate
```

### 7.6 benchmark matrix

```bash
python -m tools.benchmark_matrix --matrix configs/benchmark_matrix.json --out-json reports/benchmark_report.json --out-csv reports/benchmark_report.csv
```

---

## 8. 怎麼判讀結果

這一節是這份手冊的核心。不要只看 command 跑完，要會判讀。

### 8.1 base 成功長什麼樣

一份健康的 base report 通常會長成這樣：

- `Len` 很高，接近 episode 上限 `1000`
- `Vx` 接近 target velocity，例如 `1.0` 目標時有 `0.9+`
- `Reward` 穩定且波動不大

例如這類結果就可以判定 base 已經成形：

```text
BASE   R=2921.7±1.5   Len=1000.0±0.0   Vx=0.95±0.00
```

這代表：

- 能穩定撐滿 episode
- 能往正確方向前進
- 沒有明顯退化

### 8.2 base 還沒成形長什麼樣

如果你看到：

- `Len < 150`
- `Vx <= 0`
- reward 很低

那通常代表 policy 根本還沒學對。  
尤其 `Vx < 0` 很嚴重，代表它甚至在往後退，不是單純走不快。

### 8.3 DR gap 怎麼看

`tools.compare_eval` 的重點不是看 DR 絕對值，而是看：

- `Delta DR-BASE`

如果 base 很強，但 DR 很差，像這種：

```text
BASE   R=2921.7±1.5    Len=1000.0±0.0    Vx=0.95±0.00
DR     R=  70.4±33.0   Len=  80.5±12.6   Vx=0.23±0.34
Delta DR-BASE: Reward=-2851.2, Len=-919.5, Vx=-0.72
```

意思是：

- base policy 沒問題
- robustness 很差
- 現在該做 DR finetune

這時候**不是重跑 base**，而是進 DR finetune。

### 8.4 什麼時候不要切 DR

下列情況不要切 DR：

- base `Len` 還很低
- base `Vx` 還是負的或很接近 0
- TensorBoard 顯示還在明顯成長期
- `compare_eval` 顯示 base 本身就不穩

### 8.5 什麼時候可以切 DR

建議條件：

- base eval 已能穩定走
- `Len` 接近 `1000`
- `Vx` 接近 target
- 姿態看起來正常，不是靠奇怪扭胯撐住
- compare report 顯示 base 很強，只是 DR gap 很大

---

## 9. TensorBoard 要看什麼

### 9.1 最重要的幾條

- `rollout/ep_len_mean`
  越高越好
- `reward/tracking_lin_vel`
  越高通常表示前進能力更好
- `reward/tracking_ang_vel`
  越高通常表示角速度追蹤更穩
- `reward/orientation`
  越接近 0 越好
- `reward/contact_no_vel`
  不要越來越爛
- `reward/hip_pos`
  如果後期一直惡化，代表可能在學不好看的代償姿勢

### 9.2 一般判讀

- `ep_len_mean` 一路升，通常是好事
- `tracking_lin_vel` 後段明顯上升，通常代表開始真的會走
- `orientation` 變得不那麼負，通常代表姿態變穩
- `hip_pos` 持續惡化，表示可能用醜姿勢換穩定

### 9.3 什麼叫「不像壞掉」

如果你看到：

- `ep_len_mean` 明顯上升
- `tracking_lin_vel` 也在上升
- 沒有長時間卡在很低水平

那通常不是壞掉，而是還在學。

---

## 10. 邊訓練邊 eval 的注意事項

### 10.1 為什麼有時候 eval 會失敗

如果你在訓練途中讀：

- `best_model.zip`

可能剛好撞到 `EvalCallback` 正在覆蓋寫入。  
這時可能出現：

- `Bad CRC-32`
- `wasn't a zip-file`

現在 `eval.py` 已經做了較安全的載入：

- 先複製到暫存檔
- 再載入
- 載入失敗會重試幾次

但本質上，訓練中的 artifact 還是比訓練結束後更容易撞 race。

### 10.2 比較穩的做法

如果你只是想看訓練趨勢：

- 可以看 TensorBoard

如果你要做嚴格評估：

- 優先在訓練結束後跑
- 或使用 step checkpoint，而不是反覆讀 `best_model.zip`

---

## 11. 常見問題

### 11.1 eval / compare 卡住

以前這個 repo 有一個真 bug：

- 訓練時 env 有 `TimeLimit(max_episode_steps=1000)`
- 但 eval / compare 用的是裸 `H1Env`

結果如果 policy 沒跌倒，episode 可能一直不結束，看起來像卡死。

現在已經修掉，`eval.py` / `tools.compare_eval` / `tools.benchmark_matrix` 都套了 `TimeLimit(1000)`。

### 11.2 compare report 很怪

先檢查：

- `Model: ...`
- `VecNorm(model): ...`

如果 model / VecNorm 不匹配，報告沒有判讀價值。

### 11.3 `best_model.zip` 讀不到

先分兩種：

- 路徑錯
- 檔案正在被覆蓋寫入

目前載入邏輯已經修到直接用完整 `.zip` 路徑，且 `eval.py` 對訓練中的檔案有 retry。

### 11.4 20M base 跑到一半，中途修 code，還能不能直接接 DR

不要只看 timesteps。  
真正該看的是：

- 新版 compare report
- base 是否已經穩定撐滿 episode
- `Vx` 是否正確

如果新版 compare report 顯示 base 已經強，就不用重跑 base；  
如果還沒成形，就先把 base 跑好。

### 11.5 Fresh DR 和 DR finetune 會不會互相覆蓋

會。

因為兩者都會寫：

- `models/h1_ppo_dr.zip`
- `models/h1_vecnorm_dr.pkl`
- `models/dr_best/`

所以在做新的 DR 實驗前，要先備份你想保留的 DR artifacts。

---

## 12. 建議決策流程

### 12.1 base 訓練中

1. 先看 TensorBoard
2. 如果 `ep_len_mean` 還在明顯上升，就先繼續跑
3. 到你覺得像成形時，再跑 `eval.py` / `python -m tools.compare_eval`

### 12.2 base 訓練後

如果：

- `Len` 接近 `1000`
- `Vx` 接近 target
- 行為看起來穩

就：

```bash
python train.py --finetune models/best_model.zip --dr
```

### 12.3 DR finetune 後

再跑：

```bash
python eval.py --dr
python -m tools.compare_eval --episodes 8 --vel 1.0 --out-json reports/compare_report.json --out-csv reports/compare_report.csv
python -m tools.gate_check --report reports/compare_report.json --gates configs/gate_profiles.json --profile preprod
```

如果 DR 仍然大崩：

- 先不要怪 base
- 優先看 DR ramp、DR 強度、DR 訓練步數是否足夠

---

## 13. 建議備份策略

在 base 成功、準備進 DR 前，建議備份至少以下檔案：

```text
models/best_model.zip
models/h1_vecnorm_best.pkl
models/h1_ppo.zip
models/h1_vecnorm.pkl
```

原因很簡單：

- base best 是之後所有 DR finetune 的乾淨起點
- 不要把唯一一份穩定 base 基線覆蓋掉

---

## 14. 最後的原則

這個 repo 最容易出事的不是 MuJoCo 本身，而是：

- observation / reset 定義不一致
- model / VecNorm 配錯
- base / DR artifact 混用
- 邊訓練邊讀正在覆蓋的檔案
- 報告數字看起來像結果，其實比較方式不乾淨

所以判斷時請遵守這三條：

1. 先確認 artifact 配對正確
2. 再看 eval / compare 數字
3. 最後才做訓練決策

只要這三步顛倒，結論很容易錯。
