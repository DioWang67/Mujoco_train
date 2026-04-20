# Code Review Report (2026-04-20)

## 範圍
- `train.py`
- `h1_env.py`
- `eval.py`
- `sweep.py`
- `prepare_package.py`
- `plot_eval.py`
- `download_missing.py`

## 檢查方式
- 靜態程式閱讀（架構、健壯性、可維護性）。
- Python 語法編譯檢查：`python -m compileall ...`。

## 發現問題

### 1) reset observation 與 step observation 參考座標不一致（高風險）
**檔案**：`h1_env.py`

`step()` 會把 `base_lin_vel`、`base_ang_vel` 轉換到 body frame 再組 observation；
但 `reset_model()` 回傳 observation 時，直接使用世界座標速度 `self.data.qvel[0:3]` / `self.data.qvel[3:6]`。

這會造成 episode 首步（reset 後）的觀測分佈與後續步驟不一致，容易增加 policy 學習不穩定。

**建議**：在 `reset_model()` 內也使用與 `step()` 相同的 rotation 計算，統一為 body frame。

### 2) `_find_latest_checkpoint()` 的 `prefer_dr` 參數未生效（中風險）
**檔案**：`train.py`

函式簽名含 `prefer_dr`，但內部排序/過濾完全未使用該參數，
導致 DR / 非 DR checkpoint 同時存在時，行為不透明且與 API 語意不一致。

**建議**：
- 依 `prefer_dr` 明確優先 DR 命名模式，或
- 移除未使用參數，避免誤導。

### 3) 斷點選擇策略與函式名稱語意不一致（中風險）
**檔案**：`train.py`

`_find_latest_checkpoint()` 目前會把無步數檔名（如 `h1_ppo.zip`、`best_model.zip`）給極大排序值，
實際效果偏向「優先 final/best」，不一定是「最新 step checkpoint」。

**建議**：
- 若目標是「最新訓練進度」，應優先 `*_steps.zip` 最大步數；
- 若目標是「最佳模型」，請改名與註解為 `find_preferred_resume_model` 類型語意。

### 4) `download_missing.py` 缺乏錯誤處理與退出碼檢查（中風險）
**檔案**：`download_missing.py`

`pip download` 與 `tar` 的執行沒有對失敗流程做完整處理：
- `pip download` return code 未檢查，
- `tar` 失敗時沒有輸出 stderr 與非零退出。

這可能導致 CI/自動化流程誤判成功。

**建議**：
- 使用 `check=True` 或手動檢查 return code，
- 失敗時 `sys.exit(1)` 並印出 stderr。

## 正向觀察
- `train.py` 的 callback 分層與功能邊界清楚，且對 VecNormalize 有獨立保存策略。
- `h1_env.py` 的 reward term 可讀性高，並透過 `_DEFAULT_REWARD_SCALES` 支援 ablation。
- `eval.py` 對 DR / non-DR VecNormalize 配對有清楚註解，降低推論時統計失配風險。

## 建議後續優先順序
1. 先修正 `h1_env.py` reset 觀測座標一致性。
2. 明確化 checkpoint 選擇邏輯（語意 + 實作一致）。
3. 補強下載/封包工具腳本的錯誤處理。
