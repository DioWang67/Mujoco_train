# H1 MuJoCo Walking

這個 repo 用 PPO 在 MuJoCo 中訓練 Unitree H1 walking policy，包含：

- base training
- DR finetune
- eval / compare / benchmark
- gate check

---

## 文件入口

如果你要一份可以從頭走到尾的完整手冊，請直接看：

- [docs/TRAINING_RUNBOOK.md](./docs/TRAINING_RUNBOOK.md)

這份是目前唯一正式維護的操作文件，內容包含：

- 環境建立
- preflight / smoke
- base training
- DR finetune
- model / VecNorm / artifact 配對
- eval / compare / gate / benchmark 指令
- 結果判讀與決策規則
- 常見錯誤與排查方式

其餘文件偏補充用途，不應該拿來取代 runbook。

---

## 最短開始流程

```bash
python -m tools.preflight_check
python train.py --smoke
python train.py
```

base 跑完後：

```bash
python eval.py
python -m tools.compare_eval --episodes 8 --vel 1.0 --out-json reports/compare_report.json --out-csv reports/compare_report.csv
python -m tools.gate_check --report reports/compare_report.json --gates configs/gate_profiles.json --profile preprod
```

base 穩定後做 DR finetune：

```bash
python train.py --finetune models/best_model.zip --dr
```

---

## 常用指令

```bash
python -m tools.preflight_check
python train.py --smoke
python train.py
python train.py --resume
python train.py --finetune models/best_model.zip --dr

python eval.py
python eval.py --dr
python eval.py --auto-dr
python eval.py --record
python eval.py --record --dr

python -m tools.compare_eval --episodes 8 --vel 1.0 --out-json reports/compare_report.json --out-csv reports/compare_report.csv
python -m tools.aggregate_compare --seeds 3 --seed-start 42 --episodes 5 --vel 1.0 --out-json reports/aggregate_compare.json --out-csv reports/aggregate_compare.csv
python -m tools.gate_check --report reports/compare_report.json --gates configs/gate_profiles.json --profile preprod
python -m tools.benchmark_matrix --matrix configs/benchmark_matrix.json --out-json reports/benchmark_report.json --out-csv reports/benchmark_report.csv
```

---

## 重要原則

- 不要在 base 還沒穩時直接切 DR。
- 不要混用 model 與 VecNorm。
- `best_model.zip` 是 base best；DR best 在 `models/dr_best/`。
- 如果你要做正式判斷，以 `docs/TRAINING_RUNBOOK.md` 的流程與判讀規則為準。

---

## 相關文件

- [docs/TRAINING_RUNBOOK.md](./docs/TRAINING_RUNBOOK.md): 完整操作手冊
- [docs/REMOTE_LAYOUT.md](./docs/REMOTE_LAYOUT.md): SSH/remote 目錄結構建議
- [docs/ENV_TUNING_AND_PHASES.md](./docs/ENV_TUNING_AND_PHASES.md): env 數量與訓練階段建議
- [docs/PROJECT_STATUS.md](./docs/PROJECT_STATUS.md): 專案狀態摘要
- [docs/CODE_REVIEW.md](./docs/CODE_REVIEW.md): review 記錄
