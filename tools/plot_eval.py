"""視覺化 eval 資料。

Usage:
    python -m tools.plot_eval              # 讀 eval_ep1.csv
    python -m tools.plot_eval --file eval_ep2.csv
    python -m tools.plot_eval --save       # 存成 PNG 而不是開視窗
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd

# 設定中文字體（Windows）
for font in ["Microsoft JhengHei", "Microsoft YaHei", "SimHei", "DFKai-SB"]:
    if any(font.lower() in f.name.lower() for f in fm.fontManager.ttflist):
        plt.rcParams["font.family"] = font
        break
plt.rcParams["axes.unicode_minus"] = False  # 負號正常顯示

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LEFT_LEG  = ["L_hip_yaw", "L_hip_roll", "L_hip_pitch", "L_knee", "L_ankle"]
RIGHT_LEG = ["R_hip_yaw", "R_hip_roll", "R_hip_pitch", "R_knee", "R_ankle"]
COLORS_L  = ["#1f77b4", "#aec7e8", "#2ca02c", "#98df8a", "#d62728"]
COLORS_R  = ["#ff7f0e", "#ffbb78", "#9467bd", "#c5b0d5", "#8c564b"]


def plot(csv_path: str, save: bool):
    df = pd.read_csv(csv_path)
    t = df["step"].values
    total_steps = len(t)

    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(
        f"H1 Walking Analysis  —  {os.path.basename(csv_path)}"
        f"  ({total_steps} steps)",
        fontsize=14, fontweight="bold", y=0.98,
    )

    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    # ── 1. 骨盆軌跡（俯視圖）────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    sc = ax1.scatter(
        df["pelvis_x"], df["pelvis_y"],
        c=t, cmap="viridis", s=3, zorder=2,
    )
    ax1.plot(df["pelvis_x"].iloc[0], df["pelvis_y"].iloc[0],
             "go", ms=8, label="Start", zorder=3)
    ax1.plot(df["pelvis_x"].iloc[-1], df["pelvis_y"].iloc[-1],
             "rs", ms=8, label="End", zorder=3)
    plt.colorbar(sc, ax=ax1, label="step")
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.set_title("行走軌跡（俯視）")
    ax1.legend(fontsize=8)
    ax1.set_aspect("equal")
    ax1.grid(True, alpha=0.3)

    # ── 2. 骨盆高度 + Roll/Pitch ─────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(t, df["pelvis_z"], color="#1f77b4", label="高度 (m)", linewidth=1.2)
    ax2.axhline(1.05, color="gray", linestyle="--", linewidth=0.8, label="目標 1.05m")
    ax2.set_ylabel("高度 (m)", color="#1f77b4")
    ax2.tick_params(axis="y", labelcolor="#1f77b4")
    ax2b = ax2.twinx()
    ax2b.plot(t, df["roll_deg"],  color="#d62728", label="Roll°",  linewidth=0.8, alpha=0.7)
    ax2b.plot(t, df["pitch_deg"], color="#ff7f0e", label="Pitch°", linewidth=0.8, alpha=0.7)
    ax2b.set_ylabel("角度 (°)", color="#d62728")
    ax2b.tick_params(axis="y", labelcolor="#d62728")
    lines1, lab1 = ax2.get_legend_handles_labels()
    lines2, lab2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, lab1 + lab2, fontsize=7, loc="upper right")
    ax2.set_title("骨盆高度 / 姿態")
    ax2.set_xlabel("step")
    ax2.grid(True, alpha=0.3)

    # ── 3. 前進速度 ──────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(t, df["vel_x"], color="#2ca02c", linewidth=1.2, label="vel_x")
    ax3.plot(t, df["vel_y"], color="#9467bd", linewidth=0.8, alpha=0.7, label="vel_y")
    ax3.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, label="目標 1.0 m/s")
    ax3.set_xlabel("step")
    ax3.set_ylabel("m/s")
    ax3.set_title("前進速度")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # ── 4. 左腿關節角度 ──────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, :2])
    for name, color in zip(LEFT_LEG, COLORS_L):
        col = f"{name}_pos_deg"
        if col in df.columns:
            ax4.plot(t, df[col], color=color, linewidth=1.0,
                     label=name.replace("L_", ""))
    ax4.set_xlabel("step")
    ax4.set_ylabel("角度 (°)")
    ax4.set_title("左腿關節角度")
    ax4.legend(fontsize=8, loc="upper right", ncol=3)
    ax4.grid(True, alpha=0.3)

    # ── 5. 右腿關節角度 ──────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[2, :2])
    for name, color in zip(RIGHT_LEG, COLORS_R):
        col = f"{name}_pos_deg"
        if col in df.columns:
            ax5.plot(t, df[col], color=color, linewidth=1.0,
                     label=name.replace("R_", ""))
    ax5.set_xlabel("step")
    ax5.set_ylabel("角度 (°)")
    ax5.set_title("右腿關節角度")
    ax5.legend(fontsize=8, loc="upper right", ncol=3)
    ax5.grid(True, alpha=0.3)

    # ── 6. 腿部最大扭矩 Bar chart ────────────────────────────────────
    ax6 = fig.add_subplot(gs[1:, 2])
    all_leg = LEFT_LEG + RIGHT_LEG
    short_names = [n.replace("L_", "L ").replace("R_", "R ") for n in all_leg]
    max_torques = []
    for name in all_leg:
        col = f"{name}_torque"
        max_torques.append(float(df[col].abs().max()) if col in df.columns else 0)

    bar_colors = COLORS_L + COLORS_R
    bars = ax6.barh(short_names, max_torques, color=bar_colors, alpha=0.8)
    ax6.set_xlabel("最大扭矩 (Nm)")
    ax6.set_title("各關節最大扭矩")
    ax6.grid(True, alpha=0.3, axis="x")
    for bar, val in zip(bars, max_torques):
        ax6.text(val + 1, bar.get_y() + bar.get_height() / 2,
                 f"{val:.0f}", va="center", fontsize=7)

    if save:
        out = csv_path.replace(".csv", "_plot.png")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"圖表存到 {out}")
    else:
        plt.show()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--file", default=os.path.join(REPO_ROOT, "eval_ep1.csv"))
    p.add_argument("--save", action="store_true", help="存成 PNG")
    args = p.parse_args()

    if not os.path.exists(args.file):
        print(f"找不到 {args.file}，請先跑：python -m h1_baseline.eval --log")
        sys.exit(1)

    plot(args.file, args.save)
