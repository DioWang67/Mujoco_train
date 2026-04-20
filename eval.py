"""Render a trained H1 policy.

Usage:
    python eval.py                          # watch live
    python eval.py --episodes 3             # run 3 episodes
    python eval.py --record                 # save mp4 (no display needed)
    python eval.py --record --episodes 3    # record 3 episodes
    python eval.py --no-render --log        # headless, save CSV only
"""

import argparse
import csv
import os
import sys
import time

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from h1_env import H1Env

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "models")
BEST_PATH = os.path.join(MODEL_DIR, "best_model")
FINAL_PATH = os.path.join(MODEL_DIR, "h1_ppo")
DR_FINAL_PATH = os.path.join(MODEL_DIR, "h1_ppo_dr")
VECNORM_BEST_PATH = os.path.join(MODEL_DIR, "h1_vecnorm_best.pkl")
VECNORM_DR_PATH = os.path.join(MODEL_DIR, "h1_vecnorm_dr.pkl")
VECNORM_PATH = os.path.join(MODEL_DIR, "h1_vecnorm.pkl")

JOINT_NAMES = [
    "L_hip_yaw", "L_hip_roll", "L_hip_pitch", "L_knee", "L_ankle",
    "R_hip_yaw", "R_hip_roll", "R_hip_pitch", "R_knee", "R_ankle",
    "torso",
    "L_sh_pitch", "L_sh_roll", "L_sh_yaw", "L_elbow",
    "R_sh_pitch", "R_sh_roll", "R_sh_yaw", "R_elbow",
]


def main(episodes: int, do_log: bool, render: bool, record: bool,
         dr: bool = False, target_vel: float = 1.0, auto_dr: bool = False):
    if auto_dr and not dr:
        dr = (
            os.path.exists(VECNORM_DR_PATH)
            and (
                os.path.exists(DR_FINAL_PATH + ".zip")
                or os.path.exists(FINAL_PATH + ".zip")
            )
        )
        if dr:
            print("[auto] Detected DR artifacts, enabling DR eval mode.")

    # DR eval uses the final finetuned model (h1_ppo.zip) + DR-adapted VecNorm
    # (h1_vecnorm_dr.pkl). best_model.zip was selected by EvalCallback on a
    # non-DR eval_env, so pairing it with a DR env causes observation mismatch.
    if dr:
        if os.path.exists(DR_FINAL_PATH + ".zip"):
            ckpt = DR_FINAL_PATH
        elif os.path.exists(FINAL_PATH + ".zip"):
            ckpt = FINAL_PATH
        elif os.path.exists(BEST_PATH + ".zip"):
            ckpt = BEST_PATH
        else:
            print("No trained model found. Run `python train.py` first.")
            return
        vecnorm_candidates = [VECNORM_DR_PATH, VECNORM_PATH]
    else:
        if os.path.exists(BEST_PATH + ".zip"):
            ckpt = BEST_PATH
        elif os.path.exists(FINAL_PATH + ".zip"):
            ckpt = FINAL_PATH
        else:
            print("No trained model found. Run `python train.py` first.")
            return
        vecnorm_candidates = [VECNORM_BEST_PATH, VECNORM_PATH]

    print(f"Loading model: {ckpt}.zip")
    print(f"Eval mode: {'DR' if dr else 'BASE'}")
    model = PPO.load(ckpt, custom_objects={"learning_rate": 3e-4, "clip_range": 0.2})

    vecnorm_file = None
    for candidate in vecnorm_candidates:
        if os.path.exists(candidate):
            vecnorm_file = candidate
            break
    vec_norm = None
    if vecnorm_file:
        print(f"VecNormalize: {vecnorm_file}")
        dummy = DummyVecEnv([lambda: H1Env()])
        vec_norm = VecNormalize.load(vecnorm_file, dummy)
        vec_norm.training = False
        vec_norm.norm_reward = False

    if record:
        try:
            import imageio
        except ImportError:
            print("[error] imageio not installed: pip install imageio")
            return
        render_mode = "rgb_array"
    elif render:
        render_mode = "human"
    else:
        render_mode = None

    env = H1Env(render_mode=render_mode, domain_randomization=dr,
                target_velocity=target_vel)
    print(f"Target velocity: {target_vel} m/s")

    try:
        for ep in range(1, episodes + 1):
            obs, _ = env.reset()
            done = False
            total_r = 0.0
            steps = 0
            frames = [] if record else None

            csv_path = None
            writer = None
            csv_file = None
            if do_log:
                csv_path = os.path.join(HERE, f"eval_ep{ep}.csv")
                csv_file = open(csv_path, "w", newline="")
                fieldnames = (
                    ["step", "pelvis_x", "pelvis_y", "pelvis_z",
                     "roll_deg", "pitch_deg", "yaw_rate",
                     "vel_x", "vel_y", "vel_z", "reward"]
                    + [f"{n}_pos_deg" for n in JOINT_NAMES]
                    + [f"{n}_vel_dps" for n in JOINT_NAMES]
                    + [f"{n}_torque" for n in JOINT_NAMES]
                )
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writeheader()

            while not done:
                obs_in = (
                    vec_norm.normalize_obs(obs.reshape(1, -1))[0]
                    if vec_norm else obs
                )
                action, _ = model.predict(obs_in, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                total_r += reward
                steps += 1
                done = terminated or truncated

                if record:
                    frames.append(env.render())
                elif render:
                    time.sleep(1 / 50)

                if writer:
                    qpos = env.data.qpos
                    qvel = env.data.qvel
                    ctrl = env.data.ctrl
                    yaw_rate = float(qvel[5])
                    row = {
                        "step": steps,
                        "pelvis_x": round(float(qpos[0]), 4),
                        "pelvis_y": round(float(qpos[1]), 4),
                        "pelvis_z": round(float(qpos[2]), 4),
                        "roll_deg": round(np.degrees(info["roll"]), 2),
                        "pitch_deg": round(np.degrees(info["pitch"]), 2),
                        "yaw_rate": round(yaw_rate, 4),
                        "vel_x": round(float(qvel[0]), 4),
                        "vel_y": round(float(qvel[1]), 4),
                        "vel_z": round(float(qvel[2]), 4),
                        "reward": round(float(reward), 4),
                    }
                    for i, name in enumerate(JOINT_NAMES):
                        row[f"{name}_pos_deg"] = round(np.degrees(float(qpos[7 + i])), 2)
                        row[f"{name}_vel_dps"] = round(np.degrees(float(qvel[6 + i])), 2)
                        row[f"{name}_torque"] = round(float(ctrl[i]), 2)
                    writer.writerow(row)

            if csv_file:
                csv_file.close()
                print(f"  -> CSV saved: {csv_path}")

            if record and frames:
                mp4_path = os.path.join(HERE, f"eval_ep{ep}.mp4")
                imageio.mimsave(mp4_path, frames, fps=50)
                print(f"  -> Video saved: {mp4_path}")

            print(f"Episode {ep:2d} | steps={steps:4d} | reward={total_r:8.1f}")

            if do_log:
                data = np.genfromtxt(csv_path, delimiter=",", names=True)
                print(f"  {'Joint':<16} {'Min':>10} {'Max':>10} {'MaxTorque':>10}")
                print(f"  {'-'*50}")
                for name in ["L_hip_yaw","L_hip_roll","L_hip_pitch","L_knee","L_ankle",
                             "R_hip_yaw","R_hip_roll","R_hip_pitch","R_knee","R_ankle"]:
                    col, torq = f"{name}_pos_deg", f"{name}_torque"
                    if col in data.dtype.names:
                        print(f"  {name:<16} {float(np.min(data[col])):>9.1f}deg"
                              f" {float(np.max(data[col])):>9.1f}deg"
                              f" {float(np.max(np.abs(data[torq]))):>9.1f}Nm")

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        env.close()
        if vec_norm:
            vec_norm.venv.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--log", action="store_true", help="save per-step CSV")
    p.add_argument("--no-render", action="store_true", help="headless mode")
    p.add_argument("--record", action="store_true",
                   help="record mp4 (headless, no display needed)")
    p.add_argument("--dr", action="store_true",
                   help="enable domain randomization eval "
                        "(prefers h1_ppo_dr.zip + h1_vecnorm_dr.pkl)")
    p.add_argument("--auto-dr", action="store_true",
                   help="auto-enable --dr when DR vecnorm/model artifacts exist")
    p.add_argument("--vel", type=float, default=1.0,
                   help="target velocity m/s (default 1.0, matches eval_env in training)")
    args = p.parse_args()
    render = not args.no_render and not args.record
    main(args.episodes, args.log, render, args.record, args.dr, args.vel, args.auto_dr)
