"""Preflight checks before training.

Usage:
    python preflight_check.py
"""

import os
import platform
import sys
from datetime import datetime


def main() -> int:
    print("=== H1 Preflight Check ===")
    ok = True

    # 1) Python/package imports
    try:
        import torch
        import mujoco
        import gymnasium
        import stable_baselines3
    except Exception as e:
        print(f"[FAIL] Import dependencies failed: {e}")
        return 1

    print(f"[OK] python      : {sys.version.split()[0]}")
    print(f"[OK] platform    : {platform.platform()}")
    print(f"[OK] torch       : {torch.__version__}")
    print(f"[OK] mujoco      : {mujoco.__version__}")
    print(f"[OK] gymnasium   : {gymnasium.__version__}")
    print(f"[OK] sb3         : {stable_baselines3.__version__}")
    print(f"[OK] CUDA avail  : {torch.cuda.is_available()}")

    # 2) File layout
    required = [
        "train.py",
        "h1_env.py",
        os.path.join("mujoco_menagerie", "unitree_h1", "scene.xml"),
    ]
    for p in required:
        if os.path.exists(p):
            print(f"[OK] file exists : {p}")
        else:
            print(f"[FAIL] missing   : {p}")
            ok = False

    # 3) Runtime env smoke
    try:
        from h1_env import H1Env

        env = H1Env(render_mode=None)
        obs, _ = env.reset(seed=123)
        if obs is None or len(obs) != 73:
            print("[FAIL] env reset/obs shape mismatch")
            ok = False
        else:
            print("[OK] env reset   : obs_dim=73")

        action = env.action_space.sample()
        _obs2, reward, terminated, truncated, _info = env.step(action)
        print(
            f"[OK] env step    : reward={float(reward):.3f}, "
            f"done={bool(terminated or truncated)}",
        )
        env.close()
    except Exception as e:
        print(f"[FAIL] env smoke failed: {e}")
        ok = False

    # 4) Writable output dirs
    for d in ["models", "logs"]:
        try:
            os.makedirs(d, exist_ok=True)
            test_path = os.path.join(d, ".preflight_write_test")
            with open(test_path, "w", encoding="utf-8") as f:
                f.write(datetime.now().isoformat())
            os.remove(test_path)
            print(f"[OK] writable    : {d}/")
        except Exception as e:
            print(f"[FAIL] writable  : {d}/ ({e})")
            ok = False

    # 5) Summary
    print("-" * 56)
    if ok:
        print("Preflight PASSED. You can start training.")
        return 0
    print("Preflight FAILED. Fix above issues before training.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
