"""
Build an offline-installable package for the remote Linux GPU server.

Tested against:
  Remote OS   : Ubuntu 20.04 (glibc 2.31)
  Remote GPU  : 4x NVIDIA RTX A6000
  Remote CUDA : 12.4
  Remote Py   : 3.12
"""

import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

# ── Settings ──────────────────────────────────────────────────────────────
PY_VER = "312"                # cp312 = Python 3.12
CUDA_VER = "cu124"            # CUDA 12.4
PLATFORM = "manylinux2014_x86_64"
TORCH_IDX = f"https://download.pytorch.org/whl/{CUDA_VER}"
REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = REPO_ROOT / "artifacts" / "offline"
PKG_DIR = ARTIFACT_ROOT / "h1_package"
WHEELS = PKG_DIR / "wheels"
CODE = PKG_DIR / "code"
HERE = REPO_ROOT
# ──────────────────────────────────────────────────────────────────────────

SETUP_SH = textwrap.dedent("""\
    #!/bin/bash
    # H1 MuJoCo offline setup script
    set -e
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    WHEELS="$SCRIPT_DIR/wheels"
    CODE="$SCRIPT_DIR/code"

    echo "=== H1 MuJoCo Setup ==="
    echo "Python : $(python3 --version)"
    echo "Wheels : $WHEELS"
    echo ""

    echo "[1/3] Installing PyTorch (CUDA) ..."
    pip3 install --no-index --find-links="$WHEELS" torch torchvision

    echo "[2/3] Installing other packages ..."
    pip3 install --no-index --find-links="$WHEELS" \\
        mujoco gymnasium stable-baselines3 \\
        "numpy<2.0" tensorboard imageio optuna \\
        scipy cloudpickle tqdm rich packaging

    echo "[3/3] Verifying ..."
    python3 - <<'EOF'
import torch, mujoco, gymnasium, stable_baselines3, numpy
print(f"  torch      : {torch.__version__}")
print(f"  CUDA avail : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU        : {torch.cuda.get_device_name(0)}")
print(f"  mujoco     : {mujoco.__version__}")
print(f"  gymnasium  : {gymnasium.__version__}")
print(f"  sb3        : {stable_baselines3.__version__}")
print(f"  numpy      : {numpy.__version__}")
print("  All OK!")
EOF

    echo ""
    echo "=== Setup complete ==="
    echo "To train H1 walking:"
    echo "  cd $CODE"
    echo "  H1_N_ENVS=64 python3 train.py"
    echo ""
    echo "To train grasp baseline:"
    echo "  cd $CODE"
    echo "  python3 -m grasp_baseline.train --smoke --n-envs 1 --fixed-cube"
""")


def run(cmd: list[str], **kw) -> int:
    print("  $", " ".join(cmd))
    return subprocess.run(cmd, **kw).returncode


def pip_download(packages: list[str], extra_index: str | None = None) -> None:
    cmd = [
        sys.executable, "-m", "pip", "download",
        "--dest", str(WHEELS),
        "--platform", PLATFORM,
        "--python-version", PY_VER,
        "--only-binary", ":all:",
        "--no-deps",
    ]
    if extra_index:
        cmd += ["--extra-index-url", extra_index]
    cmd += packages
    rc = run(cmd)
    if rc != 0:
        # Fallback: drop --only-binary so source dists are also considered.
        print("  [warn] retrying without --only-binary ...")
        cmd2 = [c for c in cmd if c != ":all:" and c != "--only-binary"]
        run(cmd2)


def _force_rmtree(path: Path) -> None:
    """Remove a directory tree, handling Windows read-only files (.git)."""
    import stat

    def _on_error(func, fpath, _exc):
        os.chmod(fpath, stat.S_IWRITE)
        func(fpath)

    shutil.rmtree(path, onerror=_on_error)


def main() -> None:
    # ── Clean slate ─────────────────────────────────────────────────────
    if PKG_DIR.exists():
        _force_rmtree(PKG_DIR)
    WHEELS.mkdir(parents=True)
    CODE.mkdir(parents=True)

    # ── Copy source files ────────────────────────────────────────────────
    print("\n[1/4] Copying source code ...")
    copy_pairs = [
        (HERE / "train.py", CODE / "train.py"),
        (HERE / "h1_env.py", CODE / "h1_env.py"),
        (HERE / "eval.py", CODE / "eval.py"),
        (HERE / "requirements.txt", CODE / "requirements.txt"),
    ]
    for src, dst in copy_pairs:
        if src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(src, dst)

    for package_name in ["tools", "grasp_baseline"]:
        src_dir = HERE / package_name
        dst_dir = CODE / package_name
        if src_dir.exists():
            shutil.copytree(
                src_dir,
                dst_dir,
                dirs_exist_ok=True,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache"),
            )
    menagerie = HERE / "mujoco_menagerie"
    if menagerie.exists():
        # Exclude .git — it's large and not needed for training.
        shutil.copytree(
            menagerie, CODE / "mujoco_menagerie",
            ignore=shutil.ignore_patterns(".git", ".git/*"),
        )
    print("  Done.")

    # ── Download PyTorch (CUDA) ──────────────────────────────────────────
    # Download WITH deps so nvidia-cuda-nvrtc-cu12 etc. are included.
    print(f"\n[2/4] Downloading PyTorch {CUDA_VER} for Linux {PY_VER} ...")
    print("  (approx 4-5 GB including CUDA runtime libs, please wait)\n")
    cmd = [
        sys.executable, "-m", "pip", "download",
        "--dest", str(WHEELS),
        "--platform", PLATFORM,
        "--python-version", PY_VER,
        "--only-binary", ":all:",
        "--extra-index-url", TORCH_IDX,
        "torch", "torchvision",
    ]
    run(cmd)
    print("  Done.")

    # ── Download other dependencies ──────────────────────────────────────
    print("\n[3/4] Downloading other dependencies ...")
    pip_download([
        "mujoco>=3.1.0",
        "gymnasium>=1.0.0",
        "stable-baselines3>=2.3.0",
        "numpy<2.0",
        "tensorboard",
        "imageio",
        "optuna",
        "scipy",
        "cloudpickle",
        "tqdm",
        "rich",
        "packaging",
    ])
    print("  Done.")

    # ── Write setup.sh ───────────────────────────────────────────────────
    print("\n[4/4] Writing setup.sh ...")
    setup_path = PKG_DIR / "setup.sh"
    setup_path.write_text(SETUP_SH, encoding="utf-8")
    print(f"  Written: {setup_path}")

    # ── Pack ─────────────────────────────────────────────────────────────
    archive = ARTIFACT_ROOT / "h1_package.tar.gz"
    print(f"\nPacking into {archive} ...")
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    rc = run(["tar", "-czf", str(archive), str(PKG_DIR)])
    if rc != 0:
        print("  tar not available — please zip h1_package/ manually.")
    else:
        size_mb = archive.stat().st_size / 1024 / 1024
        print(f"  {archive}  ({size_mb:.0f} MB)")

    print("\n" + "=" * 56)
    print(f"  Package ready: {archive}")
    print()
    print("  Transfer:")
    print("    scp artifacts/offline/h1_package.tar.gz user@remote:~")
    print()
    print("  On remote:")
    print("    tar xzf h1_package.tar.gz")
    print("    cd h1_package && bash setup.sh")
    print("    cd code && H1_N_ENVS=64 python3 train.py")
    print("=" * 56 + "\n")


if __name__ == "__main__":
    main()
