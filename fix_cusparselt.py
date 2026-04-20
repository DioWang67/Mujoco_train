"""Download nvidia-cusparselt-cu12==0.6.2 to fix torch version conflict."""

import subprocess
import sys
from pathlib import Path

OUT = Path("cusparselt_fix")
OUT.mkdir(exist_ok=True)

print("Downloading nvidia-cusparselt-cu12==0.6.2 (the exact version torch 2.6.0 requires)...")

result = subprocess.run([
    sys.executable, "-m", "pip", "download",
    "nvidia-cusparselt-cu12==0.6.2",
    "--dest", str(OUT),
    "--platform", "manylinux2014_x86_64",
    "--python-version", "312",
    "--only-binary", ":all:",
    "--no-deps",
], capture_output=False)

if result.returncode != 0:
    print("\n[warn] ==0.6.2 not found on PyPI, trying version range...")
    result = subprocess.run([
        sys.executable, "-m", "pip", "download",
        "nvidia-cusparselt-cu12>=0.6.0,<0.7.0",
        "--dest", str(OUT),
        "--platform", "manylinux2014_x86_64",
        "--python-version", "312",
        "--only-binary", ":all:",
        "--no-deps",
    ])

wheels = list(OUT.glob("*.whl"))
if wheels:
    print(f"\nDownloaded: {[w.name for w in wheels]}")
    result = subprocess.run(
        ["tar", "-czf", "cusparselt_fix.tar.gz", str(OUT)],
        capture_output=True,
    )
    if result.returncode == 0:
        size = Path("cusparselt_fix.tar.gz").stat().st_size / 1024
        print(f"Packed: cusparselt_fix.tar.gz ({size:.0f} KB)")
        print("\nTransfer:")
        print("  scp -C cusparselt_fix.tar.gz root@<IP>:~/anaconda3/h1_package/")
        print("\nOn remote:")
        print("  cd ~/anaconda3/h1_package")
        print("  tar xzf cusparselt_fix.tar.gz")
        print("  pip uninstall nvidia-cusparselt-cu12 -y")
        print("  pip install --no-deps cusparselt_fix/*.whl")
        print("  pip install --no-index --find-links=all_deps --find-links=wheels --find-links=cusparselt_fix \\")
        print("      mujoco gymnasium stable-baselines3 tensorboard optuna")
        print("  python3 -c \"import torch; print(torch.cuda.is_available())\"")
else:
    print("\n[error] No wheels downloaded. Check if this version exists for this platform.")
    print("Tip: torch 2.6.0 may need nvidia-cusparselt-cu12==0.6.2 which might be Linux-only.")
    print("Check: pip index versions nvidia-cusparselt-cu12")
