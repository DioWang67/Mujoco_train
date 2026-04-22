"""Download only the missing packages not in the previous cuda_deps bundle."""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = REPO_ROOT / "artifacts" / "offline"
OUT = ARTIFACT_ROOT / "missing_deps"
OUT.mkdir(parents=True, exist_ok=True)

MISSING = [
    "nvidia-cusparselt-cu12",
]

base_cmd = [
    sys.executable, "-m", "pip", "download",
    "--dest", str(OUT),
    "--platform", "manylinux2014_x86_64",
    "--python-version", "312",
    "--only-binary", ":all:",
    "--no-deps",
]

subprocess.run(base_cmd + MISSING)

archive = ARTIFACT_ROOT / "missing_deps.tar.gz"
result = subprocess.run(
    ["tar", "-czf", str(archive), str(OUT)],
    capture_output=True,
)
if result.returncode == 0:
    size = archive.stat().st_size / 1024 / 1024
    print(f"\nPacked: {archive} ({size:.1f} MB)")
    print("\nTransfer:")
    print("  scp -C artifacts/offline/missing_deps.tar.gz root@<IP>:~/mujoco-train-system/shared/offline/archives/")
    print("\nOn remote:")
    print("  tar xzf missing_deps.tar.gz")
    print("  pip install --no-deps missing_deps/*.whl")
    print("  python3 -c \"import torch; print(torch.cuda.is_available())\"")
