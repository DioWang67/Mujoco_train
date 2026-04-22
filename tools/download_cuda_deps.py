"""Download NVIDIA CUDA runtime Python packages for the remote server."""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = REPO_ROOT / "artifacts" / "offline"
OUT = ARTIFACT_ROOT / "cuda_deps"
OUT.mkdir(parents=True, exist_ok=True)

NVIDIA_PKGS = [
    "nvidia-cuda-nvrtc-cu12==12.4.127",
    "nvidia-cuda-runtime-cu12==12.4.127",
    "nvidia-cuda-cupti-cu12==12.4.127",
    "nvidia-cublas-cu12==12.4.5.8",
    "nvidia-cufft-cu12==11.2.1.3",
    "nvidia-curand-cu12==10.3.5.147",
    "nvidia-cusolver-cu12==11.6.1.9",
    "nvidia-cusparse-cu12==12.3.1.170",
    "nvidia-cusparselt-cu12",
    "nvidia-nccl-cu12==2.21.5",
    "nvidia-nvtx-cu12==12.4.127",
    "nvidia-nvjitlink-cu12==12.4.127",
    "nvidia-cudnn-cu12==9.1.0.70",
]

print(f"Downloading {len(NVIDIA_PKGS)} NVIDIA packages -> {OUT}/")
print("(approx 1.7 GB)\n")

base_cmd = [
    sys.executable, "-m", "pip", "download",
    "--dest", str(OUT),
    "--platform", "manylinux2014_x86_64",
    "--python-version", "312",
    "--only-binary", ":all:",
    "--no-deps",
]

rc = subprocess.run(base_cmd + NVIDIA_PKGS).returncode
if rc != 0:
    print("\n[warn] Retrying without version pins ...")
    pkgs_no_ver = [p.split("==")[0] for p in NVIDIA_PKGS]
    subprocess.run(base_cmd + pkgs_no_ver)

archive = ARTIFACT_ROOT / "cuda_deps.tar.gz"
result = subprocess.run(
    ["tar", "-czf", str(archive), str(OUT)],
    capture_output=True,
)
if result.returncode == 0:
    size = archive.stat().st_size / 1024 / 1024
    print(f"\nPacked: {archive} ({size:.0f} MB)")
    print("\nOn remote:")
    print("  tar xzf cuda_deps.tar.gz")
    print("  pip install --no-deps cuda_deps/*.whl")
else:
    print("\ncuda_deps/ folder ready — zip and transfer manually.")
