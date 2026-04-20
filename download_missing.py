"""Download only the missing packages not in the previous cuda_deps bundle."""

import subprocess
import sys
from pathlib import Path

OUT = Path("missing_deps")
OUT.mkdir(exist_ok=True)

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

result = subprocess.run(
    ["tar", "-czf", "missing_deps.tar.gz", str(OUT)],
    capture_output=True,
)
if result.returncode == 0:
    size = Path("missing_deps.tar.gz").stat().st_size / 1024 / 1024
    print(f"\nPacked: missing_deps.tar.gz ({size:.1f} MB)")
    print("\nTransfer:")
    print("  scp -C missing_deps.tar.gz root@<IP>:~/anaconda3/h1_package/")
    print("\nOn remote:")
    print("  tar xzf missing_deps.tar.gz")
    print("  pip install --no-deps missing_deps/*.whl")
    print("  python3 -c \"import torch; print(torch.cuda.is_available())\"")
