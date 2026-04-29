import os

import train


def test_configure_numeric_runtime_sets_safe_defaults(monkeypatch) -> None:
    for key in (
        "MKL_THREADING_LAYER",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        monkeypatch.delenv(key, raising=False)

    train.configure_numeric_runtime()

    assert os.environ["MKL_THREADING_LAYER"] == "GNU"
    assert os.environ["OMP_NUM_THREADS"] == "1"
    assert os.environ["MKL_NUM_THREADS"] == "1"
    assert os.environ["OPENBLAS_NUM_THREADS"] == "1"
    assert os.environ["NUMEXPR_NUM_THREADS"] == "1"


def test_configure_numeric_runtime_preserves_explicit_threading_layer(
    monkeypatch,
) -> None:
    monkeypatch.setenv("MKL_THREADING_LAYER", "INTEL")

    train.configure_numeric_runtime()

    assert os.environ["MKL_THREADING_LAYER"] == "INTEL"
