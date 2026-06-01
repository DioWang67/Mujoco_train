from __future__ import annotations

from pathlib import Path

from tools.sedon_phase1_5_force_split_rollover_controller import (
    Phase15Runtime,
    _summarize,
)


def _row(
    *,
    step: int,
    support_ratio: float,
    support_side: str = "left",
    contact_state: str = "both",
    upright: float = 0.99,
    base_x: float | None = None,
) -> dict[str, object]:
    return {
        "step": step,
        "support_side": support_side,
        "base_x": float(step) * 0.001 if base_x is None else base_x,
        "base_vx": 0.01,
        "base_roll": 0.01,
        "base_pitch": 0.01,
        "upright": upright,
        "contact_state": contact_state,
        "jump": False,
        "support_force_ratio": support_ratio,
        "swing_force_ratio": 1.0 - support_ratio,
        "center_contact_left": step == 1,
        "toe_contact_left": step == 2,
        "center_contact_right": False,
        "toe_contact_right": False,
    }


def test_phase1_5_summary_passes_without_requiring_toe_handoff(tmp_path: Path) -> None:
    rows = [
        _row(step=step, support_ratio=0.60, support_side="left" if step < 15 else "right")
        for step in range(1, 25)
    ]
    for row in rows:
        row["center_contact_left"] = False
        row["toe_contact_left"] = False

    summary = _summarize(
        rows,
        Phase15Runtime(force_gate_reached_count=1, completed_steps=1),
        tmp_path / "timeline.csv",
        tmp_path / "summary.json",
    )

    assert summary.phase1_5_passed is True
    assert summary.toe_handoff_detected is False
    assert summary.support_force_ratio_hold_steps == 24


def test_phase1_5_summary_fails_when_force_split_never_opens(tmp_path: Path) -> None:
    rows = [_row(step=step, support_ratio=0.55) for step in range(1, 25)]

    summary = _summarize(
        rows,
        Phase15Runtime(),
        tmp_path / "timeline.csv",
        tmp_path / "summary.json",
    )

    assert summary.phase1_5_passed is False
    assert "support_force_ratio_peak" in summary.fail_reasons
    assert "support_force_ratio_hold_steps" in summary.fail_reasons
