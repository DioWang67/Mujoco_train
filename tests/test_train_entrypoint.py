from train_entrypoint import split_mode_args


def test_split_mode_args_defaults_to_h1() -> None:
    mode, forwarded = split_mode_args(["--smoke"])

    assert mode == "h1"
    assert forwarded == ["--smoke"]


def test_split_mode_args_routes_to_grasp() -> None:
    mode, forwarded = split_mode_args(["--grasp", "--phase", "full", "--n-envs", "32"])

    assert mode == "grasp"
    assert forwarded == ["--phase", "full", "--n-envs", "32"]


def test_split_mode_args_rejects_conflicting_modes() -> None:
    try:
        split_mode_args(["--h1", "--grasp"])
    except ValueError as exc:
        assert "Cannot use --h1 and --grasp together." in str(exc)
        return

    raise AssertionError("Expected ValueError for conflicting mode flags.")
