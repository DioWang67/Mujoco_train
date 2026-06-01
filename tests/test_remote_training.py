from tools.remote_training import env_project_value, remote_start_script, remote_status_script


def test_env_project_value_prefers_project_specific_key() -> None:
    values = {
        "SEDON_RESUME_N_ENVS": "128",
        "REMOTE_TRAIN_RESUME_N_ENVS": "32",
    }

    assert env_project_value(values, "sedon", "RESUME_N_ENVS", "4") == "128"


def test_env_project_value_falls_back_to_generic_key() -> None:
    values = {"REMOTE_TRAIN_RESUME_N_ENVS": "32"}

    assert env_project_value(values, "grasp", "RESUME_N_ENVS", "4") == "32"


def test_remote_start_script_is_project_parameterized() -> None:
    script = remote_start_script(
        remote_root="/remote/root",
        project_slug="grasp",
        total_timesteps=1000,
        n_envs=8,
        reset_noise_scale=None,
        config_overrides=None,
        extra_args="--phase full",
        train_mode="resume",
    )

    assert "--project grasp" in script
    assert "/remote/root/runs/grasp/models/grasp" in script
    assert "SEDON_CONFIG_OVERRIDES" not in script
    assert "--phase full" in script


def test_remote_start_script_supports_fresh_config_run() -> None:
    script = remote_start_script(
        remote_root="/remote/root",
        project_slug="sedon",
        total_timesteps=1000,
        n_envs=8,
        reset_noise_scale=0.005,
        config_overrides="configs/sedon/reference.json",
        extra_args="",
        train_mode="fresh",
    )

    assert "Train mode: $TRAIN_MODE" in script
    assert "SEDON_CONFIG_OVERRIDES=configs/sedon/reference.json" in script
    assert "--resume" not in script
    assert "fresh_sedon_" in script


def test_remote_status_script_filters_one_project() -> None:
    script = remote_status_script("/remote/root", "sedon")

    assert "[t]rain.py --project sedon" in script
    assert "/remote/root/runs/sedon/logs/sedon" in script
