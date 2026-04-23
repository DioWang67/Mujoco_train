from pathlib import Path

from training_config import load_grasp_train_config, load_h1_train_config


def test_load_h1_train_config_reads_expected_defaults() -> None:
    config = load_h1_train_config(Path.cwd())

    assert config.n_envs_default == 32
    assert config.quick_n_envs == 4
    assert config.max_episode_steps == 1000
    assert config.curriculum_stages[0] == (0.0, 0.2)


def test_load_grasp_train_config_reads_expected_defaults() -> None:
    config = load_grasp_train_config(Path.cwd())

    assert config.n_envs_default == 8
    assert config.max_episode_steps == 300
    assert config.eval_episodes == 5
    assert config.net_arch == [256, 256]
