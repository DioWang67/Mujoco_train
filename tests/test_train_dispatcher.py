import types

import train


class _Project:
    def __init__(self, train_module: str, display_name: str = "test robot") -> None:
        self.train_module = train_module
        self.display_name = display_name


def test_train_dispatcher_runs_configured_project_module(monkeypatch) -> None:
    captured_argv: list[str] = []
    fake_module = types.SimpleNamespace(
        main=lambda argv: captured_argv.extend(argv) or 0
    )

    monkeypatch.setattr(
        train,
        "split_mode_args",
        lambda argv: ("quadruped", ["--smoke", "--n-envs", "2"]),
    )
    monkeypatch.setattr(
        train,
        "get_robot_project",
        lambda slug: _Project("robots.quadruped.train"),
    )
    monkeypatch.setattr(train.importlib, "import_module", lambda name: fake_module)

    assert train.main(["--project", "quadruped", "--smoke"]) == 0
    assert captured_argv == ["--smoke", "--n-envs", "2"]


def test_train_dispatcher_reports_missing_main(monkeypatch) -> None:
    monkeypatch.setattr(train, "split_mode_args", lambda argv: ("badbot", []))
    monkeypatch.setattr(train, "get_robot_project", lambda slug: _Project("badbot.train"))
    monkeypatch.setattr(
        train.importlib,
        "import_module",
        lambda name: types.SimpleNamespace(),
    )

    assert train.main(["--project", "badbot"]) == 2


def test_train_dispatcher_lists_projects_without_importing_train_modules(
    monkeypatch,
) -> None:
    imported_modules: list[str] = []

    monkeypatch.setattr(
        train,
        "load_robot_projects",
        lambda: {
            "h1": _Project("h1_train"),
            "quadruped": _Project("robots.quadruped.train"),
        },
    )
    monkeypatch.setattr(
        train.importlib,
        "import_module",
        lambda name: imported_modules.append(name),
    )

    assert train.main(["--list-projects"]) == 0
    assert imported_modules == []
