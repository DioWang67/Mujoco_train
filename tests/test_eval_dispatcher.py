import types

import eval


class _Project:
    def __init__(
        self,
        eval_module: str | None,
        display_name: str = "test robot",
        train_module: str = "test.train",
    ) -> None:
        self.eval_module = eval_module
        self.display_name = display_name
        self.train_module = train_module


def test_eval_dispatcher_runs_configured_project_module(monkeypatch) -> None:
    captured_argv: list[str] = []
    fake_module = types.SimpleNamespace(
        main=lambda argv: captured_argv.extend(argv) or 0
    )

    monkeypatch.setattr(
        eval,
        "split_mode_args",
        lambda argv: ("sedon", ["--episodes", "1", "--render"]),
    )
    monkeypatch.setattr(
        eval,
        "get_robot_project",
        lambda slug: _Project("sedon_baseline.eval"),
    )
    monkeypatch.setattr(eval.importlib, "import_module", lambda name: fake_module)

    assert eval.main(["--project", "sedon", "--episodes", "1", "--render"]) == 0
    assert captured_argv == ["--episodes", "1", "--render"]


def test_eval_dispatcher_reports_missing_eval_module(monkeypatch) -> None:
    monkeypatch.setattr(eval, "split_mode_args", lambda argv: ("badbot", []))
    monkeypatch.setattr(eval, "get_robot_project", lambda slug: _Project(None))

    assert eval.main(["--project", "badbot"]) == 2


def test_eval_dispatcher_lists_projects_without_importing_eval_modules(
    monkeypatch,
) -> None:
    imported_modules: list[str] = []

    monkeypatch.setattr(
        eval,
        "load_robot_projects",
        lambda: {
            "h1": _Project("h1_baseline.eval", display_name="H1 walking"),
            "sedon": _Project("sedon_baseline.eval", display_name="Sedon"),
        },
    )
    monkeypatch.setattr(
        eval.importlib,
        "import_module",
        lambda name: imported_modules.append(name),
    )

    assert eval.main(["--list-projects"]) == 0
    assert imported_modules == []
