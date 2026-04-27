from tools.registry import TOOLS, tools_by_category


def test_tool_registry_contains_canonical_commands() -> None:
    modules = {tool.module for tool in TOOLS}

    assert "preflight_check" in modules
    assert "deploy_release" in modules
    assert "eval_grasp" in modules
    assert all(tool.command == f"python -m tools.{tool.module}" for tool in TOOLS)


def test_tool_registry_groups_tools_by_category() -> None:
    grouped = tools_by_category()

    assert "evaluation" in grouped
    assert "release" in grouped
    assert any(tool.module == "gate_check" for tool in grouped["evaluation"])
