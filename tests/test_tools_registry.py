from tools.registry import TOOLS, tools_by_category


def test_tool_registry_contains_canonical_commands() -> None:
    modules = {tool.module for tool in TOOLS}

    assert "preflight_check" in modules
    assert "project_inventory" in modules
    assert "agent_workspace" in modules
    assert "deploy_release" in modules
    assert "remote_auto_deploy" in modules
    assert "remote_training" in modules
    assert "eval_grasp" in modules
    assert "preview_sedon_blue_contact_gated" in modules
    assert "sweep_sedon_blue_contact_gated_targets" in modules
    assert "debug_sedon_com_mass_distribution_sweep" in modules
    assert "debug_sedon_com_feedback_balance_sweep" in modules
    assert "debug_sedon_foot_proxy_redesign_sweep" in modules
    assert "debug_sedon_hip_roll_force_breakdown" in modules
    assert "preview_sedon_blue_balance_controller" in modules
    assert "debug_sedon_forced_support_lift_check" in modules
    assert "debug_sedon_force_unload_controller" in modules
    assert "debug_sedon_mechanical_variant_sweep" in modules
    assert "sweep_sedon_preload" in modules
    assert "sweep_sedon_preload_v2" in modules
    assert "dynamic_preload_sweep" in modules
    assert "audit_sedon_shuffle_v0" in modules
    assert "geometry_sensitivity_sweep" in modules
    assert "lateral_authority_audit" in modules
    assert "assisted_shuffle_curriculum_eval" in modules
    assert "audit_sedon_teacher_imitation" in modules
    assert "residual_safety_sweep" in modules
    assert all(tool.command == f"python -m tools.{tool.module}" for tool in TOOLS)


def test_tool_registry_groups_tools_by_category() -> None:
    grouped = tools_by_category()

    assert "evaluation" in grouped
    assert "release" in grouped
    assert any(tool.module == "gate_check" for tool in grouped["evaluation"])
