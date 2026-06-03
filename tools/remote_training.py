"""Start or inspect remote project training using the shared remote layout."""

from __future__ import annotations

import argparse
import os
import shlex
from pathlib import Path

from tools.remote_auto_deploy import (
    DEFAULT_ENV_FILE,
    NativeRunner,
    build_askpass_env,
    build_config,
    build_ssh_command,
    parse_args as parse_deploy_args,
    parse_env_file,
    quote_remote,
    select_backend,
)


DEFAULT_TOTAL_TIMESTEPS = 2_000_000
DEFAULT_N_ENVS = 128
DEFAULT_RESET_NOISE_SCALE = 0.005


def parse_args() -> argparse.Namespace:
    """Parse the generic remote training operator surface."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument("--project", default=None)
    parser.add_argument("--status", action="store_true")
    return parser.parse_args()


def env_project_value(env_values: dict[str, str], project_slug: str, suffix: str, default: str) -> str:
    """Read project-specific env, then generic env, then default."""
    project_key = f"{project_slug.upper()}_{suffix}"
    generic_key = f"REMOTE_TRAIN_{suffix}"
    return env_values.get(project_key) or env_values.get(generic_key) or default


def remote_start_script(
    *,
    remote_root: str,
    project_slug: str,
    total_timesteps: int,
    n_envs: int,
    reset_noise_scale: float | None,
    config_overrides: str | None,
    extra_args: str,
    train_mode: str,
) -> str:
    """Return a remote shell script that starts project training in background."""
    if train_mode not in {"resume", "fresh"}:
        raise ValueError(f"Unsupported remote training mode: {train_mode}")
    run_root = f"{remote_root}/runs/{project_slug}"
    model_dir = f"{run_root}/models/{project_slug}"
    log_dir = f"{run_root}/logs/{project_slug}"
    code_dir = f"{remote_root}/code/current"
    log_prefix = shlex.quote(project_slug)

    reset_args = (
        f"--reset-noise-scale {float(reset_noise_scale)} "
        if reset_noise_scale is not None
        else ""
    )
    config_export = (
        f"SEEDON_CONFIG_OVERRIDES={quote_remote(config_overrides)} "
        if config_overrides
        else ""
    )

    lines = [
        "set -e",
        f"PROJECT={quote_remote(project_slug)}",
        f"TRAIN_MODE={quote_remote(train_mode)}",
        f"RUN_ROOT={quote_remote(run_root)}",
        f"MODEL_DIR={quote_remote(model_dir)}",
        f"LOG_DIR={quote_remote(log_dir)}",
        f"CODE_DIR={quote_remote(code_dir)}",
        "mkdir -p \"$MODEL_DIR\" \"$LOG_DIR\"",
    ]
    if train_mode == "resume":
        lines.extend(
            [
        "if [ -f \"$MODEL_DIR/stable_forward_best/stable_forward_best_model.zip\" ]; then RESUME_MODEL=\"$MODEL_DIR/stable_forward_best/stable_forward_best_model.zip\"; "
        "elif [ -f \"$MODEL_DIR/forward_best/forward_best_model.zip\" ]; then RESUME_MODEL=\"$MODEL_DIR/forward_best/forward_best_model.zip\"; "
        "elif [ -f \"$MODEL_DIR/best/best_model.zip\" ]; then RESUME_MODEL=\"$MODEL_DIR/best/best_model.zip\"; "
        "elif [ -f \"$MODEL_DIR/latest_model.zip\" ]; then RESUME_MODEL=\"$MODEL_DIR/latest_model.zip\"; "
        "elif [ -f \"$MODEL_DIR/latest_model\" ]; then RESUME_MODEL=\"$MODEL_DIR/latest_model\"; "
        "else RESUME_MODEL=$(find \"$MODEL_DIR\" -maxdepth 2 -type f -name '*.zip' -printf '%T@ %p\\n' 2>/dev/null | sort -nr | head -n 1 | cut -d' ' -f2- || true); fi",
        'if [ -z "${RESUME_MODEL:-}" ] || [ ! -f "$RESUME_MODEL" ]; then echo "No checkpoint found under $MODEL_DIR" >&2; exit 1; fi',
        "if [ -f \"$MODEL_DIR/vecnorm.pkl\" ]; then RESUME_VECNORM=\"$MODEL_DIR/vecnorm.pkl\"; else RESUME_VECNORM=\"\"; fi",
            ]
        )
    else:
        lines.extend(
            [
                "RESUME_MODEL=\"\"",
                "RESUME_VECNORM=\"\"",
            ]
        )
    lines.extend(
        [
        f"LOG_FILE=\"$LOG_DIR/{train_mode}_{log_prefix}_$(date +%Y%m%d_%H%M%S).log\"",
        "echo \"Project: $PROJECT\"",
        "echo \"Train mode: $TRAIN_MODE\"",
        "if [ -n \"${RESUME_MODEL:-}\" ]; then echo \"Resume model: $RESUME_MODEL\"; fi",
        "if [ -n \"${RESUME_VECNORM:-}\" ]; then echo \"Resume vecnorm: $RESUME_VECNORM\"; fi",
        "echo \"Training log: $LOG_FILE\"",
        "cd \"$CODE_DIR\"",
        "export "
        f"{config_export}"
        f"MUJOCO_TRAIN_LAYOUT_ROOT={quote_remote(remote_root)} "
        f"MUJOCO_TRAIN_PROJECT_SLUG={quote_remote(project_slug)} "
        "MKL_THREADING_LAYER=GNU "
        "OMP_NUM_THREADS=1 "
        "MKL_NUM_THREADS=1 "
        "OPENBLAS_NUM_THREADS=1 "
        "NUMEXPR_NUM_THREADS=1",
        "CMD=(/root/anaconda3/bin/python -u train.py "
        f"--project {quote_remote(project_slug)} "
        f"--total-timesteps {int(total_timesteps)} "
        f"--n-envs {int(n_envs)} "
        f"{reset_args})",
        *(
            [
                'if [ -n "${RESUME_MODEL:-}" ]; then CMD+=(--resume "$RESUME_MODEL"); fi',
                'if [ -n "${RESUME_VECNORM:-}" ]; then CMD+=(--resume-vecnorm "$RESUME_VECNORM"); fi',
            ]
            if train_mode == "resume"
            else []
        ),
        f"if [ -n {quote_remote(extra_args)} ]; then read -r -a EXTRA_ARGS <<< {quote_remote(extra_args)}; CMD+=(\"${{EXTRA_ARGS[@]}}\"); fi",
        "nohup bash -lc \"${CMD[*]} 2>&1 | tee $LOG_FILE\" >/tmp/remote_training_launcher.log 2>&1 &",
        "PID=$!",
        "echo \"TRAIN_PID=$PID\"",
        "echo \"LOG_FILE=$LOG_FILE\"",
        ]
    )
    return "bash -lc " + shlex.quote("\n".join(lines))


def remote_status_script(remote_root: str, project_slug: str) -> str:
    """Return a remote shell script that prints project training status."""
    log_dir = f"{remote_root}/runs/{project_slug}/logs/{project_slug}"
    pattern = f"[t]rain.py --project {project_slug}"
    return "bash -lc " + shlex.quote(
        "echo 'Training processes:' && "
        f"pgrep -af {shlex.quote(pattern)} || true && "
        "echo && echo 'Latest logs:' && "
        f"ls -1t {shlex.quote(log_dir)}/*.log 2>/dev/null | head -n 5 || true && "
        f"latest=$(ls -1t {shlex.quote(log_dir)}/*.log 2>/dev/null | head -n 1 || true); "
        "if [ -n \"$latest\" ]; then echo && echo \"Tail: $latest\" && tail -n 30 \"$latest\"; fi"
    )


def main() -> int:
    """CLI entry point."""
    args = parse_args()
    env_values = {**parse_env_file(args.env_file), **os.environ}
    deploy_args = parse_deploy_args(["--env-file", str(args.env_file)])
    config = build_config(deploy_args, env_values)
    project_slug = args.project or config.project_slug
    backend = select_backend(config)
    runner = NativeRunner(dry_run=False, secrets=[config.remote_password or ""])
    command_env = build_askpass_env(config) if backend == "askpass" else None

    if args.status:
        remote_command = remote_status_script(config.remote_root, project_slug)
    else:
        total_timesteps = int(
            env_project_value(
                env_values,
                project_slug,
                "RESUME_TOTAL_TIMESTEPS",
                str(DEFAULT_TOTAL_TIMESTEPS),
            )
        )
        n_envs = int(
            env_project_value(env_values, project_slug, "RESUME_N_ENVS", str(DEFAULT_N_ENVS))
        )
        reset_noise_raw = env_project_value(
            env_values,
            project_slug,
            "RESUME_RESET_NOISE_SCALE",
            str(DEFAULT_RESET_NOISE_SCALE),
        )
        reset_noise_scale = float(reset_noise_raw) if reset_noise_raw else None
        config_overrides = env_project_value(env_values, project_slug, "RESUME_CONFIG", "")
        extra_args = env_project_value(env_values, project_slug, "RESUME_EXTRA_ARGS", "")
        train_mode = env_project_value(env_values, project_slug, "TRAIN_MODE", "resume").lower()
        remote_command = remote_start_script(
            remote_root=config.remote_root,
            project_slug=project_slug,
            total_timesteps=total_timesteps,
            n_envs=n_envs,
            reset_noise_scale=reset_noise_scale,
            config_overrides=config_overrides or None,
            extra_args=extra_args,
            train_mode=train_mode,
        )

    runner.run(build_ssh_command(config, backend, remote_command), env=command_env)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(1) from None
