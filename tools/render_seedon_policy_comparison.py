"""Render Seedon teacher/probe policy rollouts to MP4 videos."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from seedon_baseline.env import SeedonStandingConfig, SeedonStandingEnv
from tools.audit_seedon_shuffle_v0 import _load_config


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "seedon" / "reference_teacher_pose_1_4_imitation.json"
DEFAULT_OUT_DIR = REPO_ROOT / "artifacts" / "seedon_debug" / "render"
DEFAULT_TEACHER_MODEL = REPO_ROOT / "models" / "seedon" / "teacher_safe_baseline" / "model.zip"
DEFAULT_TEACHER_VECNORM = REPO_ROOT / "models" / "seedon" / "teacher_safe_baseline" / "vecnorm.pkl"
DEFAULT_PROBE_MODEL = REPO_ROOT / "models" / "seedon" / "latest_model.zip"
DEFAULT_PROBE_VECNORM = REPO_ROOT / "models" / "seedon" / "vecnorm.pkl"


@dataclass(frozen=True)
class RenderSpec:
    """One policy render target."""

    label: str
    model_path: Path
    vecnorm_path: Path
    output_path: Path


def _load_policy_provider(
    model_path: Path,
    vecnorm_path: Path,
    env: SeedonStandingEnv,
) -> Callable[[np.ndarray], np.ndarray]:
    """Load a deterministic policy callable using existing VecNormalize stats."""
    if not model_path.is_file():
        raise FileNotFoundError(f"Policy checkpoint not found: {model_path}")
    if not vecnorm_path.is_file():
        raise FileNotFoundError(f"VecNormalize file not found: {vecnorm_path}")

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    model = PPO.load(str(model_path))
    vec_env = DummyVecEnv([lambda: env])
    vecnorm = VecNormalize.load(str(vecnorm_path), vec_env)
    vecnorm.training = False
    vecnorm.norm_reward = False

    def predict_action(obs: np.ndarray) -> np.ndarray:
        norm_obs = vecnorm.normalize_obs(obs[None, :])
        action, _ = model.predict(norm_obs, deterministic=True)
        return np.asarray(action[0], dtype=np.float64)

    return predict_action


def _make_side_camera() -> object:
    """Create a fixed side-view MuJoCo free camera."""
    import mujoco

    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = 1.45
    camera.azimuth = 90.0
    camera.elevation = -14.0
    camera.lookat[:] = np.array([0.0, 0.0, 0.34], dtype=np.float64)
    return camera


def _render_frame(renderer: object, env: SeedonStandingEnv, camera: object) -> np.ndarray:
    """Render one RGB frame from the raw MuJoCo env."""
    renderer.update_scene(env.data, camera=camera)
    frame = renderer.render()
    return np.asarray(frame, dtype=np.uint8)


def _save_mp4(path: Path, frames: list[np.ndarray], fps: int) -> None:
    """Write frames to an MP4 file."""
    if not frames:
        raise RuntimeError(f"No frames captured for {path}")
    try:
        import imageio.v2 as imageio
    except ImportError as exc:
        raise RuntimeError("Rendering MP4 requires imageio and an ffmpeg backend.") from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, frames, fps=fps)


def render_policy(
    spec: RenderSpec,
    *,
    config: SeedonStandingConfig,
    steps: int,
    seed: int,
    fps: int,
    width: int,
    height: int,
) -> list[np.ndarray]:
    """Render one deterministic Seedon policy rollout."""
    if steps <= 0:
        raise ValueError("steps must be positive.")
    if fps <= 0:
        raise ValueError("fps must be positive.")
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive.")

    import mujoco

    env = SeedonStandingEnv(reset_noise_scale=0.0, reward_config=config)
    renderer = mujoco.Renderer(env.model, height=height, width=width)
    camera = _make_side_camera()
    frames: list[np.ndarray] = []
    try:
        obs, _ = env.reset(seed=seed)
        action_provider = _load_policy_provider(spec.model_path, spec.vecnorm_path, env)
        frames.append(_render_frame(renderer, env, camera))
        for _ in range(steps):
            action = action_provider(obs)
            obs, _, terminated, truncated, _ = env.step(action)
            frames.append(_render_frame(renderer, env, camera))
            if terminated or truncated:
                break
    finally:
        renderer.close()
        env.close()
    _save_mp4(spec.output_path, frames[:steps], fps)
    return frames[:steps]


def _side_by_side(left_frames: list[np.ndarray], right_frames: list[np.ndarray]) -> list[np.ndarray]:
    """Combine two equally sized frame lists into side-by-side frames."""
    count = min(len(left_frames), len(right_frames))
    if count == 0:
        raise RuntimeError("Cannot create side-by-side video without frames.")
    return [
        np.concatenate([left_frames[index], right_frames[index]], axis=1)
        for index in range(count)
    ]


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--teacher-model", type=Path, default=DEFAULT_TEACHER_MODEL)
    parser.add_argument("--teacher-vecnorm", type=Path, default=DEFAULT_TEACHER_VECNORM)
    parser.add_argument("--probe-model", type=Path, default=DEFAULT_PROBE_MODEL)
    parser.add_argument("--probe-vecnorm", type=Path, default=DEFAULT_PROBE_VECNORM)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--height", type=int, default=368)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Render teacher, probe, and side-by-side videos."""
    args = build_parser().parse_args(argv)
    config = _load_config(args.config)
    teacher = RenderSpec(
        "teacher_safe_baseline",
        args.teacher_model,
        args.teacher_vecnorm,
        args.out_dir / "teacher_safe_baseline.mp4",
    )
    probe = RenderSpec(
        "autonomy_probe_v2_25k",
        args.probe_model,
        args.probe_vecnorm,
        args.out_dir / "autonomy_probe_v2_25k.mp4",
    )
    teacher_frames = render_policy(
        teacher,
        config=config,
        steps=args.steps,
        seed=args.seed,
        fps=args.fps,
        width=args.width,
        height=args.height,
    )
    probe_frames = render_policy(
        probe,
        config=config,
        steps=args.steps,
        seed=args.seed,
        fps=args.fps,
        width=args.width,
        height=args.height,
    )
    side_by_side_path = args.out_dir / "teacher_vs_autonomy_probe_v2_25k_side_by_side.mp4"
    _save_mp4(side_by_side_path, _side_by_side(teacher_frames, probe_frames), args.fps)
    print(f"teacher_video: {teacher.output_path}")
    print(f"probe_video: {probe.output_path}")
    print(f"side_by_side_video: {side_by_side_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
