"""PPO training for H1 walking.

Usage:
    python train.py --project h1              # fresh H1 training
    python train.py --project h1 --resume     # resume latest H1 checkpoint
    python train.py --project h1 --smoke      # short H1 sanity run
    python train.py --project h1 --quick      # quick H1 run
    python train.py --project h1 --dr         # H1 with DR + curriculum
    python train.py --dr --dr-ramp-end 0.6 --dr-start-level 0.1
    python train.py --finetune models/best_model.zip --dr  # DR finetune
    python train.py --project grasp --phase full --n-envs 32
"""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
)
from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecNormalize,
)

from h1_baseline.env import H1Env, _DEFAULT_REWARD_SCALES
from robot_learning.training_config import load_h1_train_config
from robot_learning.training_paths import resolve_training_paths

# ── Paths ────────────────────────────────────────────────────────────────
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
H1_CONFIG = load_h1_train_config(REPO_ROOT)
PATHS = resolve_training_paths(
    REPO_ROOT,
    "h1",
    legacy_model_dir="models",
    legacy_log_dir="logs",
    legacy_tb_dir=os.path.join("logs", "tb", "h1"),
)
MODEL_DIR = str(PATHS.models_root)
LOG_DIR = str(PATHS.logs_root)
TB_DIR = str(PATHS.tb_root)
MODEL_PATH = os.path.join(MODEL_DIR, "h1_ppo")
MODEL_DR_PATH = os.path.join(MODEL_DIR, "h1_ppo_dr")
VECNORM_PATH = os.path.join(MODEL_DIR, "h1_vecnorm.pkl")
VECNORM_BEST_PATH = os.path.join(MODEL_DIR, "h1_vecnorm_best.pkl")
VECNORM_DR_PATH = os.path.join(MODEL_DIR, "h1_vecnorm_dr.pkl")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(TB_DIR, exist_ok=True)

# ── Hyperparameters ──────────────────────────────────────────────────────
N_ENVS = int(os.environ.get("H1_N_ENVS", str(H1_CONFIG.n_envs_default)))
TOTAL_TIMESTEPS = H1_CONFIG.total_timesteps
SMOKE_TIMESTEPS = H1_CONFIG.smoke_timesteps
QUICK_TIMESTEPS = H1_CONFIG.quick_timesteps
QUICK_N_ENVS = H1_CONFIG.quick_n_envs
N_STEPS = H1_CONFIG.n_steps
N_EPOCHS = H1_CONFIG.n_epochs
GAMMA = H1_CONFIG.gamma
GAE_LAMBDA = H1_CONFIG.gae_lambda
ENT_COEF = H1_CONFIG.ent_coef
VF_COEF = H1_CONFIG.vf_coef
MAX_GRAD_NORM = H1_CONFIG.max_grad_norm
NET_ARCH = H1_CONFIG.net_arch
SAVE_FREQ = H1_CONFIG.save_freq

LR_FLOOR = H1_CONFIG.learning_rate_floor
CLIP_FLOOR = H1_CONFIG.clip_range_floor


def LEARNING_RATE(progress: float) -> float:
    """Linear decay with floor: 3e-4 → LR_FLOOR over training."""
    return max(H1_CONFIG.learning_rate_initial * progress, LR_FLOOR)


def CLIP_RANGE(progress: float) -> float:
    """Sync clip_range decay with LR so update magnitude scales down together."""
    return max(H1_CONFIG.clip_range_initial * progress, CLIP_FLOOR)


# ── Fine-tune hyperparameters (used when --finetune is specified) ─────────
FINETUNE_ENT_COEF = H1_CONFIG.finetune_ent_coef
FINETUNE_LR_FLOOR = H1_CONFIG.finetune_learning_rate_floor
FINETUNE_CLIP_FLOOR = H1_CONFIG.finetune_clip_range_floor


def FINETUNE_LEARNING_RATE(progress: float) -> float:
    """Linear decay with floor: 1e-4 → FINETUNE_LR_FLOOR over DR finetune."""
    return max(H1_CONFIG.finetune_learning_rate_initial * progress, FINETUNE_LR_FLOOR)


def FINETUNE_CLIP_RANGE(progress: float) -> float:
    """Sync clip_range for DR finetune."""
    return max(H1_CONFIG.finetune_clip_range_initial * progress, FINETUNE_CLIP_FLOOR)


# ── Curriculum Learning ─────────────────────────────────────────────────
CURRICULUM_STAGES = H1_CONFIG.curriculum_stages


def _compute_batch_size(n_envs: int) -> int:
    """batch_size must satisfy: batch_size <= n_steps * n_envs (SB3 requirement).

    Scales with n_envs so smoke/quick modes don't crash.
    """
    return max(512, min(n_envs * N_STEPS // 8, n_envs * N_STEPS))


def get_config(finetune_from: str | None, n_envs: int, total_steps: int,
               batch_size: int) -> dict:
    """Collect all hyperparameters into a single dict."""
    lr = FINETUNE_LEARNING_RATE if finetune_from else LEARNING_RATE
    ent = FINETUNE_ENT_COEF if finetune_from else ENT_COEF
    clip = FINETUNE_CLIP_RANGE if finetune_from else CLIP_RANGE
    return {
        "n_envs": n_envs,
        "total_timesteps": total_steps,
        "n_steps": N_STEPS,
        "batch_size": batch_size,
        "n_epochs": N_EPOCHS,
        "learning_rate": lr(1.0),
        "lr_schedule": "linear+floor",
        "gamma": GAMMA,
        "gae_lambda": GAE_LAMBDA,
        "clip_range": clip(1.0),
        "clip_range_schedule": "linear+floor",
        "ent_coef": ent,
        "vf_coef": VF_COEF,
        "max_grad_norm": MAX_GRAD_NORM,
        "net_arch": NET_ARCH,
        "reward_scales": dict(_DEFAULT_REWARD_SCALES),
    }


def config_hash(cfg: dict) -> str:
    """Short hash for identifying experiment configs."""
    raw = json.dumps(cfg, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:8]


def _git_commit_short() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(HERE),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        return out or "unknown"
    except Exception:
        return "unknown"


# ── Callbacks ────────────────────────────────────────────────────────────

class VecNormCheckpointCallback(BaseCallback):
    """Save VecNormalize statistics alongside each model checkpoint.

    Uses ``num_timesteps`` (true env-step count), unlike CheckpointCallback
    which counts _on_step calls — so save_freq here is the actual timestep
    interval regardless of n_envs.

    Keeps only the last ``keep_last_n`` versioned files to avoid unbounded
    accumulation on long runs.
    """

    def __init__(self, save_freq: int, vec_norm_path: str, keep_last_n: int = 5):
        super().__init__(0)
        self._save_freq = save_freq
        self._vec_norm_path = vec_norm_path
        self._save_dir = os.path.dirname(vec_norm_path)
        self._last_save = 0
        self._keep_last_n = keep_last_n
        self._versioned_files: deque = deque()

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_save >= self._save_freq:
            self._last_save = self.num_timesteps
            self.training_env.save(self._vec_norm_path)
            versioned = os.path.join(
                self._save_dir,
                f"h1_vecnorm_{self.num_timesteps}_steps.pkl",
            )
            self.training_env.save(versioned)
            self._versioned_files.append(versioned)
            while len(self._versioned_files) > self._keep_last_n:
                old = self._versioned_files.popleft()
                if os.path.exists(old):
                    os.remove(old)
        return True


class BestVecNormCallback(BaseCallback):
    """Save VecNormalize whenever EvalCallback updates best_model.zip.

    Saves ``eval_env`` specifically. SB3's EvalCallback calls
    ``sync_envs_normalization(training_env, eval_env)`` before every eval,
    so eval_env.obs_rms == training_env.obs_rms at the moment best_model.zip
    is written — saving eval_env is therefore equivalent in content but
    makes the pairing (best_model.zip ↔ eval_env stats) explicit.

    Must come AFTER EvalCallback in the callback list so the mtime check
    sees the fresh best_model.zip.
    """

    def __init__(self, best_model_dir: str, vecnorm_best_path: str,
                 eval_env: VecNormalize):
        super().__init__(0)
        self._best_zip = os.path.join(best_model_dir, "best_model.zip")
        self._out_path = vecnorm_best_path
        self._eval_env = eval_env
        self._last_mtime = 0.0

    def _on_step(self) -> bool:
        if os.path.exists(self._best_zip):
            mtime = os.path.getmtime(self._best_zip)
            if mtime > self._last_mtime:
                self._last_mtime = mtime
                self._eval_env.save(self._out_path)
        return True


class RewardBreakdownCallback(BaseCallback):
    """Log per-component reward means to TensorBoard every N steps."""

    LOG_FREQ = 4096

    def __init__(self):
        super().__init__(0)
        self._reward_keys: list[str] = []
        self._buffers: dict[str, deque] = {}
        self._last_log = 0

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            for k, v in info.items():
                if not k.startswith("reward_"):
                    continue
                if k not in self._buffers:
                    self._buffers[k] = deque(maxlen=200)
                    self._reward_keys.append(k)
                self._buffers[k].append(v)

        if self.num_timesteps - self._last_log >= self.LOG_FREQ:
            self._last_log = self.num_timesteps
            for k in self._reward_keys:
                buf = self._buffers[k]
                if buf:
                    tag = f"reward/{k.replace('reward_', '')}"
                    self.logger.record(tag, float(np.mean(buf)))
        return True


class CurriculumCallback(BaseCallback):
    """Adjust target velocity and DR intensity based on training progress."""

    _UPDATE_FREQ = 1000  # throttle IPC; SubprocVecEnv env_method is expensive at scale

    def __init__(
        self,
        total_timesteps: int,
        stages: list[tuple[float, float]],
        enable_dr_ramp: bool = False,
        dr_ramp_end: float = 0.35,
        dr_start_level: float = 0.0,
    ):
        super().__init__(0)
        self._total = total_timesteps
        self._stages = sorted(stages, key=lambda s: s[0])
        self._last_update = 0
        self._enable_dr_ramp = enable_dr_ramp
        self._dr_ramp_end = max(dr_ramp_end, 1e-6)
        self._dr_start_level = float(np.clip(dr_start_level, 0.0, 1.0))

    def _on_training_start(self) -> None:
        if self._enable_dr_ramp:
            self.training_env.env_method("set_dr_level", self._dr_start_level)
            self.logger.record("curriculum/dr_level", self._dr_start_level)

    def _interpolate_velocity(self, progress: float) -> float:
        if progress <= self._stages[0][0]:
            return self._stages[0][1]
        if progress >= self._stages[-1][0]:
            return self._stages[-1][1]
        for i in range(len(self._stages) - 1):
            p0, v0 = self._stages[i]
            p1, v1 = self._stages[i + 1]
            if p0 <= progress <= p1:
                t = (progress - p0) / (p1 - p0) if p1 > p0 else 0.0
                return v0 + t * (v1 - v0)
        return self._stages[-1][1]

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_update < self._UPDATE_FREQ:
            return True
        self._last_update = self.num_timesteps
        progress = self.num_timesteps / self._total
        vel = self._interpolate_velocity(progress)
        self.training_env.env_method("set_target_velocity", vel)
        self.logger.record("curriculum/target_vel", vel)
        if self._enable_dr_ramp:
            span = max(1.0 - self._dr_start_level, 1e-6)
            dr_level = min(self._dr_start_level + (progress / self._dr_ramp_end) * span, 1.0)
            self.training_env.env_method("set_dr_level", dr_level)
            self.logger.record("curriculum/dr_level", dr_level)
        return True


class TrainingProgressCallback(BaseCallback):
    PRINT_FREQ = 20_000

    def __init__(self, total_timesteps: int):
        super().__init__(0)
        self.total_timesteps = total_timesteps
        self._ep_rewards: deque = deque(maxlen=50)
        self._ep_lengths: deque = deque(maxlen=50)
        self._best = -np.inf
        self._n = 0
        self._t0 = 0.0
        self._last = 0

    def _on_training_start(self) -> None:
        self._t0 = time.time()
        print(
            f"\n{'Steps':>12}  {'Eps':>6}  {'MeanR':>9}  "
            f"{'MeanLen':>8}  {'BestR':>8}  {'FPS':>6}  "
            f"{'ETA':>9}",
        )
        print("-" * 72)

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            ep = info.get("episode")
            if ep:
                self._ep_rewards.append(ep["r"])
                self._ep_lengths.append(ep["l"])
                self._n += 1
                if ep["r"] > self._best:
                    self._best = ep["r"]

        if self.num_timesteps - self._last >= self.PRINT_FREQ:
            self._last = self.num_timesteps
            elapsed = time.time() - self._t0
            fps = int(self.num_timesteps / elapsed) if elapsed > 0 else 0
            remain = ((self.total_timesteps - self.num_timesteps) / fps
                      if fps > 0 else 0)
            mean_r = (np.mean(self._ep_rewards)
                      if self._ep_rewards else float("nan"))
            mean_l = (np.mean(self._ep_lengths)
                      if self._ep_lengths else float("nan"))
            m, s = divmod(int(remain), 60)
            h, m = divmod(m, 60)
            print(
                f"{self.num_timesteps:>12,}  {self._n:>6}  "
                f"{mean_r:>9.1f}  {mean_l:>8.1f}  "
                f"{self._best:>8.1f}  {fps:>6}  "
                f"{h:02d}:{m:02d}:{s:02d}",
            )
        return True


# ── Env factory ──────────────────────────────────────────────────────────

def make_env(
    rank: int = 0,
    seed: int = 0,
    domain_randomization: bool = False,
    randomize_commands: bool = False,
    reward_scales: dict | None = None,
):
    def _init():
        env = H1Env(
            domain_randomization=domain_randomization,
            randomize_commands=randomize_commands,
            reward_scales=reward_scales,
        )
        env = TimeLimit(env, max_episode_steps=H1_CONFIG.max_episode_steps)
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init


def _find_latest_checkpoint(prefer_dr: bool) -> str | None:
    """Return the best resume candidate, or None.

    Resume must pick a **chronologically latest** checkpoint of the matching
    mode (DR vs base). Mixing modes or picking best_model.zip would resume
    from the wrong training run.

    Priority:
      1. Highest-step ``h1_ppo_dr_*_steps.zip`` (if prefer_dr) or
         ``h1_ppo_*_steps.zip`` (if not prefer_dr, excluding DR files).
      2. Final model of matching mode: ``h1_ppo_dr.zip`` or ``h1_ppo.zip``.
      3. best_model.zip only as a last resort (EvalCallback-selected, not
         chronological — prints a warning).
    """
    import glob

    def _is_dr(path: str) -> bool:
        return os.path.basename(path).startswith("h1_ppo_dr")

    def _step_of(path: str) -> int | None:
        name = os.path.basename(path).replace(".zip", "")
        parts = name.split("_steps")
        if len(parts) < 2:
            return None  # final model (no _steps suffix)
        try:
            return int(parts[0].split("_")[-1])
        except ValueError:
            return None

    all_files = glob.glob(os.path.join(MODEL_DIR, "h1_ppo*.zip"))
    matching = [p for p in all_files if _is_dr(p) == prefer_dr]

    # 1) Highest-step checkpoint of matching mode.
    step_ckpts = [(s, p) for p in matching if (s := _step_of(p)) is not None]
    if step_ckpts:
        return max(step_ckpts)[1]

    # 2) Final model of matching mode.
    final_name = "h1_ppo_dr.zip" if prefer_dr else "h1_ppo.zip"
    final_path = os.path.join(MODEL_DIR, final_name)
    if os.path.exists(final_path):
        return final_path

    # 3) Fallback to best_model.zip only when no matching artefact found.
    best = os.path.join(MODEL_DIR, "best_model.zip")
    if os.path.exists(best):
        print("[warn] No step-numbered or final checkpoint of matching mode "
              "found; falling back to best_model.zip "
              "(may not be chronologically latest).")
        return best

    return None


def build_vec_env(
    n_envs: int,
    use_subproc: bool = True,
    domain_randomization: bool = False,
    randomize_commands: bool = False,
    reward_scales: dict | None = None,
):
    fns = [
        make_env(i, domain_randomization=domain_randomization,
                 randomize_commands=randomize_commands,
                 reward_scales=reward_scales)
        for i in range(n_envs)
    ]
    if use_subproc and n_envs > 1:
        try:
            vec = SubprocVecEnv(fns, start_method="spawn")
        except Exception as e:
            print(f"[warn] SubprocVecEnv failed ({e}); falling back to DummyVecEnv")
            vec = DummyVecEnv(fns)
    else:
        vec = DummyVecEnv(fns)
    return VecNormalize(
        vec, norm_obs=True, norm_reward=True,
        clip_obs=10.0, gamma=GAMMA,
    )


# ── Training ─────────────────────────────────────────────────────────────

def train(
    resume: bool = False,
    smoke: bool = False,
    quick: bool = False,
    domain_randomization: bool = False,
    ablate: str | None = None,
    finetune_from: str | None = None,
    dr_ramp_end: float = 0.35,
    dr_start_level: float = 0.0,
):
    if smoke:
        total_steps, n_envs = SMOKE_TIMESTEPS, 2
    elif quick:
        total_steps, n_envs = QUICK_TIMESTEPS, QUICK_N_ENVS
    else:
        total_steps, n_envs = TOTAL_TIMESTEPS, N_ENVS

    effective_batch_size = _compute_batch_size(n_envs)

    # Ablation: zero out a specific reward term.
    reward_scales = None
    if ablate:
        reward_scales = dict(_DEFAULT_REWARD_SCALES)
        if ablate not in reward_scales:
            print(f"[error] Unknown reward term '{ablate}'. Available:")
            for k in sorted(_DEFAULT_REWARD_SCALES):
                print(f"  {k}")
            return
        reward_scales[ablate] = 0.0
        print(f"[ablation] Zeroed reward term: {ablate}")

    # Save experiment config.
    cfg = get_config(finetune_from, n_envs, total_steps, effective_batch_size)
    cfg["domain_randomization"] = domain_randomization
    cfg["ablate"] = ablate
    cfg["finetune_from"] = finetune_from
    run_id = config_hash(cfg)
    run_dir = os.path.join(LOG_DIR, f"run_{run_id}")
    os.makedirs(run_dir, exist_ok=True)

    cfg["run_id"] = run_id
    cfg["timestamp"] = datetime.now().isoformat()
    cfg["smoke"] = smoke
    cfg_path = os.path.join(run_dir, "config.json")
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2, default=str)

    manifest = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(),
        "git_commit": _git_commit_short(),
        "python": sys.version,
        "platform": platform.platform(),
        "command": " ".join(sys.argv),
        "config_path": cfg_path,
    }
    manifest_path = os.path.join(run_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"Experiment config saved: {cfg_path}")
    print(f"Run manifest saved: {manifest_path}")
    print(f"Run ID: {run_id}")
    print(f"Artifacts: models={MODEL_DIR} logs={LOG_DIR} tb={TB_DIR}")

    # Training env: DR enables both physics and command randomization.
    vec_env = build_vec_env(
        n_envs, use_subproc=not smoke,
        domain_randomization=domain_randomization,
        randomize_commands=domain_randomization,
        reward_scales=reward_scales,
    )
    # Eval env: mirrors training DR for physics (so best_model is selected
    # on the distribution we care about), but keeps commands FIXED so eval
    # metrics are comparable across checkpoints.
    eval_env = build_vec_env(
        1, use_subproc=False,
        domain_randomization=domain_randomization,
        randomize_commands=False,
        reward_scales=reward_scales,
    )
    eval_env.training = False
    eval_env.norm_reward = False

    # Finetune writes DR stats to a separate file so base-training stats
    # (h1_vecnorm.pkl) are preserved for non-DR eval.
    # VecNorm path must follow the same DR/base split as the model artefact
    # (save_model_path below). Previously only --finetune branched to
    # VECNORM_DR_PATH, leaving fresh --dr runs overwriting the base VecNorm.
    vecnorm_path = VECNORM_DR_PATH if domain_randomization else VECNORM_PATH

    resume_path = _find_latest_checkpoint(prefer_dr=domain_randomization) if resume else None

    if finetune_from:
        if not os.path.exists(finetune_from):
            print(f"[error] Finetune source not found: {finetune_from}")
            vec_env.close(); eval_env.close()
            return
        # Priority: best.pkl (paired with best_model.zip) > latest.pkl
        source_vecnorm = None
        for candidate in [VECNORM_BEST_PATH, VECNORM_PATH]:
            if os.path.exists(candidate):
                source_vecnorm = candidate
                break
        if source_vecnorm is None:
            print("[error] No VecNormalize found for finetune source.")
            print("  Run a base training first (python train.py)")
            vec_env.close(); eval_env.close()
            return
        print(f"Loading VecNormalize from {source_vecnorm}")

        vec_env = VecNormalize.load(source_vecnorm, vec_env.venv)
        vec_env.training = True
        vec_env.norm_reward = True
        eval_env = VecNormalize.load(source_vecnorm, eval_env.venv)
        eval_env.training = False
        eval_env.norm_reward = False
        print(f"Fine-tuning from {finetune_from}")
        model = PPO.load(
            finetune_from, env=vec_env,
            custom_objects={
                "learning_rate": FINETUNE_LEARNING_RATE,
                "clip_range": FINETUNE_CLIP_RANGE,
                "ent_coef": FINETUNE_ENT_COEF,
                "batch_size": effective_batch_size,
            },
        )
        model.tensorboard_log = TB_DIR
        print(f"  lr    : {FINETUNE_LEARNING_RATE(1.0):.2e} → {FINETUNE_LR_FLOOR:.2e} (linear+floor)")
        print(f"  clip  : {FINETUNE_CLIP_RANGE(1.0):.2f} → {FINETUNE_CLIP_FLOOR:.2f} (linear+floor)")
        print(f"  ent   : {FINETUNE_ENT_COEF}")
    elif resume_path:
        print(f"Resuming from {resume_path}")
        # If resuming a DR run, prefer DR vecnorm; otherwise base.
        resume_vecnorm = None
        for candidate in (
            [VECNORM_DR_PATH, VECNORM_PATH] if domain_randomization
            else [VECNORM_PATH, VECNORM_DR_PATH]
        ):
            if os.path.exists(candidate):
                resume_vecnorm = candidate
                break
        if resume_vecnorm:
            print(f"  VecNormalize: {resume_vecnorm}")
            vec_env = VecNormalize.load(resume_vecnorm, vec_env.venv)
            vec_env.training = True
            eval_env = VecNormalize.load(resume_vecnorm, eval_env.venv)
            eval_env.training = False
            eval_env.norm_reward = False
        else:
            print("[warn] No VecNormalize found on resume; stats will be re-estimated. "
                  "Early eval rewards may be unreliable.")
        lr_schedule = FINETUNE_LEARNING_RATE if domain_randomization else LEARNING_RATE
        clip_schedule = FINETUNE_CLIP_RANGE if domain_randomization else CLIP_RANGE
        ent_value = FINETUNE_ENT_COEF if domain_randomization else ENT_COEF
        model = PPO.load(
            resume_path, env=vec_env,
            custom_objects={
                "learning_rate": lr_schedule,
                "clip_range": clip_schedule,
                "ent_coef": ent_value,
                "batch_size": effective_batch_size,
            },
        )
    else:
        model = PPO(
            "MlpPolicy",
            vec_env,
            n_steps=N_STEPS,
            batch_size=effective_batch_size,
            n_epochs=N_EPOCHS,
            learning_rate=LEARNING_RATE,
            gamma=GAMMA,
            gae_lambda=GAE_LAMBDA,
            clip_range=CLIP_RANGE,
            ent_coef=ENT_COEF,
            vf_coef=VF_COEF,
            max_grad_norm=MAX_GRAD_NORM,
            policy_kwargs={"net_arch": NET_ARCH},
            tensorboard_log=TB_DIR,
            verbose=0,
        )

    # Callback frequencies: CheckpointCallback.save_freq counts _on_step
    # calls (== rollout-batch count); one call == n_envs timesteps. Divide
    # by n_envs so real save interval is SAVE_FREQ timesteps, matching
    # VecNormCheckpointCallback.
    save_freq_steps = max(SAVE_FREQ // n_envs, 1)
    ckpt_prefix = "h1_ppo_dr" if domain_randomization else "h1_ppo"

    callbacks = [
        TrainingProgressCallback(total_steps),
        RewardBreakdownCallback(),
        VecNormCheckpointCallback(SAVE_FREQ, vecnorm_path),
    ]
    if domain_randomization:
        callbacks.append(
            CurriculumCallback(
                total_steps,
                CURRICULUM_STAGES,
                enable_dr_ramp=True,
                dr_ramp_end=dr_ramp_end,
                dr_start_level=dr_start_level,
            ),
        )
    # Best artifacts must not leak between base and DR training runs:
    # EvalCallback always names the file "best_model.zip", so without a
    # per-mode save dir a DR run would overwrite the base best_model.zip
    # (and BestVecNormCallback the base h1_vecnorm_best.pkl). Use a
    # dedicated subdir for DR so base artefacts stay intact.
    if domain_randomization:
        best_artifact_dir = os.path.join(MODEL_DIR, "dr_best")
        os.makedirs(best_artifact_dir, exist_ok=True)
        vecnorm_best_path = os.path.join(
            best_artifact_dir, "h1_vecnorm_best.pkl",
        )
    else:
        best_artifact_dir = MODEL_DIR
        vecnorm_best_path = VECNORM_BEST_PATH

    if not smoke:
        callbacks += [
            CheckpointCallback(
                save_freq=save_freq_steps,
                save_path=MODEL_DIR,
                name_prefix=ckpt_prefix,
            ),
            EvalCallback(
                eval_env,
                best_model_save_path=best_artifact_dir,
                log_path=LOG_DIR,
                eval_freq=save_freq_steps,
                n_eval_episodes=20,
                deterministic=True,
                verbose=1,
            ),
            # BestVecNormCallback MUST come AFTER EvalCallback: EvalCallback
            # writes best_model.zip first, then this detects the new mtime
            # and snapshots eval_env.obs_rms (already synced from vec_env).
            BestVecNormCallback(
                best_artifact_dir, vecnorm_best_path, eval_env,
            ),
        ]

    save_model_path = MODEL_DR_PATH if domain_randomization else MODEL_PATH

    status = "completed"
    try:
        model.learn(
            total_timesteps=total_steps,
            callback=callbacks,
            reset_num_timesteps=(finetune_from is not None) or (not resume),
        )
    except KeyboardInterrupt:
        status = "interrupted_by_user"
        print("\n[warn] Training interrupted — saving current state.")
    except BaseException as e:
        status = f"failed:{type(e).__name__}"
        print(f"\n[error] Training crashed: {e} — attempting partial save.")
        raise
    finally:
        # Best-effort save on any exit path. Checkpoints saved by callbacks
        # are still intact even if these throw.
        try:
            model.save(save_model_path)
            vec_env.save(vecnorm_path)
            print(f"\nSaved model -> {save_model_path}.zip")
            print(f"VecNormalize -> {vecnorm_path}")
        except Exception as save_err:
            print(f"[warn] Final save failed: {save_err}")

        manifest["finished_at"] = datetime.now().isoformat()
        manifest["status"] = status
        manifest["num_timesteps"] = int(model.num_timesteps)
        manifest["model_path"] = save_model_path + ".zip"
        manifest["vecnorm_path"] = vecnorm_path
        try:
            with open(manifest_path, "w") as f:
                json.dump(manifest, f, indent=2, default=str)
        except Exception as mf_err:
            print(f"[warn] Manifest update failed: {mf_err}")

        vec_env.close()
        eval_env.close()


def main(argv: list[str] | None = None) -> int:
    """Run H1 walking training."""
    forwarded = list(sys.argv[1:] if argv is None else argv)
    p = argparse.ArgumentParser()
    p.add_argument("--resume", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--dr", action="store_true",
                   help="Enable Domain Randomization + Curriculum Learning")
    p.add_argument("--ablate", type=str, default=None,
                   help="Zero out a reward term for ablation study (e.g. --ablate contact)")
    p.add_argument("--finetune", type=str, default=None, metavar="MODEL_PATH",
                   help="Fine-tune from a pre-trained model "
                        "(e.g. --finetune models/best_model.zip)")
    p.add_argument("--dr-ramp-end", type=float, default=0.35,
                   help="Progress ratio where DR reaches full strength "
                        "(only used with --dr, default 0.35)")
    p.add_argument("--dr-start-level", type=float, default=0.0,
                   help="Initial DR level in [0,1] when training starts "
                        "(only used with --dr, default 0.0)")
    args = p.parse_args(forwarded)
    train(
        resume=args.resume,
        smoke=args.smoke,
        quick=args.quick,
        domain_randomization=args.dr,
        ablate=args.ablate,
        finetune_from=args.finetune,
        dr_ramp_end=args.dr_ramp_end,
        dr_start_level=args.dr_start_level,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
