#!/usr/bin/env python3
"""Step 2A: Collect frozen-teacher rollout buffer for F_T training.

Loads the OLD teacher (frozen), runs it in env for N transitions, dumps a
compact time-major buffer of (h_t, a_t, structured_targets) per step.
Episode boundaries are tracked so the F_T trainer can build K-step windows
that don't cross episode breaks.

Buffer format (single file, fp32 by default):
    h:                 (T, E, dim_state) teacher latent at each step
    a:                 (T, E, dim_act)   executed teacher action
    rel_goal_delta:    (T, E, 9)         obs_{t+1}.rel_goal - obs_t.rel_goal
    object_delta:      (T, E, 13)        obs_{t+1}.object_state - obs_t.object_state
    contact:           (T, E, 1)         obs_{t+1}.touch (clamped to [0,1])
    progress:          (T, E, 1)         ||rel_t[:3]|| - ||rel_{t+1}[:3]||
    done:              (T, E) bool       episode terminated AFTER step t
    valid:             (T, E) bool       transition is not across reset boundary
    episode_id:        (T, E) int64      unique episode id per env lane
    meta:              dict             dim_state, dim_act, num_env, etc.

All env states + actions are stored after teacher.actor_net runs on h_t, so
F_T's training pair (h_t, a_t) -> h_{t+1} is well-defined.

Optional --noisy-frac: replaces a fraction of teacher actions with
clipped-gaussian noise around the teacher action, to give F_T coverage
beyond on-policy trajectories. NOTE: when computing F_T targets, we use the
EXECUTED action (whether noisy or not), so dynamics targets stay consistent.
"""
from __future__ import annotations

# IsaacGym must be imported before torch.
from isaacgym import gymtorch  # noqa: F401
from isaacgym import gymapi    # noqa: F401

import os
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Mapping
from dataclasses import replace

import torch as th
from gym import spaces

from train_ppo_arm import (
    load_agent, load_env,
)
from train_rma import Config
from util.config import recursive_replace_map
from util.hydra_cli import hydra_cli
from env.util import set_seed
from models.common import map_struct, map_tensor
from train.source_lineage import (
    TRAIN_SOURCE_FILES,
    canonical_json_sha256,
    sha256_file,
)


REPO_ROOT = Path(__file__).resolve().parents[3]


def _required_artifact(path, label):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    return path, sha256_file(path)


def _git_commit():
    return subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True,
    ).strip()


def _get_obs_field(obs, key, fallback_shape, device, dtype=th.float32):
    """Best-effort obs field fetch with shape-fallback."""
    if isinstance(obs, dict) and key in obs:
        return obs[key]
    return th.zeros(fallback_shape, device=device, dtype=dtype)


def _compute_structured(obs_t, obs_tp1, num_env, device):
    """Per-step structured targets describing transition t -> t+1."""
    rel_t   = _get_obs_field(obs_t,   "rel_goal",     (num_env, 9), device)
    rel_tp1 = _get_obs_field(obs_tp1, "rel_goal",     (num_env, 9), device)
    obj_t   = _get_obs_field(obs_t,   "object_state", (num_env, 13), device)
    obj_tp1 = _get_obs_field(obs_tp1, "object_state", (num_env, 13), device)
    touch   = _get_obs_field(obs_tp1, "touch",        (num_env, 1),  device)

    rel_t   = rel_t  [..., :9]
    rel_tp1 = rel_tp1[..., :9]
    obj_t   = obj_t  [..., :13]
    obj_tp1 = obj_tp1[..., :13]

    rel_goal_delta = (rel_tp1 - rel_t).detach()
    object_delta   = (obj_tp1 - obj_t).detach()
    contact        = touch.reshape(num_env, -1)[..., :1].detach().clamp(0.0, 1.0)
    dist_t   = th.linalg.norm(rel_t  [..., :3], dim=-1, keepdim=True)
    dist_tp1 = th.linalg.norm(rel_tp1[..., :3], dim=-1, keepdim=True)
    progress = (dist_t - dist_tp1).detach()

    return {
        "rel_goal_delta": rel_goal_delta,
        "object_delta":   object_delta,
        "contact":        contact,
        "progress":       progress,
    }


@hydra_cli(
    Config,
    config_path=str(Path(__file__).resolve().parents[2] / 'src' / 'data' / 'cfg'),
    config_name='train_rl',
)
def main(cfg: Config):
    cfg = recursive_replace_map(cfg, {'finalize': True})
    if cfg.global_device is not None:
        th.cuda.set_device(cfg.global_device)
    th.backends.cudnn.benchmark = True
    set_seed(cfg.env.seed)

    # ---- Knobs (read via env vars; hydra cfg paths kept stable for teacher) ----
    num_steps   = int(os.environ.get("BUFFER_STEPS",     "5000"))
    save_dir    = os.environ.get(
        "BUFFER_OUT_DIR",
        str(REPO_ROOT / "output" / "teacher_buffer_v1"),
    )
    noisy_frac  = float(os.environ.get("BUFFER_NOISY_FRAC", "0.0"))   # 0..1
    noise_std   = float(os.environ.get("BUFFER_NOISE_STD",  "0.1"))
    use_fp16    = os.environ.get(  "BUFFER_FP16", "0") == "1"
    artifact_class = os.environ.get(
        "ARTIFACT_CLASS", "NONPREHENSILE_SOFT_DEV")
    if artifact_class not in {"NONPREHENSILE_SOFT_DEV"}:
        raise ValueError(f"unsupported buffer artifact class: {artifact_class}")

    teacher_checkpoint, teacher_checkpoint_sha = _required_artifact(
        cfg.load_ckpt, "teacher checkpoint")
    checkpoint_name = teacher_checkpoint.stem
    if not checkpoint_name.startswith("step-"):
        raise ValueError(
            "F_T buffer collection requires an exact step-N teacher checkpoint")
    step_tag = checkpoint_name[len("step-"):]
    teacher_env_stat, teacher_env_stat_sha = _required_artifact(
        teacher_checkpoint.parent.parent / "stat" / f"env-{step_tag}.ckpt",
        "paired teacher env stat")
    icp_checkpoint, icp_checkpoint_sha = _required_artifact(
        cfg.icp_obs.icp.ckpt, "ICP checkpoint")
    split_file, split_sha = _required_artifact(
        cfg.env.single_object_scene.filter_file, "object split")
    num_object_types = int(cfg.env.single_object_scene.num_object_types)
    source_files = {
        relative_path: sha256_file(REPO_ROOT / relative_path)
        for relative_path in TRAIN_SOURCE_FILES
    }
    source_snapshot_sha = canonical_json_sha256(source_files)
    repo_commit = _git_commit()
    dgn_root = Path(cfg.env.single_object_scene.dgn.data_path).expanduser().resolve()
    dgn_pose_root = Path(
        cfg.env.single_object_scene.dgn.pose_path).expanduser().resolve()
    if not dgn_pose_root.is_dir():
        raise FileNotFoundError(dgn_pose_root)
    dgn_metadata_sha = sha256_file(dgn_root / "metadata.json")

    # ---- Build env ----
    cfg, env = load_env(cfg, None, freeze_env=True)
    obs_space = map_struct(
        env.observation_space,
        lambda src, _: src.shape,
        base_cls=spaces.Box,
        dict_cls=(Mapping, spaces.Dict),
    )
    if cfg.state_net_blocklist is not None:
        for key in cfg.state_net_blocklist:
            obs_space.pop(key, None)
    dim_act = (env.action_space.shape if isinstance(env.action_space, spaces.Box)
               else env.action_space.n)
    cfg = replace(cfg, net=replace(cfg.net, obs_space=obs_space, act_space=dim_act))

    # ---- Load frozen old teacher ----
    print(f"Loading teacher from cfg.load_ckpt = {cfg.load_ckpt}")
    teacher = load_agent(cfg, env, None, None)
    teacher.eval()
    for p in teacher.state_net.parameters():
        p.requires_grad_(False)
    for p in teacher.actor_net.parameters():
        p.requires_grad_(False)

    device = teacher.device
    num_env = env.num_env

    observation_fields = getattr(env.observation_space, "spaces", {})
    if "touch" not in observation_fields:
        raise RuntimeError(
            "F_T buffer collection requires a real touch observation; "
            "launch with ++env.collect_contact=last ++add_touch_flag=true")

    # ---- Allocate buffers (CPU, accumulated) ----
    obs_t = env.reset()
    prev_action, h_t, hidden, _ = teacher.init(obs_t)

    dim_state = h_t.shape[-1]
    dim_act_int = (dim_act[0] if isinstance(dim_act, tuple) else int(dim_act))
    print(f"\nCollection: num_env={num_env}, num_steps={num_steps}, "
          f"total_transitions={num_env * num_steps:,}")
    print(f"  dim_state={dim_state}, dim_act={dim_act_int}, "
          f"noisy_frac={noisy_frac:.2f}, fp16={use_fp16}")

    buf = {
        "h":              [],
        "a":              [],
        "rel_goal_delta": [],
        "object_delta":   [],
        "contact":        [],
        "progress":       [],
        "done":           [],
        "valid":          [],
        "episode_id":     [],
    }
    ep_counter = th.zeros(num_env, dtype=th.long, device=device)
    env_id = th.arange(num_env, dtype=th.long, device=device)
    prev_done = th.zeros(num_env, dtype=th.bool, device=device)

    # ---- Main rollout loop ----
    log_period = max(1, num_steps // 20)
    rng = th.Generator(device=device).manual_seed(int(cfg.env.seed) + 7777)
    for step in range(num_steps):
        # Match PPO's recurrent-state convention: if an env was done after the
        # previous step, zero its hidden state before taking the next action.
        # That next action will be ignored by EnvBase while the lane resets, so
        # the transition is marked invalid for F_T windows.
        with th.no_grad():
            keep = (~prev_done)[..., None]
            map_tensor(hidden, lambda src, _: src.mul_(keep))
            h_t = h_t * keep
            valid_t = ~prev_done

        with th.no_grad():
            mu_t, _ls_t = teacher.actor_net(h_t)
            a_exec = mu_t.detach().clone()

            if noisy_frac > 0:
                noise_mask = (th.rand(num_env, generator=rng, device=device)
                              < noisy_frac).float().unsqueeze(-1)
                noise = th.randn(a_exec.shape, generator=rng, device=device) * noise_std
                a_exec = a_exec + noise_mask * noise
                a_exec = a_exec.clamp(-1.0, 1.0)

        # Step env
        with th.no_grad():
            obs_tp1, _rew, done_t, _info = env.step(a_exec)

        # Structured targets (transition t -> t+1)
        struct_t = _compute_structured(obs_t, obs_tp1, num_env, device)

        # Append (CPU, optional fp16 for floats)
        cast = (lambda x: x.detach().to('cpu', dtype=th.float16)) if use_fp16 \
               else (lambda x: x.detach().to('cpu', dtype=th.float32))
        buf["h"             ].append(cast(h_t))
        buf["a"             ].append(cast(a_exec))
        buf["rel_goal_delta"].append(cast(struct_t["rel_goal_delta"]))
        buf["object_delta"  ].append(cast(struct_t["object_delta"]))
        buf["contact"       ].append(cast(struct_t["contact"]))
        buf["progress"      ].append(cast(struct_t["progress"]))
        buf["done"          ].append(done_t.detach().to('cpu', dtype=th.bool))
        buf["valid"         ].append(valid_t.detach().to('cpu', dtype=th.bool))
        unique_episode_id = ep_counter * num_env + env_id
        buf["episode_id"    ].append(unique_episode_id.detach().clone().to('cpu'))

        # Update teacher state for next step
        with th.no_grad():
            h_t, hidden = teacher.state_net(hidden, a_exec, obs_tp1)

        # Episode bookkeeping (next ep starts after a done)
        ep_counter = ep_counter + done_t.long()
        prev_done = done_t
        obs_t = obs_tp1

        if (step + 1) % log_period == 0 or step == 0:
            print(f"  [{step+1:>6}/{num_steps}] "
                  f"max_episode_id={int(ep_counter.max().item())}")

    # ---- Stack & save ----
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    save_path = os.path.join(save_dir, "buffer.pt")

    out = {}
    for k, lst in buf.items():
        # Keep time/env axes intact. Flattening would make idx+1 jump to the
        # next env lane, which silently corrupts multi-step future targets.
        out[k] = th.stack(lst, dim=0)

    out["meta"] = {
        "artifact_class": artifact_class,
        "layout": "time_env",
        "dim_state": int(dim_state),
        "dim_act":   int(dim_act_int),
        "num_env":   int(num_env),
        "num_steps": int(num_steps),
        "num_object_types": num_object_types,
        "total_transitions": int(num_env * num_steps),
        "noisy_frac": float(noisy_frac),
        "noise_std":  float(noise_std),
        "fp16":       bool(use_fp16),
        "load_ckpt":  str(cfg.load_ckpt),
        "env_seed":   int(cfg.env.seed),
        "contact_source": "simulator_add_approx_touch_flag",
        "contact_positive_count": int((out["contact"] > 0).sum().item()),
        "repo_commit": repo_commit,
        "source_snapshot_sha256": source_snapshot_sha,
        "teacher_checkpoint_sha256": teacher_checkpoint_sha,
        "teacher_env_stat": str(teacher_env_stat),
        "teacher_env_stat_sha256": teacher_env_stat_sha,
        "icp_checkpoint": str(icp_checkpoint),
        "icp_checkpoint_sha256": icp_checkpoint_sha,
        "split_file": str(split_file),
        "split_sha256": split_sha,
        "dgn_root": str(dgn_root),
        "dgn_pose_root": str(dgn_pose_root),
        "dataset_validation": "metadata_hash_only_not_full_asset_lock",
        "dgn_metadata_sha256": dgn_metadata_sha,
    }
    if out["meta"]["contact_positive_count"] <= 0:
        raise RuntimeError(
            "Collected buffer has no positive contact labels; refusing to save "
            "a degenerate F_T training artifact")
    th.save(out, save_path)

    total = out["h"].shape[0] * out["h"].shape[1]
    valid_total = int(out["valid"].sum().item())
    print(f"\nSaved {total:,} transitions to {save_path}")
    print(f"  valid transitions = {valid_total:,}")
    print(f"  buffer file size = {os.path.getsize(save_path) / 1024**3:.2f} GB")
    print(f"  meta = {out['meta']}")


if __name__ == "__main__":
    main()
