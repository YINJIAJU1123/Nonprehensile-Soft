#!/usr/bin/env python3
"""Step 2B: Train F_T offline on the teacher buffer.

Loads the buffer dumped by collect_teacher_buffer.py, builds K-step windows
that don't cross episode boundaries, and trains FTSidecar via
multistep_rollout_loss. Does not require IsaacGym.

Inputs/outputs are governed by env vars (kept simple — no hydra here):
    BUFFER_PATH      : input buffer.pt              (default: teacher_buffer_v1)
    FT_OUT_DIR       : output dir for ckpts + logs  (default: ft_ckpt_v1)
    K                : multi-step K                 (default: 3)
    LATENT_MODE      : cosine_mse / norm_mse / cosine / mse  (default: cosine_mse)
    LATENT_WEIGHT    : float                        (default: 1.0)
    STRUCTURED_WEIGHT: float                        (default: 1.0)
    DISCOUNT         : gamma                        (default: 1.0)
    BATCH            : batch size                   (default: 1024)
    EPOCHS           : training epochs              (default: 50)
    LR               : adam lr                      (default: 3e-4)
    WD               : weight decay                 (default: 1e-5)
    VAL_FRAC         : holdout fraction             (default: 0.05)
    SEED             : random seed                  (default: 56081)
"""
from __future__ import annotations

import os
import json
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

# Local imports
import sys
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
SRC_ROOT = REPO_ROOT / "dywa" / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from modules_teacher_FT import (
    FTSidecar, FTSidecarConfig,
    multistep_rollout_loss,
)
from train.source_lineage import (
    TRAIN_SOURCE_FILES,
    canonical_json_sha256,
    sha256_file,
)


# =====================================================================
# Buffer + windowing
# =====================================================================

def _build_window_indices(buf: dict, K: int) -> th.Tensor:
    """Return (M, 2) tensor of (t, e) pairs that start a valid K-step window.

    Buffer is time-major: shape (T, E, ...). A start (t, e) is valid iff:
      - valid[t..t+K, e]  all True   (no transition crosses a reset boundary)
      - done [t..t+K-1, e] all False (no episode terminates inside window)
      - episode_id[t, e] == episode_id[t+K, e]
      - t + K < T  (h_{t+K} is reachable as target)
    """
    valid = buf["valid"]       # (T, E) bool
    done  = buf["done"]        # (T, E) bool
    ep    = buf["episode_id"]  # (T, E) long
    T, E = valid.shape
    if T <= K:
        return th.empty(0, 2, dtype=th.long)

    # valid over [t, t+K] inclusive (K+1 timesteps)
    valid_window = valid[: T - K].clone()
    for k in range(1, K + 1):
        valid_window = valid_window & valid[k : T - K + k]

    # No 'done' inside [t, t+K-1]
    no_done_inside = th.ones(T - K, E, dtype=th.bool)
    for k in range(K):
        no_done_inside = no_done_inside & (~done[k : T - K + k])

    # Same episode at t and t+K (defense in depth — implied by the above)
    same_ep = (ep[: T - K] == ep[K : T])

    mask = valid_window & no_done_inside & same_ep  # (T-K, E)
    t_idx, e_idx = th.nonzero(mask, as_tuple=True)
    return th.stack([t_idx, e_idx], dim=1)  # (M, 2)


def _gather_window(buf: dict, idx_te: th.Tensor, K: int) -> Dict[str, th.Tensor]:
    """For each (t, e) start index, gather the K-step window."""
    h_arr = buf["h"]              # (T, E, dim_state)
    a_arr = buf["a"]              # (T, E, dim_act)
    rel   = buf["rel_goal_delta"] # (T, E, 9)
    obj   = buf["object_delta"]   # (T, E, 13)
    con   = buf["contact"]        # (T, E, 1)
    pro   = buf["progress"]       # (T, E, 1)
    t = idx_te[:, 0]
    e = idx_te[:, 1]

    h0       = h_arr[t, e].float()                                          # (B, ds)
    actions  = th.stack([a_arr[t + k, e].float() for k in range(K)], dim=1) # (B, K, da)
    h_target = th.stack([h_arr[t + k + 1, e].float() for k in range(K)], dim=1)  # (B, K, ds)

    structured_targets: List[Dict[str, th.Tensor]] = []
    for k in range(K):
        structured_targets.append({
            "rel_goal_delta": rel[t + k, e].float(),
            "object_delta":   obj[t + k, e].float(),
            "contact":        con[t + k, e].float(),
            "progress":       pro[t + k, e].float(),
        })

    return {
        "h0":                 h0,
        "actions":            actions,
        "h_target":           h_target,
        "structured_targets": structured_targets,
    }


# =====================================================================
# Training
# =====================================================================

def main():
    BUFFER_PATH       = os.environ.get(
        "BUFFER_PATH",
        str(REPO_ROOT / "output" / "teacher_buffer_v1" / "buffer.pt"),
    )
    FT_OUT_DIR        = os.environ.get(
        "FT_OUT_DIR",
        str(REPO_ROOT / "output" / "ft_ckpt_v1"),
    )
    K                 = int  (os.environ.get("K",                "3"))
    LATENT_MODE       =       os.environ.get("LATENT_MODE",      "cosine_mse")
    LATENT_WEIGHT     = float(os.environ.get("LATENT_WEIGHT",    "1.0"))
    STRUCTURED_WEIGHT = float(os.environ.get("STRUCTURED_WEIGHT","1.0"))
    DISCOUNT          = float(os.environ.get("DISCOUNT",         "1.0"))
    BATCH             = int  (os.environ.get("BATCH",            "1024"))
    EPOCHS            = int  (os.environ.get("EPOCHS",           "50"))
    LR                = float(os.environ.get("LR",               "3e-4"))
    WD                = float(os.environ.get("WD",               "1e-5"))
    VAL_FRAC          = float(os.environ.get("VAL_FRAC",         "0.05"))
    SEED              = int  (os.environ.get("SEED",             "56081"))

    th.manual_seed(SEED)
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    Path(FT_OUT_DIR).mkdir(parents=True, exist_ok=True)
    buffer_path = Path(BUFFER_PATH).expanduser().resolve()
    buffer_sha256 = sha256_file(buffer_path)
    source_files = {
        relative_path: sha256_file(REPO_ROOT / relative_path)
        for relative_path in TRAIN_SOURCE_FILES
    }
    source_snapshot_sha256 = canonical_json_sha256(source_files)
    repo_commit = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True,
    ).strip()
    print(f"Loading buffer: {buffer_path}")
    buf = th.load(str(buffer_path), map_location="cpu")

    meta = buf.get("meta", {})
    required_lineage = {
        "artifact_class", "repo_commit", "source_snapshot_sha256",
        "teacher_checkpoint_sha256", "teacher_env_stat",
        "teacher_env_stat_sha256", "icp_checkpoint",
        "icp_checkpoint_sha256", "split_file", "split_sha256",
    }
    missing_lineage = required_lineage - set(meta)
    if missing_lineage:
        raise RuntimeError(
            "F_T buffer lacks collection lineage: "
            f"{sorted(missing_lineage)}")
    artifact_class = meta["artifact_class"]
    if artifact_class not in {
            "NONPREHENSILE_SOFT_DEV"}:
        raise RuntimeError(f"unsupported F_T artifact class: {artifact_class}")
    requested_class = os.environ.get("ARTIFACT_CLASS")
    if requested_class is not None and requested_class != artifact_class:
        raise RuntimeError(
            "F_T trainer cannot change the buffer artifact class")
    if meta["repo_commit"] != repo_commit:
        raise RuntimeError("F_T buffer and offline trainer use different commits")
    if meta["source_snapshot_sha256"] != source_snapshot_sha256:
        raise RuntimeError(
            "F_T buffer and offline trainer use different source snapshots")
    print(f"  meta: {meta}")
    dim_state = int(meta["dim_state"])
    dim_act   = int(meta["dim_act"])
    print(f"  dim_state={dim_state}, dim_act={dim_act}")

    done = buf["done"]              # (T, E) bool
    T, E = int(done.shape[0]), int(done.shape[1])
    N_total = T * E
    print(f"  buffer shape: T={T} E={E}, total transitions: {N_total:,}")

    # ---- Build valid K-step (t, e) window indices ----
    valid_idx = _build_window_indices(buf, K)            # (M, 2)
    n_valid = int(valid_idx.shape[0])
    print(f"  valid K={K} window starts: {n_valid:,}  "
          f"({100.0*n_valid/max(N_total,1):.1f}% of N)")
    if n_valid < BATCH * 10:
        raise RuntimeError(f"Too few valid windows ({n_valid}); collect more buffer or shrink K")

    # ---- Train / val split (deterministic) ----
    perm = th.randperm(n_valid, generator=th.Generator().manual_seed(SEED))
    n_val = int(round(n_valid * VAL_FRAC))
    val_idx   = valid_idx[perm[:n_val]]                  # (n_val, 2)
    train_idx = valid_idx[perm[n_val:]]                  # (n_train, 2)
    print(f"  train windows: {train_idx.shape[0]:,}, val windows: {val_idx.shape[0]:,}")

    # ---- Build F_T ----
    cfg = FTSidecarConfig(dim_state=dim_state, dim_act=dim_act)
    f_t = FTSidecar(cfg).to(device)
    optim = th.optim.AdamW(f_t.parameters(), lr=LR, weight_decay=WD)

    n_params = sum(p.numel() for p in f_t.parameters())
    print(f"  F_T params: {n_params:,}")

    writer = SummaryWriter(log_dir=os.path.join(FT_OUT_DIR, "tb"))

    def _eval_loss():
        f_t.eval()
        with th.no_grad():
            n = val_idx.shape[0]
            n_chunks = max(1, (n + BATCH - 1) // BATCH)
            tot = 0.0
            tot_lat = 0.0
            tot_str = 0.0
            cnt = 0
            for c in range(n_chunks):
                ii = val_idx[c * BATCH : (c + 1) * BATCH]
                w = _gather_window(buf, ii, K)
                w = {k: (v if isinstance(v, list)
                         else v.to(device, non_blocking=True))
                     for k, v in w.items()}
                w["structured_targets"] = [
                    {kk: vv.to(device, non_blocking=True) for kk, vv in s.items()}
                    for s in w["structured_targets"]
                ]
                total, logs = multistep_rollout_loss(
                    f_t,
                    h0_target=w["h0"],
                    actions=w["actions"],
                    h_target_traj=w["h_target"],
                    structured_targets=w["structured_targets"],
                    discount=DISCOUNT,
                    latent_mode=LATENT_MODE,
                    latent_weight=LATENT_WEIGHT,
                    structured_weight=STRUCTURED_WEIGHT,
                )
                tot     += float(total.item()) * ii.numel()
                tot_lat += float(logs["latent_total"].item()) * ii.numel()
                tot_str += float(logs["structured_total"].item()) * ii.numel()
                cnt     += ii.numel()
        f_t.train()
        return tot / cnt, tot_lat / cnt, tot_str / cnt

    # ---- Training loop ----
    best_val = float("inf")
    global_step = 0
    n_train = train_idx.shape[0]
    steps_per_epoch = (n_train + BATCH - 1) // BATCH
    print(f"\nTraining: {EPOCHS} epochs * {steps_per_epoch} steps/epoch")
    t0 = time.time()
    f_t.train()

    for epoch in range(EPOCHS):
        perm_e = th.randperm(n_train, generator=th.Generator().manual_seed(SEED + epoch))
        order = train_idx[perm_e]                         # (n_train, 2)
        ep_loss = 0.0
        for step_i in range(steps_per_epoch):
            ii = order[step_i * BATCH : (step_i + 1) * BATCH]
            w = _gather_window(buf, ii, K)
            w["h0"]       = w["h0"].to(device, non_blocking=True)
            w["actions"]  = w["actions"].to(device, non_blocking=True)
            w["h_target"] = w["h_target"].to(device, non_blocking=True)
            w["structured_targets"] = [
                {kk: vv.to(device, non_blocking=True) for kk, vv in s.items()}
                for s in w["structured_targets"]
            ]

            total, logs = multistep_rollout_loss(
                f_t,
                h0_target=w["h0"],
                actions=w["actions"],
                h_target_traj=w["h_target"],
                structured_targets=w["structured_targets"],
                discount=DISCOUNT,
                latent_mode=LATENT_MODE,
                latent_weight=LATENT_WEIGHT,
                structured_weight=STRUCTURED_WEIGHT,
            )
            optim.zero_grad(set_to_none=True)
            total.backward()
            th.nn.utils.clip_grad_norm_(f_t.parameters(), max_norm=1.0)
            optim.step()

            ep_loss += float(total.item())
            global_step += 1
            if global_step % 100 == 0:
                writer.add_scalar("train/total",          float(total.item()),                  global_step)
                writer.add_scalar("train/latent_total",   float(logs["latent_total"].item()),   global_step)
                writer.add_scalar("train/structured_total", float(logs["structured_total"].item()), global_step)
                for k_log, v_log in logs.items():
                    if isinstance(v_log, th.Tensor) and v_log.numel() == 1:
                        writer.add_scalar(f"train_detail/{k_log}", float(v_log.item()), global_step)

        # End of epoch: val
        val_total, val_lat, val_str = _eval_loss()
        elapsed = time.time() - t0
        print(f"  epoch {epoch+1:>3}/{EPOCHS}  "
              f"train_avg={ep_loss/steps_per_epoch:.5f}  "
              f"val_total={val_total:.5f}  val_lat={val_lat:.5f}  val_str={val_str:.5f}  "
              f"({elapsed:.0f}s)")
        writer.add_scalar("val/total",      val_total, epoch + 1)
        writer.add_scalar("val/latent",     val_lat,   epoch + 1)
        writer.add_scalar("val/structured", val_str,   epoch + 1)

        # Checkpointing
        ckpt = {
            "artifact_class": artifact_class,
            "f_t_state_dict": f_t.state_dict(),
            "cfg": cfg.__dict__,
            "epoch": epoch + 1,
            "val_total": val_total,
            "buffer_meta": meta,
            "buffer_path": str(buffer_path),
            "buffer_sha256": buffer_sha256,
            "repo_commit": repo_commit,
            "source_snapshot_sha256": source_snapshot_sha256,
            "train_meta": {
                "K": K, "LATENT_MODE": LATENT_MODE,
                "LATENT_WEIGHT": LATENT_WEIGHT, "STRUCTURED_WEIGHT": STRUCTURED_WEIGHT,
                "DISCOUNT": DISCOUNT, "BATCH": BATCH, "EPOCHS": EPOCHS,
                "LR": LR, "WD": WD, "VAL_FRAC": VAL_FRAC, "SEED": SEED,
            },
        }
        last_path = os.path.join(FT_OUT_DIR, "ft_last.pt")
        th.save(ckpt, last_path)
        if val_total < best_val:
            best_val = val_total
            best_path = os.path.join(FT_OUT_DIR, "ft_best.pt")
            th.save(ckpt, best_path)
            print(f"    new best val_total={val_total:.5f}  saved to {best_path}")

    writer.close()

    # ---- Summary ----
    summary = {
        "artifact_class": artifact_class,
        "best_val_total": best_val,
        "epochs": EPOCHS,
        "n_train_windows": int(train_idx.shape[0]),
        "n_val_windows":   int(val_idx.shape[0]),
        "K": K,
        "latent_mode": LATENT_MODE,
        "buffer_path": str(buffer_path),
        "buffer_sha256": buffer_sha256,
        "repo_commit": repo_commit,
        "source_snapshot_sha256": source_snapshot_sha256,
        "ft_ckpt_dir": FT_OUT_DIR,
    }
    with open(os.path.join(FT_OUT_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nFinal: best_val_total={best_val:.5f}")
    print(f"       ckpts in {FT_OUT_DIR}")


if __name__ == "__main__":
    main()
