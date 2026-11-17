#!/usr/bin/env python3
"""Step 3 K>=2: delayed multi-step F_T distillation for student training.

Three differences from train_rma_FT.py (K=1):

1. New trainer DAggerTrainerEnv_FT_kN holds a length-K queue of snapshots:
       snap(τ) = (obs_τ, a_τ, h_τ_teacher, memory_{τ-1}, valid_mask)
   memory_{τ-1} is stashed BEFORE get_output() runs so that re-encoding the
   popped obs uses the historical RNN state, not the current one.

2. At each step (post env.step):
     - If queue is full (length K), pop the oldest snap and run a K-step
       FT loss on it using actions a_{τ..τ+K-1} (queue + current action)
       and targets h_{τ+1..τ+K} (or F_T rollout). The loss is injected into
       self.aux_losses['student_FT'] and will be backwarded by the
       subsequent update_policy call.
     - Then AND-update existing snaps' valid bit with ~done_now (this is
       the K-th done for the oldest remaining snap).
     - Push snap(now) with valid_init = ~done_now.

3. New student class StudentAgentRMA_FT_kN disables the K=1 in-place FT
   loss path and exposes reencode_obs_with_memory() + compute_FT_loss_kN().

Frozen-teacher invariant: unchanged. Policy/action path: unchanged.
Structured targets: real-env deltas in BOTH target_source modes (matches
K=1 convention).
"""
from __future__ import annotations

import isaacgym
import os
import yaml
from collections import deque
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Deque, Dict, List, Optional

import torch as th
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from icecream import ic

from env.env.wrap.base import WrapperEnv
from env.util import set_seed
from train.wandb import with_wandb
from util.config import recursive_replace_map
from util.hydra_cli import hydra_cli

from train_rma import (
    AddTensorboardWriter,
    Config as BaseConfig,
    DAggerTrainerEnv as BaseDAggerTrainerEnv,
    PlotlyPointCloudRecorder,
    RMATrainerEnv,
    get_config_path,
    load_agent,
    load_env,
    setup_logging,
    setup_rma_env_v2,
    update_net_cfg,
    upload_ckpt,
)

from distill_FT_distill_kN import (
    StudentAgentRMA_FT_kN,
    _snapshot_memory,
)
from student_FT_distill import (
    StudentFTHeadConfig,
    load_FT_frozen,
    build_FT_rollout_targets,
)
from modules_teacher_FT import FTSidecar


FT_TARGET_SOURCES = frozenset({"teacher_latent", "f_t_rollout"})


# =====================================================================
# Snapshot construction helpers
# =====================================================================

def _shallow_detach_obs(obs: Dict[str, th.Tensor]) -> Dict[str, th.Tensor]:
    """detach+clone every tensor value in obs; keep dict structure."""
    return {k: (v.detach().clone() if isinstance(v, th.Tensor) else v)
            for k, v in obs.items()}


def _structured_delta(obs_a: Dict[str, th.Tensor],
                      obs_b: Dict[str, th.Tensor]) -> Dict[str, th.Tensor]:
    """Real-env structured target for transition obs_a -> obs_b. Matches
    the K=1 builder in train_rma_FT.py's _build_supervision_k1.
    """
    rel_a   = obs_a.get("rel_goal",     None)
    rel_b   = obs_b.get("rel_goal",     None)
    obj_a   = obs_a.get("object_state", None)
    obj_b   = obs_b.get("object_state", None)
    touch_b = obs_b.get("touch",        None)
    if rel_a is None or rel_b is None:
        raise KeyError("_FT structured target needs rel_goal in obs")
    if obj_a is None or obj_b is None:
        raise KeyError("_FT structured target needs object_state in obs")
    rel_a = rel_a[..., :9].detach()
    rel_b = rel_b[..., :9].detach()
    obj_a = obj_a[..., :13].detach()
    obj_b = obj_b[..., :13].detach()
    B = rel_a.shape[0]
    if touch_b is None:
        raise KeyError(
            "K-step FT supervision requires a real touch observation; "
            "launch with ++env.collect_contact=last ++add_touch_flag=true")
    contact = touch_b.reshape(B, -1)[..., :1].detach().clamp(0.0, 1.0)
    dist_a = th.linalg.norm(rel_a[..., :3], dim=-1, keepdim=True)
    dist_b = th.linalg.norm(rel_b[..., :3], dim=-1, keepdim=True)
    return {
        "rel_goal_delta": (rel_b - rel_a).detach(),
        "object_delta":   (obj_b - obj_a).detach(),
        "contact":        contact,
        "progress":       (dist_a - dist_b).detach(),
    }


# =====================================================================
# DAgger trainer with K-step delayed FT supervision
# =====================================================================

class DAggerTrainerEnv_FT_kN(BaseDAggerTrainerEnv):
    """K-step delayed FT supervision via per-step snapshot queue.

    Time bookkeeping in step() at iteration i:
      pre-state:
        self.last_obs = obs_i
        agg.aggregator.memory = memory_{i-1}
        queue contains snap(i-K)..snap(i-1) once warmed up (size = K)
        snap(j).valid has accumulated ~done_j..~done_{i-1}

      flow:
        1. stash memory_pre = memory_{i-1}
        2. get_output(obs_i)  [memory advances to memory_i]
        3. choose a_i; env.step(a_i) -> obs_{i+1}, done_i
        4. if queue is full (size == K):
              peek front = snap(i-K) with K dones accumulated (done_{i-K..i-1})
              POP front; compute K-step FT loss using:
                action_seq    = [popped.action, *queue_after_pop.action]   (K total)
                              = [a_{i-K}, a_{i-K+1}, ..., a_{i-1}]
                              NOTE: a_i (just chosen this iteration) is NOT
                              included — it belongs to snap(i)'s window.
                h_target_traj = (f_t_rollout)  F_T.rollout(popped.h, action_seq)
                              | (teacher_lat)  [*queue_after_pop.h, last_obs.teacher_state]
                structured    = K real-env deltas across
                                [popped.obs, *queue_after_pop.obs, last_obs]
                stashed_mem   = popped.memory_pre
                valid_mask    = popped.valid
        5. AND queue with ~done_i
        6. push snap(i) with memory_pre, valid_init = ~done_i
        7. update_policy(last_obs) — backwards action loss + injected FT loss
        8. last_obs = obs_{i+1}; reset RNN for done_i envs
    """

    @dataclass
    class Config(BaseDAggerTrainerEnv.Config):
        FT_target_source: str = "f_t_rollout"   # "f_t_rollout" or "teacher_latent"
        K_window: int = 3

    def __init__(self, cfg, env, student, *, start_step=0,
                 frozen_FT: Optional[FTSidecar] = None, **kwds):
        super().__init__(cfg, env, student, start_step=start_step, **kwds)
        self.frozen_FT = frozen_FT
        self._FT_target_source = self.cfg.FT_target_source
        self._K = int(self.cfg.K_window)
        if self._FT_target_source not in FT_TARGET_SOURCES:
            choices = ", ".join(sorted(FT_TARGET_SOURCES))
            raise ValueError(
                f"Unknown FT_target_source={self._FT_target_source!r}; "
                f"expected one of: {choices}"
            )
        if self._K < 2:
            raise ValueError(
                f"K_window must be >= 2; got {self._K}. Use train_rma_FT.py for K=1."
            )
        if self._FT_target_source == "f_t_rollout" and frozen_FT is None:
            raise ValueError(
                "FT_target_source='f_t_rollout' but frozen_FT is None"
            )
        self.queue: Deque[dict] = deque()

    # ---- target builders ----
    def _build_h_targets(self, popped: dict, queue_after_pop: List[dict],
                         last_obs: Dict[str, th.Tensor],
                         action_seq: th.Tensor) -> th.Tensor:
        """Return (B, K, dim_state) latent target trajectory."""
        if self._FT_target_source == "f_t_rollout":
            h_target_traj, _structured_FT = build_FT_rollout_targets(
                self.frozen_FT, popped["h_teacher"], action_seq)
            return h_target_traj
        # teacher_latent: real h_{τ+1..τ+K} from queue + current last_obs
        if "teacher_state" not in last_obs:
            raise KeyError(
                "_FT teacher_latent target needs obs['teacher_state'] "
                "(set add_teacher_state=true)"
            )
        h_list = [s["h_teacher"] for s in queue_after_pop]
        h_list.append(last_obs["teacher_state"].detach())
        return th.stack(h_list, dim=1)

    def _build_structured_targets(self, popped: dict,
                                  queue_after_pop: List[dict],
                                  last_obs: Dict[str, th.Tensor],
                                  ) -> List[Dict[str, th.Tensor]]:
        """K structured deltas for transitions across popped + queue + last_obs.

        Always from real-env (matches K=1 convention; F_T's decoder structured
        is discarded even in f_t_rollout mode).
        """
        # obs sequence covers K+1 timesteps: [popped, *queue, last_obs]
        obs_seq = [popped["obs"]] + [s["obs"] for s in queue_after_pop] + [last_obs]
        if len(obs_seq) != self._K + 1:
            raise RuntimeError(
                f"obs_seq length {len(obs_seq)} != K+1 = {self._K + 1}; "
                f"queue bookkeeping broken"
            )
        return [_structured_delta(obs_seq[k], obs_seq[k + 1])
                for k in range(self._K)]

    # ---- main step ----
    def step(self, actions: Optional[th.Tensor] = None):
        cfg = self.cfg
        K = self._K

        # ----- (1) stash memory_pre BEFORE encoder runs on obs_i -----
        agg = self.student_module.aggregator
        memory_pre_i = (_snapshot_memory(agg.aggregator.memory)
                        if (agg.cfg.aggregator_type in ["gru", "deep_gru", "lstm"]
                            and hasattr(agg.aggregator, "memory"))
                        else None)

        # ----- (2) student action; memory advances to memory_i -----
        self._DAggerTrainerEnv__next_action = self.last_obs.get("teacher_action")
        self._DAggerTrainerEnv__next_student_action, aux = self.student_module.get_output(
            self.last_obs)

        if actions is None:
            actions = self.get_action(
                self._DAggerTrainerEnv__next_student_action,
                deterministic=cfg.deterministic_action,
            )

        record_policy_aux = getattr(self.env, "record_policy_aux", None)
        if record_policy_aux is not None:
            record_policy_aux(aux)

        # ----- (3) env.step (a_i) -> obs_{i+1}, done_i -----
        obs, rew, done, info = WrapperEnv.step(self, actions)
        done_i = done

        # ----- (4) if queue full, pop oldest and compute K-step FT loss -----
        if len(self.queue) == K:
            popped = self.queue.popleft()
            queue_after_pop = list(self.queue)  # K-1 snaps remaining

            # Action sequence covering K transitions a_{τ..τ+K-1}:
            #   τ          : popped.action          (= a_{i-K})
            #   τ+1..τ+K-2 : queue_after_pop.action (= a_{i-K+1..i-1})
            #   τ+K-1      : actions  (just chosen in this iteration, = a_i would
            #                          mean wrong indexing — we need a_{i-1}!)
            #
            # WAIT — at this iteration the popped snap represents time i-K. Its
            # K-step window covers transitions i-K -> i-K+1 -> ... -> i-1 -> i.
            # Actions are a_{i-K..i-1}. NOT a_i (that belongs to snap(i)).
            #
            # So action_seq = [popped.action, *queue_after_pop.action]; that's
            # 1 + (K-1) = K actions = a_{i-K..i-1}. `actions` (= a_i) is NOT
            # included here.
            actions_list = [popped["action"]] + [s["action"] for s in queue_after_pop]
            action_seq = th.stack(actions_list, dim=1)  # (B, K, dim_act)

            # Latent + structured targets
            h_target_traj = self._build_h_targets(
                popped, queue_after_pop, self.last_obs, action_seq)
            structured_targets = self._build_structured_targets(
                popped, queue_after_pop, self.last_obs)

            # Compute K-step FT loss; injects into self.aux_losses['student_FT']
            self.student_module.compute_FT_loss_kN(
                obs_t=popped["obs"],
                action_seq=action_seq,
                h_target_traj=h_target_traj,
                structured_targets=structured_targets,
                stashed_memory=popped["memory_pre"],
                valid_mask=popped["valid"],
                step=self._DAggerTrainerEnv__step,
            )

        # ----- (5) AND-update existing snaps with ~done_i -----
        for snap in self.queue:
            snap["valid"] = snap["valid"] & (~done_i)

        # ----- (6) push snap(i) -----
        snap_i = {
            "obs":         _shallow_detach_obs(self.last_obs),
            "action":      actions.detach().clone(),
            "h_teacher":   self.last_obs["teacher_state"].detach().clone(),
            "memory_pre":  memory_pre_i,
            "valid":       (~done_i).detach().clone(),
        }
        self.queue.append(snap_i)
        if len(self.queue) > K:
            # Shouldn't happen given step-4 pop, but guard anyway.
            raise RuntimeError(f"FT_kN queue overflow: {len(self.queue)} > {K}")

        # ----- (7) action loss (+ injected FT loss) -----
        self.student_module.update_policy(
            self.last_obs,
            self._DAggerTrainerEnv__step,
            self._DAggerTrainerEnv__next_student_action,
            aux,
        )
        self.last_obs = obs

        # ----- (8) RNN reset for done envs -----
        from models.common import map_tensor
        with th.no_grad():
            keep = (~done_i)[..., None, None]
            map_tensor(
                self._DAggerTrainerEnv__next_student_action,
                lambda src, _: src.mul_(keep),
            )
            self.student_module.reset_state(done_i)
            # Stashed snaps in queue keep their memory_pre intact (they refer
            # to the encoding-time memory, not the current one). Valid masks
            # already exclude any env that hit done within the window.

        self._DAggerTrainerEnv__step += 1
        return obs, rew, done, info


# =====================================================================
# Config
# =====================================================================

@dataclass
class Config(BaseConfig):
    FT_ckpt_path: Optional[str] = None
    FT_target_source: str = "f_t_rollout"
    K_window: int = 3
    student_FT_head: StudentFTHeadConfig = field(default_factory=StudentFTHeadConfig)
    allow_structured_phase_film_warmstart: bool = False


def _cast_student_cfg_to_FT(base_student_cfg, ft_head_cfg):
    """Build StudentAgentRMA_FT_kN.StudentAgentRMAConfig from base student cfg."""
    Subcls = StudentAgentRMA_FT_kN.StudentAgentRMAConfig
    base_kw = {f.name: getattr(base_student_cfg, f.name)
               for f in fields(base_student_cfg)
               if hasattr(base_student_cfg, f.name)}
    base_kw["student_FT_head"] = ft_head_cfg
    return Subcls(**base_kw)


def _load_structured_phase_film_warmstart(student, checkpoint: str) -> dict:
    """Strictly load Contact/SEER, allowing only the new FiLM parameters.

    This deliberately avoids ``strict=False``: a changed Contact encoder,
    action decoder, or F_S key must abort the experiment rather than silently
    produce a partially initialized model.
    """
    path = Path(checkpoint)
    payload = th.load(str(path), map_location="cpu")
    if "self" not in payload or not isinstance(payload["self"], dict):
        raise ValueError(f"student checkpoint has no 'self' state: {path}")
    module = student.module if isinstance(student, nn.DataParallel) else student
    saved = payload["self"]
    expected = module.state_dict()
    runtime = {"history_tokenizer.history", "aggregator.aggregator.memory"}
    missing = set(expected) - set(saved)
    unexpected = set(saved) - set(expected)
    phase_film_keys = {
        key for key in expected if key.startswith("structured_phase_film.")}
    missing_model_keys = missing - runtime
    bad_missing = missing_model_keys - phase_film_keys
    if bad_missing or unexpected:
        raise ValueError(
            "structured Phase-FiLM warm-start mismatch: "
            f"bad_missing={sorted(bad_missing)}, "
            f"unexpected={sorted(unexpected)}")
    # Two auditable warm-start modes are supported:
    #   1) Contact-only checkpoint: exactly the new Phase-FiLM branch is absent.
    #   2) Phase-FiLM checkpoint: no model key is absent (short SFT/resume).
    # Any partial branch is rejected because it would silently mix two models.
    if missing_model_keys not in (set(), phase_film_keys):
        raise ValueError(
            "structured Phase-FiLM warm-start has a partial branch: "
            f"missing={sorted(missing_model_keys)}, "
            f"phase_keys={sorted(phase_film_keys)}")
    warmstart_mode = (
        "contact_to_phase" if missing_model_keys == phase_film_keys
        else "phase_resume")
    # Runtime buffers are batch-size dependent and must retain their freshly
    # constructed shapes (the production run may use a different num_env).
    loadable = {key: value for key, value in saved.items() if key not in runtime}
    result = module.load_state_dict(loadable, strict=False)
    return {
        "checkpoint": str(path),
        "mode": warmstart_mode,
        "allowed_missing": sorted(phase_film_keys),
        "loader_missing": sorted(result.missing_keys),
        "loader_unexpected": sorted(result.unexpected_keys),
    }


# =====================================================================
# Main
# =====================================================================

@hydra_cli(Config, config_path=get_config_path(), config_name="train_rl")
@with_wandb
def main(cfg: Config):
    th.backends.cudnn.benchmark = True

    ic.configureOutput(includeContext=True)
    cfg.project = 'rma'
    cfg = recursive_replace_map(cfg, {'finalize': True})

    if cfg.FT_target_source not in FT_TARGET_SOURCES:
        choices = ", ".join(sorted(FT_TARGET_SOURCES))
        raise ValueError(
            f"Unknown FT_target_source={cfg.FT_target_source!r}; "
            f"expected one of: {choices}"
        )

    # Pin student_FT_head.K to the trainer's K_window and force enabled.
    cfg = replace(cfg, student_FT_head=replace(
        cfg.student_FT_head,
        enabled=True,
        K=cfg.K_window,
        target_source=cfg.FT_target_source,
    ))

    if cfg.global_device is not None:
        th.cuda.set_device(cfg.global_device)
    path = setup_logging(cfg)
    writer = SummaryWriter(path.tb_train)
    _ = set_seed(cfg.env.seed)
    if (cfg.use_nvdr_record_episode or cfg.use_nvdr_record_viewer):
        cfg.env.track_debug_lines = True
    cfg, env = load_env(cfg, path, freeze_env=True, check_viewer=False)
    env.unwrap(target=AddTensorboardWriter).set_writer(writer)

    # ---- frozen old teacher ----
    cfg = replace(cfg, net=update_net_cfg(cfg.net, env, cfg.state_net_blocklist))
    teacher_agent = load_agent(cfg, env, None, None)
    teacher_agent.eval()
    for p in teacher_agent.state_net.parameters():
        p.requires_grad_(False)
    for p in teacher_agent.actor_net.parameters():
        p.requires_grad_(False)
    ic(teacher_agent)

    teacher_dim_state = int(
        teacher_agent.state_net.state_aggregator.cfg.dim_out
    )
    teacher_dim_act = (env.action_space.shape[0] if hasattr(env.action_space, 'shape')
                       else env.action_space.n)
    print(f"[FT_kN] K={cfg.K_window}, target_source={cfg.FT_target_source}, "
          f"teacher_dim_state={teacher_dim_state}, teacher_dim_act={teacher_dim_act}")

    # ---- F_T (required for f_t_rollout) ----
    frozen_FT = None
    if cfg.FT_ckpt_path is not None and os.path.exists(cfg.FT_ckpt_path):
        frozen_FT = load_FT_frozen(cfg.FT_ckpt_path, env.device)
    elif cfg.FT_target_source == "f_t_rollout":
        raise ValueError(
            f"FT_target_source='f_t_rollout' but FT_ckpt_path "
            f"({cfg.FT_ckpt_path}) does not exist."
        )

    # ---- student (FT_kN subclass) ----
    base_student_cfg = replace(cfg.student, batch_size=env.num_env)
    student_FT_cfg = _cast_student_cfg_to_FT(base_student_cfg, cfg.student_FT_head)
    student = StudentAgentRMA_FT_kN(
        student_FT_cfg,
        writer=writer,
        device=env.device,
        teacher_dim_state=teacher_dim_state,
        teacher_dim_act=teacher_dim_act,
        frozen_teacher=teacher_agent,
        frozen_FT=frozen_FT,
    )
    student = nn.DataParallel(student, cfg.dp_gpus) if cfg.dp_gpus is not None else student
    student = student.to(env.device)

    if cfg.load_student is not None:
        if cfg.student_FT_head.structured_phase_film_enabled:
            if not cfg.allow_structured_phase_film_warmstart:
                raise ValueError(
                    "structured Phase-FiLM warm-start requires "
                    "allow_structured_phase_film_warmstart=true")
            warmstart = _load_structured_phase_film_warmstart(
                student, cfg.load_student)
            print(f"[StructuredPhaseFiLM] strict warm-start: {warmstart}")
        else:
            student.load(cfg.load_student, strict=False)

    ic(student)

    # ---- env wrap ----
    env = setup_rma_env_v2(cfg, env, teacher_agent,
                           state_size=128,
                           is_student=False,
                           dagger=cfg.dagger)

    if cfg.load_environment is not None:
        env_ckpt = Path(cfg.load_environment)
        env.load(env_ckpt, strict=True)

    if cfg.plot_pc:
        env = PlotlyPointCloudRecorder(cfg.plotly_pc_recorder, env)

    prefix = f'{cfg.name}@{cfg.global_device}'
    student.train()

    if cfg.dagger:
        base_kw = asdict(cfg.dagger_train_env)
        train_cfg = DAggerTrainerEnv_FT_kN.Config(
            **{**base_kw,
               'FT_target_source': cfg.FT_target_source,
               'K_window': cfg.K_window})
        train_env = DAggerTrainerEnv_FT_kN(
            train_cfg, env, student,
            start_step=cfg.start_step,
            frozen_FT=frozen_FT,
        )
    else:
        train_env = RMATrainerEnv(cfg.rma_train_env, env, student)
    train_env.reset()

    # ==== save config ====
    cfg_dict = asdict(cfg)
    with open(str(path.stat.parent / "cfg.yaml"), 'w') as fcfg:
        yaml.dump(cfg_dict, fcfg, default_flow_style=False)

    student_module = student.module if isinstance(student, nn.DataParallel) else student
    start_step = cfg.start_step
    if start_step > 0:
        print(f"[Resume] Starting from step {start_step}, training to {cfg.train_step}")
    try:
        for step in tqdm(range(start_step, cfg.train_step),
                         desc=prefix, mininterval=5,
                         initial=start_step, total=cfg.train_step):
            train_env.step(None)
            if step % cfg.save_step == 0 and step > start_step:
                student_module.save(path.ckpt / f'step-{step:06d}.ckpt')
                if hasattr(env, 'save'):
                    env.save(path.stat / f'env-{step:06d}.ckpt')
        # Loop completed normally: expose the terminal step to eval watchers.
        student_module.save(path.ckpt / f'step-{cfg.train_step:06d}.ckpt')
        if hasattr(env, 'save'):
            env.save(path.stat / f'env-{cfg.train_step:06d}.ckpt')
    finally:
        student_module.save(path.ckpt / 'last.ckpt')
        if hasattr(env, 'save'):
            env.save(path.stat / 'env-last.ckpt')
        if cfg.use_hfhub and (cfg.hf_repo_id is not None):
            upload_ckpt(cfg.hf_repo_id, (path.ckpt / 'last.ckpt'), cfg.name)
            upload_ckpt(cfg.hf_repo_id, (path.stat / 'env-last.ckpt'), cfg.name + '_stat')


if __name__ == '__main__':
    main()
