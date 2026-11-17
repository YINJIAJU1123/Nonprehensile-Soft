"""Development fixed-trial evaluation; not the archived paper's formal protocol."""
import isaacgym  # Must precede torch.
from dataclasses import dataclass, replace
import json
from pathlib import Path
from typing import Optional
import torch as th
from distill import StudentAgentRMA
from distill_FT_distill_kN import StudentAgentRMA_FT_kN
from env.util import set_seed
from env.env.wrap.normalize_env import NormalizeEnv
from envs.cube_env_wrappers import CountFixedTrialsSuccess
from policy_runtime import load_policy_state, run_fixed_trials
from rma_env import setup_rma_env_v2
from train.source_lineage import sha256_file
from train_ppo_arm import AddTensorboardWriter, load_env
from train_rma import load_agent, update_net_cfg
from train_rma_FT_kN import Config as BaseConfig, get_config_path, _cast_student_cfg_to_FT
from util.config import recursive_replace_map
from util.hydra_cli import hydra_cli


@dataclass
class Config(BaseConfig):
    policy: str = 'seer'
    force_rad: Optional[float] = .05
    force_ang: Optional[float] = .1
    force_vel: Optional[float] = .1
    fixed_trials_per_object: int = 200
    fixed_trials_result_dir: Optional[str] = None
    max_eval_steps: int = 30000


@hydra_cli(Config, config_path=get_config_path(), config_name='train_rl')
def main(cfg):
    cfg = recursive_replace_map(cfg, {'finalize': True})
    if cfg.policy not in {'dywa', 'seer'}:
        raise ValueError(cfg.policy)
    if (cfg.student_FT_head.feed_to_policy or cfg.student_FT_head.future_film_enabled
            or cfg.student_FT_head.structured_phase_film_enabled):
        raise ValueError('This evaluator supports prediction-free baseline policies only')
    if cfg.load_environment is None or cfg.load_student is None:
        raise ValueError('Evaluation requires paired student and environment checkpoints')
    th.cuda.set_device(cfg.global_device)
    set_seed(cfg.env.seed)
    cfg = replace(cfg, add_teacher_state=False, dagger=False,
                  student=replace(cfg.student, without_teacher=True, max_delay_steps=0))
    cfg, env = load_env(cfg, None, freeze_env=True, check_viewer=False)
    env.unwrap(target=AddTensorboardWriter).set_writer(None)
    expected = set(json.loads(Path(cfg.env.single_object_scene.filter_file).read_text()))
    if set(env.scene.keys) != expected:
        raise ValueError('Environment object set differs from requested split')
    student_cfg = replace(cfg.student, batch_size=env.num_env)
    if cfg.policy == 'seer':
        # Teacher is used only to reconstruct the latent dimension, then discarded.
        cfg = replace(cfg, net=update_net_cfg(cfg.net, env, cfg.state_net_blocklist))
        teacher = load_agent(cfg, env, None, None)
        teacher_dim = int(teacher.state_net.state_aggregator.cfg.dim_out)
        del teacher
        student = StudentAgentRMA_FT_kN(
            _cast_student_cfg_to_FT(student_cfg, cfg.student_FT_head),
            writer=None, device=env.device, teacher_dim_state=teacher_dim,
            teacher_dim_act=int(env.action_space.shape[0])).to(env.device)
        for name in ('seer_future_residual', 'seer_future_advantage'):
            option = getattr(student.cfg, name, None)
            if option is not None and getattr(option, 'enabled', False):
                raise ValueError(name + ' is outside this evaluator')
    else:
        student = StudentAgentRMA(student_cfg, writer=None, device=env.device).to(env.device)
    load_policy_state(student, th.load(cfg.load_student, map_location='cpu')['self'])
    student.eval()
    env = setup_rma_env_v2(cfg, env, student,
                          state_size=student.cfg.aggregator.state_size,
                          is_student=False, dagger=False)
    norm = env.unwrap(target=NormalizeEnv)
    saved = th.load(cfg.load_environment, map_location='cpu')['normalizer']
    runtime = norm.normalizer.state_dict()
    if set(saved) != set(runtime):
        raise ValueError('Environment normalizer keys differ from checkpoint')
    for key in saved:
        if saved[key].shape != runtime[key].shape or not bool(th.isfinite(saved[key]).all()):
            raise ValueError('Invalid normalizer tensor: ' + key)
    norm.load(cfg.load_environment, strict=True)
    norm.normalizer.eval()
    if cfg.is_phase2 and 'Phase2Training:0' not in getattr(norm, '_restored_environment_state_keys', ()):
        raise ValueError('Paired environment checkpoint did not restore Phase-2 state')
    counter = CountFixedTrialsSuccess(env, cfg.fixed_trials_per_object, cfg.fixed_trials_result_dir)
    try:
        completed = run_fixed_trials(counter, student, cfg.max_eval_steps, cfg.student.use_amp)
    finally:
        counter.save()
    result_path = Path(cfg.fixed_trials_result_dir) / 'fixed_trials_summary.json'
    result = json.loads(result_path.read_text())
    if not result.get('complete') or result['total_trials'] != len(expected) * cfg.fixed_trials_per_object:
        raise RuntimeError('Incomplete evaluation trial bank')
    result.update({'artifact_class': 'NONPREHENSILE_SOFT_DEV', 'policy': cfg.policy,
                   'seed': cfg.env.seed, 'completed_vector_steps': completed,
                   'student_sha256': sha256_file(cfg.load_student),
                   'environment_sha256': sha256_file(cfg.load_environment),
                   'split_sha256': sha256_file(cfg.env.single_object_scene.filter_file)})
    result_path.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
