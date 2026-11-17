#!/usr/bin/env python3
"""Portable launch commands; --dry-run needs only the Python standard library."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / 'dywa/exp/train'


def hydra_value(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'))


def build(args):
    env = os.environ.copy()
    env['PYTHONPATH'] = os.pathsep.join(filter(None, [str(ROOT / 'dywa/src'), str(TRAIN), env.get('PYTHONPATH')]))
    env['PYTORCH_JIT'] = '0'
    if args.gpu is not None:
        env['CUDA_VISIBLE_DEVICES'] = args.gpu
    out = Path(args.output or ROOT / 'outputs' / (args.command + '-' + args.policy + '-' + datetime.now().strftime('%Y%m%d-%H%M%S'))).expanduser().resolve()
    required_files, required_dirs = [], []

    def external(name, directory=False):
        raw = env.get(name)
        if not raw and not args.dry_run:
            raise ValueError('Set ' + name + ' (see .env.example)')
        value = str(Path(raw).expanduser().resolve()) if raw else '/SET_ME/' + name
        (required_dirs if directory else required_files).append(Path(value))
        return value

    overrides = {}
    groups = []
    if args.command == 'world-model':
        entry = 'train_FT_offline.py'
        env['BUFFER_PATH'] = external('BUFFER_PATH')
        env['FT_OUT_DIR'] = str(out)
    else:
        entry = {'teacher': 'train_ppo_arm.py', 'collect': 'collect_teacher_buffer.py', 'train': 'train_rma.py' if args.policy == 'dywa' else 'train_rma_FT_kN.py', 'eval': 'evaluate.py'}[args.command]
        groups = ['+platform=debug', '+env=' + ('icra_base' if args.command == 'teacher' else 'abs_goal_3view_ral_v2'), '+run=teacher_base']
        if args.command != 'teacher':
            groups.append('+student=dywa/base')
        split = Path(args.split or ROOT / 'data/splits/ral_v2/teacher80_train_v2.json').expanduser().resolve()
        objects = json.loads(split.read_text())
        if not isinstance(objects, list) or not objects or len(objects) != len(set(objects)):
            raise ValueError('Split must be a nonempty list of unique object keys')
        required_files.append(split)
        dgn = external('DGN_ROOT', directory=True)
        pose = env.get('DGN_POSE_ROOT', str(Path(dgn) / 'unique_dgn_poses'))
        pose = str(Path(pose).expanduser().resolve())
        required_dirs.append(Path(pose))
        overrides.update({
            'name': out.name, 'path.root': str(out), 'path.key': 'run-000',
            'env.seed': args.seed, 'env.num_env': args.num_env,
            'env.graphics_device_id': -1,
            'env.single_object_scene.num_object_types': len(objects),
            'env.single_object_scene.filter_file': str(split),
            'env.single_object_scene.dgn.data_path': dgn,
            'env.single_object_scene.dgn.pose_path': pose,
            'icp_obs.icp.ckpt': external('ICP_CKPT'),
            'global_device': 'cuda:0', 'use_wandb': False, 'use_hfhub': False,
        })
        if args.command != 'teacher' or args.stage == 2:
            overrides['load_ckpt'] = external('TEACHER_CKPT')
            overrides.update({'is_phase2': True,
                'phase2.adaptive_dof_pos_limit': True,
                'phase2.start_dof_pos_offset': .03,
                'phase2.pos_maximum_reduction': .03,
                'phase2.adaptive_residual_scale': False,
                'phase2.initial_residual_scale': [.06, .1],
                'phase2.target_residual_scale': [.02, .03]})
            if args.command != 'teacher':
                overrides['phase2.min_reset_to_update'] = 1000000000
        if args.command == 'teacher':
            overrides['agent.train.train_steps'] = args.steps
        elif args.command == 'collect':
            env['BUFFER_OUT_DIR'] = str(out)
            env['BUFFER_STEPS'] = str(args.steps)
            overrides.update({'env.collect_contact': 'last', 'add_touch_flag': True})
        else:
            overrides.update(json.loads((ROOT / 'configs' / (args.policy + '.json')).read_text()))
            overrides['load_environment'] = external('ENV_CKPT')
            if args.command == 'train':
                overrides.update({'train_step': args.steps, 'save_step': args.save_step})
                if args.policy == 'seer':
                    overrides['FT_ckpt_path'] = external('FT_CKPT')
            else:
                overrides.update({'policy': args.policy, 'load_student': external('STUDENT_CKPT'),
                    'fixed_trials_per_object': args.trials, 'max_eval_steps': args.steps,
                    'fixed_trials_result_dir': str(out), 'dagger': False,
                    'add_teacher_state': False, 'student.without_teacher': True})
    extra = args.overrides
    if extra and extra[0] == '--':
        extra = extra[1:]
    if args.command == 'world-model' and extra:
        raise ValueError('Offline trainer accepts environment variables, not Hydra overrides')
    command = [args.python, '-u', str(TRAIN / entry)] + groups
    command += ['++' + key + '=' + hydra_value(value) for key, value in overrides.items()]
    command += extra
    return command, env, out, required_files, required_dirs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['teacher', 'collect', 'world-model', 'train', 'eval'])
    parser.add_argument('--policy', choices=['dywa', 'seer'], default='seer')
    parser.add_argument('--stage', type=int, choices=[1, 2], default=1)
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--gpu', help='Physical GPU id mapped to cuda:0')
    parser.add_argument('--num-env', type=int, default=512)
    parser.add_argument('--seed', type=int, default=66001)
    parser.add_argument('--steps', type=int, default=200000, help='Training/vector steps (collect: buffer steps; eval: cap)')
    parser.add_argument('--save-step', type=int, default=5000)
    parser.add_argument('--trials', type=int, default=200)
    parser.add_argument('--split')
    parser.add_argument('--output')
    parser.add_argument('--dry-run', action='store_true')
    args, args.overrides = parser.parse_known_args()
    if min(args.num_env, args.steps, args.save_step, args.trials) < 1:
        parser.error('Counts must be positive')
    try:
        command, env, out, files, dirs = build(args)
        if args.dry_run:
            print(json.dumps({'cwd': str(TRAIN), 'argv': command, 'output': str(out), 'runtime_env': {k: env[k] for k in ('PYTHONPATH', 'PYTORCH_JIT', 'CUDA_VISIBLE_DEVICES', 'BUFFER_PATH', 'BUFFER_OUT_DIR', 'BUFFER_STEPS', 'FT_OUT_DIR') if k in env}}, indent=2))
            return
        for path in files:
            if not path.is_file():
                raise FileNotFoundError(path)
        for path in dirs:
            if not path.is_dir():
                raise NotADirectoryError(path)
        # Distinct run roots prevent accidental overwrites of checkpoints/results.
        out.mkdir(parents=True, exist_ok=False)
        (out / 'launch.json').write_text(json.dumps({'argv': command, 'cwd': str(TRAIN)}, indent=2) + '\n')
        print(shlex.join(command), flush=True)
        subprocess.run(command, cwd=TRAIN, env=env, check=True)
    except (ValueError, OSError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
