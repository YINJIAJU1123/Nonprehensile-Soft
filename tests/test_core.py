"""CPU contracts; history methods are isolated from the legacy GPU import graph."""
import ast
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
import torch as th

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'dywa/exp/train'), str(ROOT / 'dywa/src')]
from modules_teacher_FT import FTSidecar, FTSidecarConfig
from student_FT_distill import StudentFTHead, StudentFTHeadConfig, compute_student_FT_loss, build_FT_rollout_targets
from policy_runtime import load_policy_state, run_fixed_trials
from train.source_lineage import ROOT as LINEAGE_ROOT, TRAIN_SOURCE_FILES


def isolated_function(path, name, cls=None):
    tree = ast.parse((ROOT / path).read_text())
    body = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls).body if cls else tree.body
    node = next(n for n in body if isinstance(n, ast.FunctionDef) and n.name == name)
    namespace = {'th': th}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[name]


class CoreTests(unittest.TestCase):
    def test_source_fingerprints_resolve(self):
        self.assertEqual(LINEAGE_ROOT, ROOT)
        self.assertGreater(len(TRAIN_SOURCE_FILES), 170)
        self.assertTrue(all((ROOT / p).is_file() for p in TRAIN_SOURCE_FILES))

    def test_both_history_buffers(self):
        for path, cls in [('dywa/src/models/modules.py', 'HistoryEncoder'), ('dywa/exp/train/seer_history.py', 'SeerHistoryEncoder')]:
            with self.subTest(policy=cls):
                put = isolated_function(path, 'put_history', cls)
                state = SimpleNamespace(history=th.zeros(2, 3, 2, 4), num_tokens=2, cfg=SimpleNamespace(persistent=True), safe_copy=True)
                snapshots = []
                for i in range(1, 5):
                    snapshots.append(put(state, th.full((2, 2, 4), float(i), requires_grad=True)))
                self.assertEqual(state.history[0, :, 0, 0].tolist(), [2., 3., 4.])
                self.assertEqual(snapshots[0][0, :, 0, 0].tolist(), [0., 0., 1.])
                self.assertFalse(snapshots[-1].requires_grad)
                reset = isolated_function('dywa/src/models/modules.py', 'reset_history', 'HistoryEncoder')
                reset(state, th.tensor([True, False]))
                self.assertEqual(state.history[0].count_nonzero().item(), 0)
                self.assertEqual(state.history[1, -1, 0, 0].item(), 4.)
                with self.assertRaises(ValueError):
                    put(state, th.zeros(2, 3, 4))

    def test_future_loss_and_frozen_targets(self):
        th.manual_seed(7)
        cfg = StudentFTHeadConfig(enabled=True, K=3, dim_hidden=16, contact_conditioned=True, contact_context_dim=8)
        head = StudentFTHead(cfg, in_dim=12, dim_act=6, teacher_dim_state=10)
        z = th.randn(4, 12, requires_grad=True)
        context = th.randn(4, 8, requires_grad=True)
        actions = th.randn(4, 3, 6)
        target = th.randn(4, 3, 10, requires_grad=True)
        structured = [{k: th.rand(4, dim, requires_grad=True) for k, dim in [('rel_goal_delta',9), ('object_delta',13), ('contact',1), ('progress',1)]} for _ in range(3)]
        loss, logs = compute_student_FT_loss(head, z, actions, target, structured, contact_context=context, cfg=cfg)
        self.assertTrue(th.isfinite(loss))
        loss.backward()
        self.assertGreater(z.grad.abs().sum().item(), 0)
        self.assertIsNone(target.grad)
        self.assertTrue(all(v.grad is None for s in structured for v in s.values()))
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for n,p in head.named_parameters() if 'contact' in n))
        teacher = FTSidecar(FTSidecarConfig(dim_state=10, dim_act=6, dim_hidden=16))
        rollout, targets = build_FT_rollout_targets(teacher, th.randn(4,10,requires_grad=True), actions)
        self.assertEqual(tuple(rollout.shape), (4,3,10))
        self.assertFalse(rollout.requires_grad)
        self.assertTrue(all(not value.requires_grad for s in targets for value in s.values()))

    def test_episode_windows(self):
        build = isolated_function('dywa/exp/train/train_FT_offline.py', '_build_window_indices')
        buf = {'valid': th.ones(8,2,dtype=th.bool), 'done': th.zeros(8,2,dtype=th.bool), 'episode_id': th.zeros(8,2,dtype=th.long)}
        buf['done'][2,0] = True
        buf['episode_id'][3:,0] = 1
        buf['valid'][5,1] = False
        found = set(map(tuple, build(buf,3).tolist()))
        self.assertEqual(found, {(3,0),(4,0),(0,1),(1,1)})

    def test_strict_checkpoint_loading(self):
        model = th.nn.Linear(2,3)
        good = {k:v.clone() for k,v in model.state_dict().items()}
        load_policy_state(model, good)
        with self.assertRaises(ValueError):
            load_policy_state(model, {'weight': good['weight']})
        good['weight'][0,0] = float('nan')
        with self.assertRaises(FloatingPointError):
            load_policy_state(model, good)

    def test_student_only_rollout_and_incomplete_failure(self):
        class Env:
            num_env=2
            device='cpu'
            def reset(self): self.steps=0; return {}
            def step(self, action):
                self.steps += 1
                th.testing.assert_close(action, th.ones(2,3))
                return {}, None, th.tensor([True,False]), {}
            def done(self): return self.steps == 3
        class Policy:
            def __init__(self): self.resets=[]
            def reset_state(self, done): self.resets.append(done.clone())
            def get_output(self, obs): return th.stack([th.ones(2,3), th.zeros(2,3)],dim=1), {}
        policy=Policy()
        self.assertEqual(run_fixed_trials(Env(),policy,5),3)
        self.assertEqual(len(policy.resets),4)
        with self.assertRaises(RuntimeError): run_fixed_trials(Env(),Policy(),1)

    def test_launch_profiles_and_hydra(self):
        from hydra import initialize_config_dir, compose
        for mode, policy in [('teacher','dywa'),('collect','seer'),('train','dywa'),('train','seer'),('eval','dywa'),('eval','seer')]:
            payload = json.loads(subprocess.check_output([sys.executable,str(ROOT/'scripts/run.py'),mode,'--policy',policy,'--dry-run','--output','/tmp/run with spaces'], text=True))
            with initialize_config_dir(config_dir=str(ROOT/'dywa/src/data/cfg'),version_base=None):
                cfg=compose(config_name='train_rl',overrides=payload['argv'][3:])
            self.assertEqual(cfg.path.root, '/tmp/run with spaces')
            if mode=='train' and policy=='seer':
                self.assertTrue(cfg.student.view_contact_transformer)
                self.assertEqual(cfg.K_window,3)
                self.assertFalse(cfg.student_FT_head.feed_to_policy)
            if mode=='eval':
                self.assertFalse(cfg.add_teacher_state)
                self.assertFalse(cfg.dagger)


if __name__ == '__main__': unittest.main()
