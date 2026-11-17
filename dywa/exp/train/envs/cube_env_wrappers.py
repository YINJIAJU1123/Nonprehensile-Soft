#!/usr/bin/env python3

from isaacgym import (gymtorch, gymapi, gymutil)
from isaacgym.gymutil import (AxesGeometry, WireframeBBoxGeometry, draw_lines)

from typing import (Optional, Iterable, Callable,
                    Tuple, Dict)
from dataclasses import dataclass
from util.config import ConfigBase, recursive_replace_map
from skimage.color import label2rgb

import matplotlib.pyplot as plt
import pickle
import csv
import json
from pathlib import Path
from gym import spaces
import trimesh

import numpy as np
import torch as th

from pytorch3d.ops.points_alignment import iterative_closest_point

from env.env.base import EnvBase, EnvIface
from env.env.wrap.base import (
    WrapperEnv,
    ObservationWrapper,
    add_obs_field, remove_obs_field
)
from env.env.wrap.monitor_env import MonitorEnv
from env.env.wrap.normalize_env import NormalizeEnv

from env.robot.fe_gripper import FEGripper
from util.path import ensure_directory
from util.torch_util import dcn, dot, merge_shapes
from util.math_util import (
    matrix_from_quaternion,
    quaternion_from_matrix,
    quat_from_axa,

    apply_pose,
    compose_pose_tq,
    invert_pose_tq,
    apply_pose_tq,

    quat_rotate,
    quat_multiply,
    quat_inverse,
    quaternion_to_6D
)
from env.util import draw_keypoints, draw_sphere
from functools import partial
from env.scene.tabletop_with_object_scene import _array_from_map
from train.ckpt import load_ckpt, last_ckpt

# FIXME: better import path
from data.transforms.aff import get_gripper_mesh
from models.cloud.point_mae import (
    subsample
)

import nvtx
from icecream import ic

from omegaconf import OmegaConf
import sys
import logging

try:
    # sys.path.append("/tmp/point2vec/")
    from point2vec.models import Point2Vec
except ImportError:
    logging.warn('Skipping Point2Vec import.')

from models.rl.net.icp import ICPNet
from models.rl.net.pointnet import PointNetEncoder
from models.common import transfer

from copy import deepcopy


def _copy_if(src: th.Tensor, dst: Optional[th.Tensor]) -> th.Tensor:
    if dst is None:
        return src.detach().clone()
    else:
        dst.copy_(src)
    return dst


class DrawPose(WrapperEnv):
    """
    Args:
        env: Base environment to wrap.
        eps: below this mass, boxes will not be drawn.
    """

    def __init__(self, env, pose_fn: Callable[[None], th.Tensor],
                 check_viewer: bool = True):
        super().__init__(env)
        self.check_viewer = check_viewer
        self.pose_fn = pose_fn

    def __draw(self):
        if (self.check_viewer) and (self.viewer is None):
            return
        poses = self.pose_fn()
        if poses is None:
            return

        gym = self.gym
        viewer = self.viewer
        for i in range(self.num_env):
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(*poses[i, 0:3])
            pose.r = gymapi.Quat(*poses[i, 3:7])
            geom = AxesGeometry(0.5, pose)
            draw_lines(geom, gym, viewer,
                       self.envs[i], None)

    def step(self, *args, **kwds):
        out = self.env.step(*args, **kwds)
        self.__draw()
        return out


class DrawGoalPose(DrawPose):
    def __init__(self, env, check_viewer: bool = True):
        super().__init__(env, self.__get_pose, check_viewer)

    def __get_pose(self):
        if self.task.goal.shape[-1] == 7:
            return self.task.goal
        else:
            return None


class DrawObjPose(DrawPose):
    def __init__(self, env, check_viewer: bool = True):
        super().__init__(env, self.__get_pose, check_viewer)

    def __get_pose(self):
        obj_ids = self.scene.cur_ids.long()
        return self.tensors['root'][obj_ids, :7]


class DrawDebugLines(WrapperEnv):
    @dataclass
    class Config(ConfigBase):
        draw_workspace: bool = False
        draw_wrench_target: bool = False
        draw_cube_action: bool = False

    def __init__(self, cfg: Config, env: EnvIface,
                 check_viewer: bool = True):
        super().__init__(env)
        self.check_viewer = check_viewer
        self.cfg = cfg
        self._prev_cube_action: th.Tensor = None

    def draw(self):
        cfg = self.cfg
        # FIXME: introspection
        if isinstance(self.env, EnvBase):
            dt: float = self.env.cfg.dt
        else:
            dt: float = self.env.unwrap(target=EnvBase).cfg.dt

        if (self.check_viewer) and (self.viewer is None):
            return

        if cfg.draw_workspace:
            wss = dcn(self.task.ws_bound)
            for i in range(self.num_env):
                ws = wss[i]
                box = WireframeBBoxGeometry(ws)
                draw_lines(box, self.gym, self.viewer,
                           self.envs[i], None)
            draw_lines(box, self.gym, self.viewer,
                       self.envs[0], None)

        if cfg.draw_wrench_target:
            object_ids = self.scene.cur_ids
            obj_pos = dcn(self.tensors['root'][object_ids.long(), :3])
            wrench = dcn(self._prev_wrench_target)
            k = dt / 0.17
            line = np.stack([obj_pos, obj_pos + k * wrench[..., :3]],
                            axis=-2)  # Num_env X 2 X 3
            line[..., 2] += 0.1
            for i in range(self.num_env):
                self.gym.add_lines(self.viewer,
                                   self.envs[i],
                                   1,
                                   line[None, i],
                                   np.asarray([1, 0, 1], dtype=np.float32)
                                   )

        if cfg.draw_cube_action:
            if self._prev_cube_action is not None:
                cube_state = self.tensors['root'][
                    self.robot.actor_ids.long()]
                cube_pos = dcn(cube_state[..., :3])

                wrench = dcn(self._prev_cube_action)
                k = dt / 0.17
                line = np.stack([cube_pos, cube_pos + k * wrench[..., :3]],
                                axis=-2)  # Num_env X 2 X 3
                line[..., 2] += 0.1
                for i in range(self.num_env):
                    self.gym.add_lines(
                        self.viewer, self.envs[i],
                        1, line[None, i],
                        np.asarray([1, 0, 1],
                                   dtype=np.float32))

    def step(self, action):
        self.draw()
        out = self.env.step(action)
        self._prev_cube_action = action
        return out


class DrawTargetPose(DrawPose):
    def __init__(self, env, check_viewer: bool = True):
        super().__init__(env, self.__get_pose, check_viewer)

    def __get_pose(self):
        # if isinstance(self.robot.pose_error, CartesianControlError):
        target = None
        if self.robot.cfg.ctrl_mode in ['osc', 'CI']:
            target = self.robot.pose_error.target
        else:
            target = self.robot.pose_error.pose_error.target
        return target


class DrawPosBound(WrapperEnv):
    def __init__(self, env, check_viewer: bool = True):
        super().__init__(env)
        self.check_viewer = check_viewer

    def __draw(self):
        if (self.check_viewer) and (self.viewer is None):
            return
        bounds = self.robot.pose_error.pos_bound
        if bounds is None:
            return
        if len(bounds.shape) == 3:
            for i in range(self.num_env):
                bound = bounds[i]
                box = WireframeBBoxGeometry(bound)
                draw_lines(box, self.gym, self.viewer,
                           self.envs[i], None)

    def step(self, action):
        self.__draw()
        return self.env.step(action)


class DrawPatchCenter(WrapperEnv):
    """
    Args:
        env: Base environment to wrap.
        eps: below this mass, boxes will not be drawn.
    """

    def __init__(self, env, check_viewer: bool = True):
        super().__init__(env)
        self.check_viewer = check_viewer
        self._net = None

    def register(self, net):
        self._net = net

    def __get_pose(self):
        obj_ids = self.scene.cur_ids.long()
        return self.tensors['root'][obj_ids, :7]

    def __draw(self):
        if (self.check_viewer) and (self.viewer is None):
            return
        if self.scene.patch_centers is None:
            return
        gym = self.gym
        viewer = self.viewer
        obj_pose = self.__get_pose()
        centers = 1.1 * self.scene.cur_patch_centers
        cur_centers = dcn(obj_pose[..., None, 0:3] +
                          quat_rotate(obj_pose[..., None, 3:7], centers))

        alpha = None
        if self._net is not None:
            if self._net._attn is not None:
                alpha = dcn(self._net._attn)
                patch_indices = alpha.argmax(axis=-1)
                head_colors = label2rgb(np.arange(patch_indices.shape[-1]))

        for index, env in enumerate(self.envs):
            if alpha is not None:
                if True:
                    for hi, pi in enumerate(patch_indices[index]):
                        draw_sphere(gym, viewer, env,
                                    pos=cur_centers[index, pi],
                                    color=tuple(head_colors[hi]),
                                    radius=0.005)
                else:
                    # aggregate attention on that
                    # specific patch across 4 heads
                    net_attn = alpha[index].max(axis=-2)
                    alpha_i = (net_attn / net_attn.max())
                    draw_keypoints(
                        gym,
                        viewer,
                        env,
                        cur_centers[index],
                        alpha=alpha_i,
                        min_alpha=0.9
                    )
            else:
                draw_keypoints(gym, viewer, env,
                               cur_centers[index])

    def step(self, *args, **kwds):
        out = super().step(*args, **kwds)
        self.__draw()
        return out


class AddCubeRobotState(ObservationWrapper):
    """
    Add the full object state into the observation.
    """

    def __init__(self, env: EnvIface, key: str = 'cube_state'):
        super().__init__(env, self._wrap_obs)
        obs_space, update_fn = add_obs_field(env.observation_space,
                                             key,
                                             spaces.Box(-np.inf, np.inf, (13,))
                                             )
        self._obs_space = obs_space
        self._update_fn = update_fn

    @property
    def observation_space(self):
        return self._obs_space

    def _wrap_obs(self, obs):
        cube_state = self.tensors['root'][
            self.robot.actor_ids.long()]
        return self._update_fn(obs, cube_state)


class AddObjectMass(ObservationWrapper):
    """
    Add object mass to observation.

    Args:
        env: Base environment to wrap.
    """

    def __init__(self, env: EnvIface, key: str = 'object_mass',
                 min_mass: float = 0.0,
                 max_mass: float = 2.0):
        super().__init__(env, self._wrap_obs)
        self.__masses = None
        self.__masses_copy = None
        obs_space, update_fn = add_obs_field(
            env.observation_space, key, spaces.Box(
                min_mass, max_mass, (1,)))
        self._obs_space = obs_space
        self._update_fn = update_fn

    def __get_masses(self):
        gym = self.gym
        masses = []
        actor_handles = dcn(self.scene.cur_handles)
        for (env, actor_handle) in zip(self.envs, actor_handles):
            prop = gym.get_actor_rigid_body_properties(env, actor_handle)
            assert (len(prop) == 1)
            mass = prop[0].mass
            masses.append(mass)
        return th.as_tensor(masses,
                            dtype=th.float,
                            device=self.device)

    @property
    def observation_space(self):
        return self._obs_space

    def _wrap_obs(self, obs):
        if self.__masses is None:
            self.__masses = self.__get_masses()
            self.__masses_copy = self.__masses.detach().clone()
        if True:
            delta = th.abs(self.__masses - self.__masses_copy).max()
            if delta > 1e-3:
                raise ValueError('self.__masses has been modified !')
        return self._update_fn(obs, self.__masses[..., None])


class AddObjectEmbedding(ObservationWrapper):
    """
    Add object geometric embeddings to info.

    Args:
        env: Base environment to wrap.
    """

    def __init__(self,
                 env: EnvIface,
                 key: str = 'object_embedding'):
        super().__init__(env, self._wrap_obs)
        emb_shape = tuple(self.scene.cur_embeddings.shape[1:])
        obs_space, update_fn = add_obs_field(
            env.observation_space,
            key,
            spaces.Box(
                -float('inf'),
                +float('inf'),
                emb_shape)
        )
        self._obs_space = obs_space
        self._update_fn = update_fn

    @property
    def observation_space(self):
        return self._obs_space

    def _wrap_obs(self, obs):
        return self._update_fn(obs,
                               self.scene.cur_embeddings)


class AddObjectKeypoint(ObservationWrapper):
    """
    Add object keypoint to info.

    Args:
        env: Base environment to wrap.
    """

    def __init__(self,
                 env: EnvIface,
                 key: str = 'keypoint'):
        super().__init__(env, self._wrap_obs)
        obs_space, update_fn = add_obs_field(
            env.observation_space, key, spaces.Box(
                -float('inf'), +float('inf'), (24,)))
        self._obs_space = obs_space
        self._update_fn = update_fn

    @property
    def observation_space(self):
        return self._obs_space

    def _wrap_obs(self, obs):
        obj_ids = self.scene.cur_ids.long()
        obj_pose = self.tensors['root'][obj_ids, :7]
        bboxes = self.scene.cur_bboxes
        # new_bboxes = rotate_keypoint(bboxes, obj_pose)
        new_bboxes = (quat_rotate(obj_pose[..., None, 3:7], bboxes)
                      + obj_pose[..., None, 0:3])
        return self._update_fn(obs,
                               new_bboxes.reshape(-1, 24))


class AddObjectFullCloud(ObservationWrapper):
    """
    Add object keypoint to info.

    Args:
        env: Base environment to wrap.
    """

    def __init__(self,
                 env: EnvIface,
                 key: str = 'cloud',
                 goal_key: Optional[str] = None,
                 num_point: Optional[int] = 512
                 ):
        super().__init__(env, self._wrap_obs)

        # NOTE: for now, we do not support online sampling
        # or cloud-size configuration in general.
        # assert (self.scene.cloud.shape[-2] == num_point)
        obs_space, update_fn = add_obs_field(
            env.observation_space, key,
            spaces.Box(-float('inf'), +float('inf'), (num_point, 3))
        )

        update_goal_fn = None
        if goal_key is not None:
            ic(F'goal_key={goal_key}')
            obs_space, update_goal_fn = add_obs_field(
                obs_space,
                goal_key,
                spaces.Box(-float('inf'), +float('inf'), (num_point, 3))
            )
            ic(obs_space)

        self._obs_space = obs_space
        self._update_fn = update_fn
        self._update_goal_fn = update_goal_fn

    @property
    def observation_space(self):
        return self._obs_space

    def _wrap_obs(self, obs): ### rel point cloud
        obj_ids = self.scene.cur_ids.long()
        obj_pose = self.tensors['root'][obj_ids, :7]
        base_cloud = self.scene.cur_cloud  # Nx512x3

        # [1] Add object point cloud at object pose.
        obj_cloud = apply_pose(obj_pose[..., None, 3:7],
                               obj_pose[..., None, 0:3], base_cloud)
        out = self._update_fn(obs, obj_cloud)

        # [2] Also add object point cloud at goal pose.
        if self._update_goal_fn is not None:
            goal_pose = self.task.goal
            goal_cloud = apply_pose(goal_pose[..., None, 3:7],
                                    goal_pose[..., None, 0:3], base_cloud)
            out = self._update_goal_fn(out, goal_cloud)

        return out


class AddFingerFullCloud(ObservationWrapper):
    """
    Add point clouds from robot finger to cloud.

    Args:
        env: Base environment to wrap.
    """

    def __init__(self,
                 env: EnvIface,
                 key: str = 'finger_cloud',
                 goal_key: Optional[str] = None,
                 num_point: Optional[int] = 64):
        super().__init__(env, self._wrap_obs)

        self.finger_mesh: trimesh.Trimesh = get_gripper_mesh(
            cat=True, frame='panda_hand', links=[
                'panda_leftfinger', 'panda_rightfinger'])

        # Oversample `finger_pcd`.
        self.num_point = num_point
        self.finger_pcd = trimesh.sample.sample_surface(self.finger_mesh,
                                                        2 * num_point)
        self.finger_pcd = th.as_tensor(self.finger_pcd,
                                       dtype=th.float,
                                       device=self.device)

        # NOTE: for now, we do not support online sampling
        # or cloud-size configuration in general.
        # assert (self.scene.cloud.shape[-2] == num_point)
        obs_space, update_fn = add_obs_field(
            env.observation_space, key,
            spaces.Box(-float('inf'), +float('inf'), (num_point, 3))
        )
        self._obs_space = obs_space
        self._update_fn = update_fn

    @property
    def observation_space(self):
        return self._obs_space

    def _wrap_obs(self, obs):
        # obj_ids = self.scene.cur_ids.long()
        # obj_pose = self.tensors['root'][obj_ids, :7]
        # hand_pose =
        body_tensors = self.tensors['body']
        body_indices = self.robot.ee_body_indices.long()
        hand_pose = body_tensors[body_indices, :]
        base_cloud = subsample(self.finger_pcd,
                               self.num_point)

        # [1] Add object point cloud at object pose.
        obj_cloud = apply_pose(hand_pose[..., None, 3:7],
                               hand_pose[..., None, 0:3], base_cloud)
        out = self._update_fn(obs, obj_cloud)

        return out


class AddPrevCubeWrench(ObservationWrapper):
    def __init__(self, env: EnvIface,
                 key: str = 'previous_wrench',
                 wrench_generator=None,
                 max_force: float = 20.0,):
        super().__init__(env, self._wrap_obs)
        # assert isinstance(env.robot, 'FEGripper')
        obs_space, update_fn = add_obs_field(env.observation_space,
                                             key,
                                             env.robot.action_space)
        self._obs_space = obs_space
        self._update_fn = update_fn

    @property
    def observation_space(self):
        return self._obs_space

    def _wrap_obs(self, obs):
        prev_wrench = self.robot._prev_actions
        return self._update_fn(obs, prev_wrench)


class AddPhysParams(ObservationWrapper):
    """
    Add physics parameters to observation.

    Args:
        env: Base environment to wrap.
    """

    def __init__(self, env: EnvIface,
                 key: str = 'phys_params',
                 min_mass: float = 0.0,
                 max_mass: float = 2.0,
                 min_friction: float = 0.0,
                 max_friction: float = 2.0,
                 min_restitution: float = 0.0,
                 max_restitution: float = 1.0
                 ):
        super().__init__(env, self._wrap_obs)
        self.__masses = None

        # NOTE: `3` stands for
        # table friction, object friction, hand friction,
        # respsectively.
        lo = np.asarray([min_mass] + [min_friction] * 3 + [min_restitution],
                        dtype=np.float32)
        hi = np.asarray([max_mass] + [max_friction] * 3 + [max_restitution],
                        dtype=np.float32)
        obs_space, update_fn = add_obs_field(
            env.observation_space, key, spaces.Box(lo, hi, (5,))
        )
        self._obs_space = obs_space
        self._update_fn = update_fn

    def __get_masses(self):
        gym = self.gym
        masses = []
        actor_handles = dcn(self.scene.cur_handles)
        for (env, actor_handle) in zip(self.envs, actor_handles):
            prop = gym.get_actor_rigid_body_properties(env, actor_handle)
            assert (len(prop) == 1)
            mass = prop[0].mass
            masses.append(mass)
        return th.as_tensor(masses,
                            dtype=th.float,
                            device=self.device)

    @property
    def observation_space(self):
        return self._obs_space

    def _wrap_obs(self, obs):
        if self.__masses is None:
            self.__masses = self.__get_masses()

        mass = self.__masses[..., None]
        tbl_fr = self.scene.cur_table_friction[..., None]
        obj_fr = self.scene.cur_object_friction
        hnd_fr = self.robot.cur_hand_friction[..., None]
        obj_rs = self.scene.cur_object_restitution
        phys = th.cat([mass, tbl_fr, obj_fr, hnd_fr, obj_rs], dim=-1)
        return self._update_fn(obs, phys)


class AddPrevArmWrench(ObservationWrapper):
    def __init__(self, env: EnvIface,
                 key: str = 'previous_wrench'):
        super().__init__(env, self._wrap_obs)
        # assert isinstance(env.robot, 'FEGripper')
        obs_space, update_fn = add_obs_field(env.observation_space,
                                             key,
                                             # env.robot.action_space
                                             spaces.Box(-np.inf, +np.inf, (6,))
                                             )
        self._obs_space = obs_space
        self._update_fn = update_fn

    @property
    def observation_space(self):
        return self._obs_space

    def _wrap_obs(self, obs):
        prev_wrench = self.robot.ee_wrench
        return self._update_fn(obs, prev_wrench)


class AddPrevAction(ObservationWrapper):
    def __init__(self, env: EnvIface,
                 key: str = 'previous_action',
                 zero_out: bool = False):
        super().__init__(env, self._wrap_obs)
        self._zero_out = zero_out

        if isinstance(env.robot.action_space, spaces.Discrete):
            # By default, __prev_action will be stored as one-hot.
            act_obs_space = spaces.Box(0.0, 1.0, (env.robot.action_space.n,))
            self.__prev_action = th.zeros(
                (env.num_env, env.robot.action_space.n),
                dtype=th.float,
                device=env.device)
        else:
            act_obs_space = env.robot.action_space
            self.__prev_action = th.zeros(
                merge_shapes(env.num_env, env.robot.action_space.shape),
                dtype=th.float,
                device=env.device)

        obs_space, update_fn = add_obs_field(env.observation_space,
                                             key,
                                             act_obs_space)
        self._obs_space = obs_space
        self._update_fn = update_fn

    @property
    def observation_space(self):
        return self._obs_space

    def _wrap_obs(self, obs):
        if self.__prev_action is None:
            return obs
        # FIXME: we multiply by 0.0 only for compatibility
        if self._zero_out:
            return self._update_fn(obs, 0.0 * self.__prev_action)
        else:
            return self._update_fn(obs, self.__prev_action)

    def reset_indexed(self,
                      indices: Optional[Iterable[int]] = None):
        # FIXME: this will fail in case
        # `0` is a meaningful action in the domain.
        if indices is None:
            self.__prev_action.fill_(0)
        else:
            self.__prev_action.index_fill_(0, indices, 0)
        return super().reset_indexed(indices)

    def step(self, act):
        obs, rew, done, info = super().step(act)
        self.__prev_action = act
        obs = self._wrap_obs(obs)
        return (obs, rew, done, info)


class AddWrenchPenalty(WrapperEnv):
    def __init__(self,
                 env: EnvIface,
                 k_wrench: float,
                 key: Optional[str] = 'env/wrench_cost',
                 log_period: Optional[int] = 1024
                 ):
        super().__init__(env)
        self.__k_wrench = k_wrench

        # == logging stuff ==
        self.__key = key
        self.__log_period = log_period
        self.__step_since_log = 0
        # FIXME: since __global_step is a
        # private variable, it's very much a "local_step"...
        self.__global_step = 0

    def step(self, *args, **kwds):
        obs, rew, done, info = super().step(*args, **kwds)
        wrench_cost = self.__k_wrench * th.linalg.norm(
            obs['previous_wrench'], dim=-1)
        rew = rew - wrench_cost

        # TODO: Can we somehow streamline the
        # logging process across multiple different loggers?
        self.__step_since_log += 1
        self.__global_step += 1
        if (self.writer is not None) and (
                self.__step_since_log >= self.__log_period):
            self.writer.add_scalar(self.__key,
                                   wrench_cost.mean().item(),
                                   global_step=self.__global_step)
            self.__step_since_log = 0

        return (obs, rew, done, info)


class AddWrenchTarget(ObservationWrapper):
    def __init__(self, env: EnvIface,
                 key: str = 'wrench_target',
                 wrench_generator=None):
        super().__init__(env, self._wrap_obs)
        self.wrench_generator = wrench_generator
        obs_space, update_fn = add_obs_field(env.observation_space,
                                             key,
                                             wrench_generator.observation_space
                                             )
        self._obs_space = obs_space
        self._update_fn = update_fn

    @property
    def observation_space(self):
        return self._obs_space

    def _wrap_obs(self, obs):
        wrench_target = self.wrench_generator(obs)
        return self._update_fn(obs, wrench_target)


class MatchWrenchTarget(WrapperEnv):
    """
    In the case wrench_target is available,
    add wrench-matching auxiliary reward to the environment reward term.
    """

    @dataclass
    class Config(ConfigBase):
        wrench_coefs: Tuple[float, float, float, float, float, float] = (
            1e-1, 1e-1, 1e-1,
            # 1e-3, 1e-3, 1e-3
            0, 0, 0
        )

    def __init__(self, cfg: Config, env: EnvIface):
        super().__init__(env)
        self.cfg = cfg
        self._prev_wrench_target: th.Tensor = th.zeros(
            (env.num_env, 6), dtype=th.float,
            device=self.device)

        self.k_w = th.as_tensor(cfg.reward.wrench_coefs,
                                device=env.device,
                                dtype=th.float)

    def _aux_rew(self, target, actual):
        wrench_reward = dot(
            actual * self.k_w,
            target * self.k_w
        )
        return wrench_reward

    def reset_indexed(self,
                      indices: Optional[Iterable[int]] = None):
        if indices is None:
            self._prev_wrench_target.fill_(0)
        else:
            self._prev_wrench_target.index_fill_(0, indices, 0)
        return self.env.reset_indexed(indices)

    def step(self, *args, **kwds):
        obs, rew, done, info = super().step(*args, **kwds)

        # FIXME: this is not entirely correct
        wrench = self.tensors['force_sensor'].squeeze(dim=1)
        rew = rew + self._aux_rew(self._prev_wrench_target, wrench)

        if 'wrench_target' in obs:
            self._prev_wrench_target[...] = obs['wrench_target']
        return (obs, rew, done, info)


class QuatToDCM(ObservationWrapper):
    def __init__(self, env, entries: Dict[str, int]):
        super().__init__(env, self._wrap_obs)
        self.__entries = entries

        src_spaces = {k: v for (k, v) in env.observation_space.items()}
        dst_spaces = dict(src_spaces)
        for k, v in entries.items():
            obs_space = env.observation_space[k]
            assert (isinstance(obs_space, spaces.Box))
            assert (len(obs_space.shape) == 1)
            lo = np.concatenate([
                obs_space.low[:v],
                [-1] * 9,
                obs_space.low[v + 4:]])
            hi = np.concatenate([
                obs_space.high[:v],
                [+1] * 9,
                obs_space.high[v + 4:]])
            dst_spaces[k] = spaces.Box(lo, hi)
        self.__obs_space = spaces.Dict(dst_spaces)

    @property
    def observation_space(self):
        return self.__obs_space

    def _wrap_obs(self, obs):
        out = dict(obs)
        for k, v in self.__entries.items():
            src = obs[k]
            q = src[..., v:v + 4]
            dcm = matrix_from_quaternion(q).reshape(
                *q.shape[:-1], -1)
            dst = th.cat([
                src[..., :v],
                dcm,
                src[..., v + 4:]
            ], dim=-1)
            out[k] = dst
        return out


class QuatTo6D(ObservationWrapper):
    def __init__(self, env, entries: Dict[str, int]):
        super().__init__(env, self._wrap_obs)
        self.__entries = entries

        src_spaces = {k: v for (k, v) in env.observation_space.items()}
        dst_spaces = dict(src_spaces)
        for k, v in entries.items():
            obs_space = env.observation_space[k]
            assert (isinstance(obs_space, spaces.Box))
            assert (len(obs_space.shape) == 1)
            lo = np.concatenate([
                obs_space.low[:v],
                [-1] * 6,
                obs_space.low[v + 4:]])
            hi = np.concatenate([
                obs_space.high[:v],
                [+1] * 6,
                obs_space.high[v + 4:]])
            dst_spaces[k] = spaces.Box(lo, hi)
        self.__obs_space = spaces.Dict(dst_spaces)

    @property
    def observation_space(self):
        return self.__obs_space

    def _wrap_obs(self, obs):
        out = dict(obs)
        for k, v in self.__entries.items():
            src = obs[k]
            q = src[..., v:v + 4]
            dcm = matrix_from_quaternion(q)[..., :, :2].reshape(
                *q.shape[:-1], -1)
            dst = th.cat([
                src[..., :v],
                dcm,
                src[..., v + 4:]
            ], dim=-1)
            out[k] = dst
        return out


class RelGoal(ObservationWrapper):
    """
    Rewrite goal so that it's relative to the
    current pose of the object.
    """

    def __init__(self, env,
                 key_goal: str = 'goal',
                 key_obj: str = 'object_state',
                 use_6d: bool = True
                 ):
        super().__init__(env, self._wrap_obs)
        self.__key_goal = key_goal
        self.__key_obj = key_obj
        self.__use_6d = use_6d

        n: int = 9 if use_6d else 7
        obs_space, update_fn = add_obs_field(
            env.observation_space,
            'rel_goal', spaces.Box(-1.0, 1.0, (n,))
        )
        self._obs_space = obs_space
        self._update_fn = update_fn

    @property
    def observation_space(self):
        return self._obs_space

    def _wrap_obs(self, obs):
        T0 = obs[self.__key_obj]
        T1 = obs[self.__key_goal]
        # dq = quat_multiply(
        #     quat_inverse(T1[..., 3:7]),
        #     T0[..., 3:7])
        dq = quat_multiply(T1[..., 3:7], quat_inverse(T0[..., 3:7]))
        # dt = T1[..., 0:3] - quat_rotate(dq, T0[..., 0:3])
        dt = T1[..., 0:3] - T0[..., 0:3]
        if self.__use_6d:
            rot_mat = matrix_from_quaternion(dq)
            dq = rot_mat[..., :, :2].reshape(*rot_mat.shape[:-2], -1)
        dT = th.cat([dt, dq], dim=-1)
        return self._update_fn(obs, dT)

class AddAbsGoal(ObservationWrapper):
    """
    Rewrite goal so that it's relative to the
    current pose of the object.
    """

    def __init__(self, env,
                 key_goal: str = 'goal',
                 use_6d: bool = True
                 ):
        super().__init__(env, self._wrap_obs)
        self.__key_goal = key_goal
        self.__use_6d = use_6d

        n: int = 9 if use_6d else 7
        obs_space, update_fn = add_obs_field(
            env.observation_space,
            'abs_goal', spaces.Box(-1.0, 1.0, (n,))
        )
        self._obs_space = obs_space
        self._update_fn = update_fn

    @property
    def observation_space(self):
        return self._obs_space

    def _wrap_obs(self, obs):
        T = obs[self.__key_goal]

        t = T[..., 0:3]
        q = T[..., 3:7]
        if self.__use_6d:
            rot_mat = matrix_from_quaternion(q)
            q = rot_mat[..., :, :2].reshape(*rot_mat.shape[:-2], -1)
        T1 = th.cat([t, q], dim=-1)
        return self._update_fn(obs, T1)

class AbsAndRelGoal(ObservationWrapper):
    """
    Rewrite goal to include both absolute and relative goals.
    """

    def __init__(self, env,
                 key_goal: str = 'goal',
                 key_obj: str = 'object_state',
                 use_6d: bool = True
                 ):
        super().__init__(env, self._wrap_obs)
        self.__key_goal = key_goal
        self.__key_obj = key_obj
        self.__use_6d = use_6d

        n: int = 9 if use_6d else 7
        obs_space, update_abs_fn = add_obs_field(
            env.observation_space,
            'abs_goal', spaces.Box(-1.0, 1.0, (n,))
        )
        obs_space, update_rel_fn = add_obs_field(
            obs_space,
            'rel_goal', spaces.Box(-1.0, 1.0, (n,))
        )
        obs_space, self._remove_fn = remove_obs_field(obs_space, key= key_goal)
        self._obs_space = obs_space
        self._update_abs_fn = update_abs_fn
        self._update_rel_fn = update_rel_fn

    @property
    def observation_space(self):
        return self._obs_space

    def _wrap_obs(self, obs):
        T0 = obs[self.__key_obj]
        T1 = obs[self.__key_goal]

        # 计算相对目标
        q1 = T1[..., 3:7]
        dq = quat_multiply(q1, quat_inverse(T0[..., 3:7]))
        dt = T1[..., 0:3] - T0[..., 0:3]
        if self.__use_6d:
            dq = quaternion_to_6D(dq)
            q1 = quaternion_to_6D(T1[..., 3:7])
        dT = th.cat([dt, dq], dim=-1)
        T = th.cat([T1[..., 0:3], q1], dim = -1)

        # 更新观察字典，保留绝对和相对目标
        obs = self._update_rel_fn(obs, dT)
        # obs = self._update_abs_fn(obs, dT)
        obs = self._update_abs_fn(obs, T)
        obs = self._remove_fn(obs)  # 添加绝对目标
        return obs

class CopyObs(ObservationWrapper):
    """
    Rewrite goal so that it's relative to the
    current pose of the object.
    """

    def __init__(self, env,
                 key_origin: str = None,
                 key_new: str = None,
                 space = spaces.Box(-1.0, 1.0, (9,))
                 ):
        super().__init__(env, self._wrap_obs)
        obs_space, update_fn = add_obs_field(
            env.observation_space,
            key_new, space)

        self._obs_space = obs_space
        self._update_fn = update_fn
        self.key_origin = key_origin

    @property
    def observation_space(self):
        return self._obs_space

    def _wrap_obs(self, obs):
        T = deepcopy(obs[self.key_origin])

        return self._update_fn(obs, T)


class AddTrackingReward(WrapperEnv):
    def __init__(self, env,
                 coef: float = 1e-3,
                 max_dr: float = 1e-4
                 ):
        super().__init__(env)
        self.coef = coef
        self.max_dr = max_dr
        self.__prv_pose = None
        self.__prv_cloud = None
        self.__prv_error = None
        self.__has_prv = th.zeros((env.num_env,),
                                  dtype=bool,
                                  device=env.device)
        # self.__test_icp()

    def __test_icp(self):
        pcd0 = th.randn(size=(1, 512, 3))
        t = 0.01 * th.randn(size=(1, 1, 3))
        q = quat_from_axa(0.01 * th.randn(size=(1, 1, 3)))
        tq = th.cat([t, q], dim=-1)
        pcd1 = apply_pose_tq(tq, pcd0)
        ic(tq, self.__icp(pcd0, pcd1))

    # @th.jit.script
    @nvtx.annotate("icp-outer")
    def __icp(self, pcd0, pcd1):
        sol = iterative_closest_point(
            pcd0, pcd1,
            max_iterations=16)
        # `sol.R` is called R but it's actually R.T
        RT = sol.RTs.R
        t = sol.RTs.T
        # X.R+t=X1
        q = quaternion_from_matrix(RT.swapaxes(-1, -2))
        return th.cat([t, q], dim=-1)

    @nvtx.annotate("pose_error")
    def __pose_error(self,
                     pose0,
                     pose1,
                     radius: th.Tensor):
        # Nx3x3
        cardinal_points = radius[..., None, None] * th.eye(3,
                                                           dtype=th.float,
                                                           device=self.device)
        error = (
            apply_pose_tq(pose0[..., None, :], cardinal_points)
            - apply_pose_tq(pose1[..., None, :], cardinal_points)
        )
        return th.linalg.norm(error, dim=-1).mean(dim=-1)

    def step(self, *args, **kwds):
        obs, rew, done, info = super().step(*args, **kwds)

        with th.no_grad():
            obj_ids = self.scene.cur_ids.long()
            cur_pose = self.tensors['root'][obj_ids, :7]
            cur_cloud = obs['cloud']
            cur_radius = self.scene.cur_radii

            if self.__prv_pose is None:
                self.__prv_pose = cur_pose
            if self.__prv_cloud is None:
                self.__prv_cloud = cur_cloud

            true_pose_delta = compose_pose_tq(cur_pose,
                                              invert_pose_tq(self.__prv_pose))
            pred_pose_delta = self.__icp(self.__prv_cloud, cur_cloud)

            cur_error = self.__pose_error(pred_pose_delta,
                                          true_pose_delta,
                                          cur_radius)
            if self.__prv_error is None:
                self.__prv_error = cur_error
            dr = self.coef * self.__has_prv * (
                cur_error - self.__prv_error)
            rew = rew + dr.clamp_(-self.max_dr, +self.max_dr)

            # Update cache
            with nvtx.annotate("icp/copy-prv"):
                self.__prv_pose = _copy_if(cur_pose, self.__prv_pose)
                self.__prv_cloud = _copy_if(cur_cloud, self.__prv_cloud)
                self.__prv_error = _copy_if(cur_error, self.__prv_error)
                self.__has_prv[...] = ~done

        return obs, rew, done, info


class AddApproxTouchFlag(ObservationWrapper):
    def __init__(self, env: EnvIface,
                 key: str = 'touch',
                 min_force: float = 5e-2,
                 min_speed: float = 1e-2
                 ):
        super().__init__(env, self._wrap_obs)
        obs_space, update_fn = add_obs_field(env.observation_space,
                                             key,
                                             spaces.Box(0.0, 1.0, (1,))
                                             )
        self._obs_space = obs_space
        self._update_fn = update_fn
        self._min_force = min_force
        self._min_speed = min_speed

    @property
    def observation_space(self):
        return self._obs_space

    def reset_indexed(self,
                      indices: Optional[Iterable[int]] = None):
        return self.env.reset_indexed(indices)

    def _wrap_obs(self, obs):
        if isinstance(self.robot, FEGripper):
            hand_state = self.tensors['root'][self.robot.actor_ids.long()]
        else:
            # franka
            # hand_state = self.tensors['root'][self.robot.actor_ids.long()]
            hand_ids = self.robot.ee_body_indices.long()
            hand_state = self.tensors['body'][hand_ids, :]

        obj_state = self.tensors['root'][self.scene.cur_ids.long()]

        # Check if bounding spheres overlap.
        dist = th.linalg.norm(hand_state[..., :3] - obj_state[..., :3],
                              dim=-1)
        rad = (self.robot.robot_radius + self.scene.cur_radii)
        is_near = (dist < rad)

        # Check if object is receiving sideways force.
        contact_tensor = self.tensors['net_contact']
        object_ids = self.scene.body_ids
        object_contact = contact_tensor[object_ids.long()]
        lateral_force = th.linalg.norm(object_contact[..., :2], dim=-1)
        lateral_push = (lateral_force > self._min_force)

        # Check if object has nonzero velocity.
        moving = th.linalg.norm(obj_state[..., 7:10], dim=-1) > self._min_speed

        touch = th.logical_and(th.logical_and(
            is_near, lateral_push), moving)
        # if touch.sum() > 0:
        #    ic('touch')
        return self._update_fn(obs, touch.float()[..., None])


class AddTouchCount(ObservationWrapper):
    def __init__(self, env: EnvIface, key: str = 'touch_count'):
        super().__init__(env, self._wrap_obs)
        obs_space, update_fn = add_obs_field(env.observation_space,
                                             key,
                                             spaces.Box(0.0, 1.0, (1,)))
        self._obs_space = obs_space
        self._update_fn = update_fn
        self.__touch_count: th.Tensor = th.zeros(
            (env.num_env, 1), dtype=th.float,
            device=self.device)

    @property
    def observation_space(self):
        return self._obs_space

    def reset_indexed(self,
                      indices: Optional[Iterable[int]] = None):
        if indices is None:
            self.__touch_count.fill_(0)
        else:
            self.__touch_count.index_fill_(0, indices, 0)
        return self.env.reset_indexed(indices)

    def _wrap_obs(self, obs):
        self.__touch_count += obs['touch']
        return self._update_fn(obs, self.__touch_count.float())


class AddSuccessAsObs(WrapperEnv):
    def __init__(self, env: EnvIface, key: str = 'success'):
        super().__init__(env)
        obs_space, update_fn = add_obs_field(env.observation_space,
                                             key,
                                             spaces.Box(0.0, 1.0, ()))
        self._obs_space = obs_space
        self._update_fn = update_fn

    @property
    def observation_space(self):
        return self._obs_space

    def reset(self):
        suc = th.zeros((self.num_env,),
                       dtype=th.float,
                       device=self.device)
        obs = self.env.reset()
        return self._update_fn(obs, suc)

    def step(self, actions: th.Tensor):
        obs, rew, done, info = self.env.step(actions)
        obs = self._update_fn(obs, info['success'].float())
        return (obs, rew, done, info)


class P2VembObs(ObservationWrapper):

    @dataclass
    class Config(ConfigBase):
        dim_in: Tuple[int, ...] = (512, 3)  # cloud size
        cfg_path: str = "/tmp/p2v/shapenet.yaml"
        ckpt_path: str = "/tmp/p2v/pre_point2vec-epoch.799-step.64800.ckpt"
        use_amp: bool = True

    def __init__(self, env: EnvIface, cfg: Config, key: str = 'p2v_emb'):
        super().__init__(env, self._wrap_obs)
        self.cfg = cfg
        obs_space, update_fn = add_obs_field(env.observation_space,
                                             key,
                                             spaces.Box(-np.inf, +np.inf,
                                                        (cfg.dim_in[0] // 32, 384)))
        self._obs_space = obs_space
        self._update_fn = update_fn

        self.p2v = OmegaConf.load(cfg.cfg_path)
        self.encoder = Point2Vec(**OmegaConf.to_container(self.p2v.model))
        # Load pretrained point2vec encoder
        ckpt = th.load(cfg.ckpt_path)
        self.encoder.load_state_dict(ckpt['state_dict'], strict=False)
        self.encoder.to(device=env.device)
        # Frozen encoder
        self.encoder.eval()

    @property
    def observation_space(self):
        return self._obs_space

    def _wrap_obs(self, obs):
        cfg = self.cfg
        with th.inference_mode():
            with th.cuda.amp.autocast(cfg.use_amp):
                embeddings, centers = self.encoder.tokenizer(obs['cloud'])
                pos = self.encoder.positional_encoding(centers)
                # Get patch embeddings
                emb = self.encoder.student(embeddings, pos).last_hidden_state
        return self._update_fn(obs, emb)


class ICPEmbObs(ObservationWrapper):
    @dataclass
    class Config(ConfigBase):
        icp: ICPNet.Config = recursive_replace_map(
            ICPNet.Config(), {
                'dim_in': (512, 3),
                'headers': [],
                'encoder_channel': 128,
                'num_query': 1,
                'ckpt': "corn/corn-/col:col-052-1800",
                'keys': {'hand_state': 7},
                'pre_ln_bias': True,
                'encoder.num_hidden_layers': 2,
                'patch_size': 32,
                'p_drop': 0.0,
                'patch_encoder_type': 'mlp',
                'patch_overlap': 1.0,
                'group_type': 'fps',
                'patch_type': 'mlp',
            })

        def __post_init__(self):
            self.icp.headers = ()

    def __init__(self,
                 env: EnvIface,
                 cfg: Config,
                 key: str = 'icp_emb',
                 cloud_key: str = 'cloud'):
        super().__init__(env, self._wrap_obs)

        patch_size = cfg.icp.patch_size
        num_patch = cfg.icp.dim_in[0] // patch_size
        embed_size = cfg.icp.encoder_channel  # 128
        if cfg.icp.keys is None:
            num_keys = 0
        else:
            num_keys = len(cfg.icp.keys)
        num_token = num_keys + num_patch
        obs_space, update_fn = add_obs_field(env.observation_space,
                                             key,
                                             spaces.Box(-np.inf, +np.inf,
                                                        (num_token, embed_size)))
        self._obs_space = obs_space
        self._update_fn = update_fn
        self._cloud_key = cloud_key

        self.encoder = ICPNet(cfg.icp)
        self.encoder.to(device=env.device)
        # Frozen encoder
        self.encoder.eval()

    @property
    def observation_space(self):
        return self._obs_space

    def _wrap_obs(self, obs):
        with th.inference_mode():
            ctx = {}
            _, emb = self.encoder(obs[self._cloud_key], obs)
        return self._update_fn(obs, emb)


class PNEmbObs(ObservationWrapper):
    @dataclass
    class Config(ConfigBase):
        pn: PointNetEncoder.Config = recursive_replace_map(
            PointNetEncoder.Config(), {
                'dim_in': (512, 3),
                # 'headers': [],
                # 'encoder_channel': 128,
                # 'num_query': 1,
                # TODO: fill in with default ckpt
                # 'ckpt': "corn/corn-/col:col-052-1800",
                'keys': {'hand_state': 9},
                # 'pre_ln_bias': True,
                # 'encoder.num_hidden_layers': 2,
                # 'patch_size': 32,
                # 'p_drop': 0.0,
                # 'patch_encoder_type': 'mlp',
                # 'patch_overlap': 1.0,
                # 'group_type': 'fps',
                # 'patch_type': 'mlp',
            })
        use_amp: bool = True
        ckpt: Optional[str] = None

    def __init__(self,
                 env: EnvIface,
                 cfg: Config,
                 key: str = 'pn_emb',
                 cloud_key: str = 'cloud'):
        super().__init__(env, self._wrap_obs)
        self.cfg = cfg
        obs_space, update_fn = add_obs_field(env.observation_space,
                                             key,
                                             spaces.Box(-np.inf, +np.inf,
                                                        (cfg.pn.dim_out,)))
        self._obs_space = obs_space
        self._update_fn = update_fn
        self._cloud_key = cloud_key

        self.encoder = PointNetEncoder(cfg.pn)

        if cfg.ckpt is not None:
            # FIXME: does not work
            # load_ckpt(dict(model=self.encoder),
            #           last_ckpt(cfg.ckpt),
            #           strict=True)
            if not Path(cfg.ckpt).exists():
                filename = last_ckpt(cfg.ckpt)
            else:
                filename = cfg.ckpt
            params = th.load(filename, map_location='cpu')
            output = transfer(self.encoder,
                              params['model'],
                              prefix_map={
                                  'encoder.': '',
                              },
                              verbose=True)
            print(F'output from transfer = {output}')

        self.encoder.to(device=env.device)
        # Frozen encoder
        self.encoder.eval()

    @property
    def observation_space(self):
        return self._obs_space

    def _wrap_obs(self, obs):
        cfg = self.cfg
        with th.inference_mode():
            with th.cuda.amp.autocast(cfg.use_amp):
                CHUNK: int = 2
                m = (obs[self._cloud_key].shape[0] + (CHUNK - 1)) // CHUNK
                embs = [None for _ in range(CHUNK)]
                for i in range(CHUNK):
                    ctx_i = {
                        'hand_state': obs['hand_state'][i * m: (i + 1) * m]}
                    embs[i] = self.encoder(
                        obs[self._cloud_key][i * m:(i + 1) * m], ctx_i)
                emb = th.cat(embs, dim=0)
                # emb = self.encoder(obs[self._cloud_key], obs)
        return self._update_fn(obs, emb.clone())


class CountCategoricalSuccess(WrapperEnv):

    def __init__(self, env: EnvIface,
                 save_dir: Optional[str] = None):
        super().__init__(env)
        if save_dir is None:
            sets = '-'.join(env.scene.cfg.base_set)
            save_dir = Path('output/test_rma/dywa/result') / sets
        self._save_dir: Path = ensure_directory(save_dir)
        self.counting = {}

    def reset(self):
        self.counting = {}
        return self.env.reset()

    def step(self, actions: th.Tensor):
        obs, rew, done, info = self.env.step(actions)
        env_reset_indices = done.view(
            self.env.num_env, 1).all(
            dim=1).nonzero(
            as_tuple=False)
        if len(env_reset_indices) > 0:
            for idx in env_reset_indices:
                name = self.env.scene.cur_names[idx.item()]
                # print(name, idx)
                if name in self.counting:
                    self.counting[name]['reset_count'] += 1
                    self.counting[name]['success_count'] += (
                        info['success'][idx].float()
                    )
                else:
                    temp = {}
                    temp['reset_count'] = 1
                    temp['success_count'] = info['success'][idx].float()
                    self.counting[name] = temp

        return (obs, rew, done, info)

    def save(self, threshold: Optional[float] = None):
        x = []
        y = []
        Keys = list(self.counting.keys())
        Keys.sort()
        sorted_results = {i: self.counting[i] for i in Keys}

        total_success = 0
        total_trials = 0

        for k, v in sorted_results.items():
            sr = (v['success_count'] / v['reset_count']).detach().cpu().item()
            print(
                F"{k} succeed {v['success_count'].item()}"
                F"with {v['reset_count']} tries: {sr}")
            x.append(k.split('/')[-1])
            y.append(sr)
            total_success += v['success_count'].item()
            total_trials += v['reset_count']

        # Print and save global average success rate
        if total_trials > 0:
            avg_success_rate = total_success / total_trials
            print(f"\nGlobal average success rate: {avg_success_rate:.4f}")
            with open(str(self._save_dir / 'avg_success_rate.txt'), "w") as f:
                f.write(f"{avg_success_rate:.4f}\n")
        else:
            print("\nGlobal average success rate: Cannot calculate (total trials is 0)")

        x = np.array(x)
        y = np.array(y)
        indices = np.argsort(y)
        y = np.sort(y)
        x = x[indices]
        fig, ax = plt.subplots(figsize=(15, 30))
        y_pos = np.arange(len(x))

        ax.barh(y_pos, y)
        ax.set_yticks(y_pos, labels=x)
        ax.invert_yaxis()  # labels read top-to-bottom
        ax.set_xlabel('Performance')
        ax.axvline(x=0.6, color='r')
        plt.tight_layout()

        plt.savefig(str(self._save_dir /
                        'categorical_result.png'), dpi=400)
        with open(str(self._save_dir / 'categorical_result.pkl'), "wb") as fp:
            pickle.dump(sorted_results, fp)


class CountFixedTrialsSuccess(WrapperEnv):
    PHASE_NAMES = (
        'free_space', 'pre_contact', 'contact_stalled',
        'contact_moving', 'near_goal')

    def __init__(self, env: EnvIface, trials_per_object: int,
                 save_dir: Optional[str] = None):
        super().__init__(env)
        self.trials_per_object = int(trials_per_object)
        if save_dir is None:
            sets = '-'.join(env.scene.cfg.base_set)
            save_dir = Path('output/test_rma/dywa/fixed_trials') / sets
        self._save_dir: Path = ensure_directory(save_dir)
        self.counting = {}
        self.target_names = sorted({str(k) for k in env.scene.keys})
        self._diag_enabled = True
        self._diag_error_logged = False
        self.trial_rows = []
        self._init_diag_buffers()

    def _init_diag_buffers(self):
        n = self.env.num_env
        device = self.env.device
        self._ep_start_goal_dist = th.full((n,), float('nan'), device=device)
        self._ep_min_goal_dist = th.full((n,), float('inf'), device=device)
        self._ep_min_hand_obj_dist = th.full((n,), float('inf'), device=device)
        self._ep_action_norm_sum = th.zeros((n,), device=device)
        self._ep_action_goal_cos_sum = th.zeros((n,), device=device)
        self._ep_step_count = th.zeros((n,), device=device)
        self._ep_wm_gate_sum = th.zeros((n,), device=device)
        self._ep_wm_gate_count = th.zeros((n,), device=device)
        self._ep_wm_residual_norm_sum = th.zeros((n,), device=device)
        self._ep_wm_residual_count = th.zeros((n,), device=device)
        p = len(self.PHASE_NAMES)
        self._ep_phase_steps = th.zeros((n, p), device=device)
        self._ep_phase_residual_norm_sum = th.zeros((n, p), device=device)
        self._ep_phase_residual_active = th.zeros((n, p), device=device)
        self._ep_phase_goal_delta_sum = th.zeros((n, p), device=device)
        self._ep_lost_contact = th.zeros((n,), device=device)
        self._prev_goal_dist = th.full((n,), float('nan'), device=device)
        self._prev_phase = th.full((n,), -1, dtype=th.long, device=device)
        self._step_residual_norm = th.zeros((n,), device=device)
        self._step_residual_active = th.zeros((n,), dtype=th.bool, device=device)
        self._ep_initial_obj_pose = th.full((n, 7), float('nan'), device=device)
        self._ep_initial_goal = th.full((n, 7), float('nan'), device=device)

    def reset(self):
        self.counting = {}
        self.trial_rows = []
        obs = self.env.reset()
        self._init_diag_buffers()
        return obs

    def _entry(self, name: str):
        if name not in self.counting:
            self.counting[name] = {
                'reset_count': 0,
                'success_count': 0.0,
                'timeout_count': 0.0,
                'final_goal_dist_sum': 0.0,
                'start_goal_dist_sum': 0.0,
                'min_goal_dist_sum': 0.0,
                'progress_sum': 0.0,
                'min_hand_obj_dist_sum': 0.0,
                'mean_action_norm_sum': 0.0,
                'mean_action_goal_cos_sum': 0.0,
                'contact_like_count': 0.0,
                'no_contact_count': 0.0,
                'contact_no_progress_count': 0.0,
                'moved_away_count': 0.0,
                'success_progress_sum': 0.0,
                'failure_progress_sum': 0.0,
                'success_final_goal_dist_sum': 0.0,
                'failure_final_goal_dist_sum': 0.0,
                'wm_gate_sum': 0.0,
                'wm_gate_episode_count': 0.0,
                'wm_residual_norm_sum': 0.0,
                'wm_residual_episode_count': 0.0,
                'phase_step_count': {name: 0.0 for name in self.PHASE_NAMES},
                'phase_residual_norm_sum': {
                    name: 0.0 for name in self.PHASE_NAMES},
                'phase_residual_active_count': {
                    name: 0.0 for name in self.PHASE_NAMES},
                'phase_goal_delta_sum': {
                    name: 0.0 for name in self.PHASE_NAMES},
                'lost_contact_count': 0.0,
            }
        return self.counting[name]

    def record_policy_aux(self, aux: Optional[Dict] = None):
        """Accumulate optional policy diagnostics for the action being used."""
        if not aux:
            return

        gate = aux.get('wm_gate', None)
        if gate is not None:
            gate = gate.detach()
            while gate.ndim > 1:
                gate = gate.mean(dim=-1)
            self._ep_wm_gate_sum += gate.to(self._ep_wm_gate_sum.device)
            self._ep_wm_gate_count += 1

        residual = aux.get('wm_residual_mu', None)
        if residual is None:
            base_output = aux.get('base_output', None)
            wm_output = aux.get('wm_output', None)
            if base_output is not None and wm_output is not None:
                residual = wm_output[..., 0, :] - base_output[..., 0, :]
        if residual is not None:
            residual = residual.detach()
            if residual.ndim > 2:
                residual = residual.reshape(residual.shape[0], -1)
            residual_norm = th.linalg.norm(residual, dim=-1)
            self._ep_wm_residual_norm_sum += residual_norm.to(
                self._ep_wm_residual_norm_sum.device)
            self._ep_wm_residual_count += 1
            self._step_residual_norm.copy_(residual_norm.to(
                self._step_residual_norm.device))
            self._step_residual_active.copy_(
                self._step_residual_norm > 1.0e-4)
        else:
            self._step_residual_norm.zero_()
            self._step_residual_active.zero_()

    def _capture_initial_signature(self):
        first = th.isnan(self._ep_initial_obj_pose[:, 0])
        if not bool(first.any()):
            return
        obj_ids = self.env.scene.cur_ids.long()
        obj_pose = self.env.tensors['root'][obj_ids, :7]
        goal = self.env.task.goal
        goal7 = th.zeros_like(obj_pose)
        width = min(goal.shape[-1], goal7.shape[-1])
        goal7[..., :width] = goal[..., :width]
        self._ep_initial_obj_pose[first] = obj_pose[first]
        self._ep_initial_goal[first] = goal7[first]

    def _current_diag(self, actions: th.Tensor):
        goal = self.env.task.goal
        obj_ids = self.env.scene.cur_ids.long()
        obj_state = self.env.tensors['root'][obj_ids, :]
        body_indices = self.env.robot.ee_body_indices.long()
        hand_state = self.env.tensors['body'][body_indices]

        obj_xy = obj_state[..., :2]
        goal_xy = goal[..., :2]
        goal_vec = goal_xy - obj_xy
        goal_dist = th.linalg.norm(goal_vec, dim=-1)
        hand_obj_dist = th.linalg.norm(hand_state[..., :3] - obj_state[..., :3],
                                       dim=-1)

        if actions is None or actions.numel() == 0:
            action_norm = th.zeros_like(goal_dist)
            action_goal_cos = th.zeros_like(goal_dist)
        else:
            act = actions
            if act.ndim > 2:
                act = act.reshape(act.shape[0], -1)
            act_xy = act[..., :2]
            action_norm = th.linalg.norm(act[..., :min(3, act.shape[-1])],
                                         dim=-1)
            denom = (
                th.linalg.norm(act_xy, dim=-1) *
                th.linalg.norm(goal_vec, dim=-1)
            ).clamp_min(1e-6)
            action_goal_cos = (act_xy * goal_vec).sum(dim=-1) / denom

        object_speed = th.linalg.norm(obj_state[..., 7:10], dim=-1)
        object_ids = self.env.scene.body_ids.long()
        object_contact = self.env.tensors['net_contact'][object_ids]
        lateral_force = th.linalg.norm(object_contact[..., :2], dim=-1)
        contact_radius = (
            self.env.robot.robot_radius + self.env.scene.cur_radii)
        # Net-contact tensors can be sparse at wrapper sampling time. A tight
        # bounding-sphere overlap provides a stable geometric fallback, while
        # the wider shell remains the pre-contact phase.
        geometric_contact = hand_obj_dist < (contact_radius + 0.015)
        contact = (lateral_force > 5.0e-2) | geometric_contact
        near_contact = hand_obj_dist < (contact_radius + 0.08)
        goal_thresh = getattr(self.env.task, 'goal_radius_samples', None)
        if goal_thresh is None:
            goal_thresh = th.full_like(goal_dist, 0.05)
        near_goal = goal_dist <= (1.5 * goal_thresh)

        phase = th.zeros_like(self._prev_phase)
        phase[near_contact] = 1
        phase[contact & (object_speed <= 1.0e-2)] = 2
        phase[contact & (object_speed > 1.0e-2)] = 3
        phase[near_goal] = 4
        return (goal_dist, hand_obj_dist, action_norm, action_goal_cos,
                phase)

    def _update_diag(self, actions: th.Tensor):
        if not self._diag_enabled:
            return None
        try:
            goal_dist, hand_obj_dist, action_norm, action_goal_cos, phase = (
                self._current_diag(actions))
        except Exception as exc:
            self._diag_enabled = False
            if not self._diag_error_logged:
                print(f'CountFixedTrialsSuccess diagnostics disabled: {exc}')
                self._diag_error_logged = True
            return None

        first = th.isnan(self._ep_start_goal_dist)
        self._ep_start_goal_dist[first] = goal_dist[first]
        self._ep_min_goal_dist = th.minimum(self._ep_min_goal_dist, goal_dist)
        self._ep_min_hand_obj_dist = th.minimum(
            self._ep_min_hand_obj_dist, hand_obj_dist)
        self._ep_action_norm_sum += action_norm
        self._ep_action_goal_cos_sum += action_goal_cos
        self._ep_step_count += 1
        goal_delta = th.where(
            th.isnan(self._prev_goal_dist), th.zeros_like(goal_dist),
            self._prev_goal_dist - goal_dist)
        was_contact = (self._prev_phase == 2) | (self._prev_phase == 3)
        is_contact = (phase == 2) | (phase == 3)
        self._ep_lost_contact += (was_contact & ~is_contact).float()
        for phase_id in range(len(self.PHASE_NAMES)):
            mask = phase == phase_id
            self._ep_phase_steps[:, phase_id] += mask.float()
            self._ep_phase_residual_norm_sum[:, phase_id] += (
                mask.float() * self._step_residual_norm)
            self._ep_phase_residual_active[:, phase_id] += (
                mask & self._step_residual_active).float()
            self._ep_phase_goal_delta_sum[:, phase_id] += (
                mask.float() * goal_delta)
        self._prev_goal_dist.copy_(goal_dist)
        self._prev_phase.copy_(phase)
        return goal_dist

    def _record_diag(self, entry: Dict, idx: int, final_goal_dist: th.Tensor,
                     info: Dict, object_name: str, trial_index: int):
        if final_goal_dist is None:
            return
        step_count = float(self._ep_step_count[idx].clamp_min(1).item())
        start_goal = float(self._ep_start_goal_dist[idx].item())
        final_goal = float(final_goal_dist[idx].item())
        min_goal = float(self._ep_min_goal_dist[idx].item())
        min_hand_obj = float(self._ep_min_hand_obj_dist[idx].item())
        mean_action_norm = float(
            (self._ep_action_norm_sum[idx] / step_count).item())
        mean_action_goal_cos = float(
            (self._ep_action_goal_cos_sum[idx] / step_count).item())

        entry['final_goal_dist_sum'] += final_goal
        entry['start_goal_dist_sum'] += start_goal
        entry['min_goal_dist_sum'] += min_goal
        entry['progress_sum'] += start_goal - final_goal
        entry['min_hand_obj_dist_sum'] += min_hand_obj
        entry['mean_action_norm_sum'] += mean_action_norm
        entry['mean_action_goal_cos_sum'] += mean_action_goal_cos
        progress = start_goal - final_goal
        success = float(info['success'][idx].float())
        contact_like = min_hand_obj < 0.08
        moved_away = progress < -0.01
        contact_no_progress = contact_like and progress < 0.01

        entry['contact_like_count'] += float(contact_like)
        entry['no_contact_count'] += float(not contact_like)
        entry['contact_no_progress_count'] += float(contact_no_progress)
        entry['moved_away_count'] += float(moved_away)
        if success > 0.5:
            entry['success_progress_sum'] += progress
            entry['success_final_goal_dist_sum'] += final_goal
        else:
            entry['failure_progress_sum'] += progress
            entry['failure_final_goal_dist_sum'] += final_goal

        gate_count = float(self._ep_wm_gate_count[idx].item())
        if gate_count > 0:
            entry['wm_gate_sum'] += float(
                (self._ep_wm_gate_sum[idx] / gate_count).item())
            entry['wm_gate_episode_count'] += 1.0
        residual_count = float(self._ep_wm_residual_count[idx].item())
        if residual_count > 0:
            entry['wm_residual_norm_sum'] += float(
                (self._ep_wm_residual_norm_sum[idx] / residual_count).item())
            entry['wm_residual_episode_count'] += 1.0
        if 'timeout' in info:
            entry['timeout_count'] += float(info['timeout'][idx].float())
        phase_record = {}
        for phase_id, phase_name in enumerate(self.PHASE_NAMES):
            steps = float(self._ep_phase_steps[idx, phase_id].item())
            residual_sum = float(
                self._ep_phase_residual_norm_sum[idx, phase_id].item())
            active = float(
                self._ep_phase_residual_active[idx, phase_id].item())
            goal_delta = float(
                self._ep_phase_goal_delta_sum[idx, phase_id].item())
            entry['phase_step_count'][phase_name] += steps
            entry['phase_residual_norm_sum'][phase_name] += residual_sum
            entry['phase_residual_active_count'][phase_name] += active
            entry['phase_goal_delta_sum'][phase_name] += goal_delta
            phase_record[phase_name] = {
                'steps': steps,
                'residual_norm_sum': residual_sum,
                'residual_active_steps': active,
                'goal_delta_sum': goal_delta,
            }
        lost_contact = float(self._ep_lost_contact[idx].item())
        entry['lost_contact_count'] += lost_contact
        self.trial_rows.append({
            'object': object_name,
            'trial_index': int(trial_index),
            'success': success,
            'timeout': float(info.get('timeout', th.zeros_like(
                info['success']))[idx].float()),
            'start_goal_dist': start_goal,
            'final_goal_dist': final_goal,
            'progress': progress,
            'initial_object_pose': [float(x) for x in
                                    self._ep_initial_obj_pose[idx].tolist()],
            'initial_goal': [float(x) for x in
                             self._ep_initial_goal[idx].tolist()],
            'lost_contact_count': lost_contact,
            'phases': phase_record,
        })

    def _clear_diag_indices(self, indices: th.Tensor):
        if indices.numel() <= 0:
            return
        self._ep_start_goal_dist[indices] = float('nan')
        self._ep_min_goal_dist[indices] = float('inf')
        self._ep_min_hand_obj_dist[indices] = float('inf')
        self._ep_action_norm_sum[indices] = 0.0
        self._ep_action_goal_cos_sum[indices] = 0.0
        self._ep_step_count[indices] = 0.0
        self._ep_wm_gate_sum[indices] = 0.0
        self._ep_wm_gate_count[indices] = 0.0
        self._ep_wm_residual_norm_sum[indices] = 0.0
        self._ep_wm_residual_count[indices] = 0.0
        self._ep_phase_steps[indices] = 0.0
        self._ep_phase_residual_norm_sum[indices] = 0.0
        self._ep_phase_residual_active[indices] = 0.0
        self._ep_phase_goal_delta_sum[indices] = 0.0
        self._ep_lost_contact[indices] = 0.0
        self._prev_goal_dist[indices] = float('nan')
        self._prev_phase[indices] = -1
        self._step_residual_norm[indices] = 0.0
        self._step_residual_active[indices] = False
        self._ep_initial_obj_pose[indices] = float('nan')
        self._ep_initial_goal[indices] = float('nan')

    def step(self, actions: th.Tensor):
        self._capture_initial_signature()
        obs, rew, done, info = self.env.step(actions)
        final_goal_dist = self._update_diag(actions)
        env_reset_indices = done.view(
            self.env.num_env, 1).all(dim=1).nonzero(as_tuple=False)
        if len(env_reset_indices) > 0:
            for idx in env_reset_indices:
                name = str(self.env.scene.cur_names[idx.item()])
                entry = self._entry(name)
                if entry['reset_count'] >= self.trials_per_object:
                    continue
                entry['reset_count'] += 1
                entry['success_count'] += float(info['success'][idx].float())
                self._record_diag(
                    entry, idx.item(), final_goal_dist, info, name,
                    entry['reset_count'] - 1)
            self._clear_diag_indices(env_reset_indices.view(-1))
        return (obs, rew, done, info)

    def done(self) -> bool:
        if self.trials_per_object <= 0:
            return False
        for name in self.target_names:
            entry = self.counting.get(name)
            if entry is None or entry['reset_count'] < self.trials_per_object:
                return False
        return True

    def save(self):
        ensure_directory(self._save_dir)
        rows = []
        total_success = 0.0
        total_trials = 0

        for name in self.target_names:
            entry = self.counting.get(name, {
                'reset_count': 0,
                'success_count': 0.0,
            })
            trials = int(entry['reset_count'])
            success = float(entry['success_count'])
            success_rate = success / trials if trials > 0 else float('nan')
            denom = max(trials, 1)
            rows.append({
                'object': name,
                'success_count': success,
                'reset_count': trials,
                'success_rate': success_rate,
                'timeout_count': float(entry.get('timeout_count', 0.0)),
                'timeout_rate': (
                    float(entry.get('timeout_count', 0.0)) / denom),
                'mean_final_goal_dist': (
                    float(entry.get('final_goal_dist_sum', 0.0)) / denom),
                'mean_start_goal_dist': (
                    float(entry.get('start_goal_dist_sum', 0.0)) / denom),
                'mean_min_goal_dist': (
                    float(entry.get('min_goal_dist_sum', 0.0)) / denom),
                'mean_progress': (
                    float(entry.get('progress_sum', 0.0)) / denom),
                'mean_min_hand_obj_dist': (
                    float(entry.get('min_hand_obj_dist_sum', 0.0)) / denom),
                'contact_like_rate': (
                    float(entry.get('contact_like_count', 0.0)) / denom),
                'no_contact_rate': (
                    float(entry.get('no_contact_count', 0.0)) / denom),
                'contact_no_progress_rate': (
                    float(entry.get('contact_no_progress_count', 0.0)) / denom),
                'moved_away_rate': (
                    float(entry.get('moved_away_count', 0.0)) / denom),
                'mean_action_norm': (
                    float(entry.get('mean_action_norm_sum', 0.0)) / denom),
                'mean_action_goal_cos': (
                    float(entry.get('mean_action_goal_cos_sum', 0.0)) / denom),
                'mean_success_progress': (
                    float(entry.get('success_progress_sum', 0.0)) /
                    max(success, 1.0)),
                'mean_failure_progress': (
                    float(entry.get('failure_progress_sum', 0.0)) /
                    max(trials - success, 1.0)),
                'mean_success_final_goal_dist': (
                    float(entry.get('success_final_goal_dist_sum', 0.0)) /
                    max(success, 1.0)),
                'mean_failure_final_goal_dist': (
                    float(entry.get('failure_final_goal_dist_sum', 0.0)) /
                    max(trials - success, 1.0)),
                'mean_wm_gate': (
                    float(entry.get('wm_gate_sum', 0.0)) /
                    max(float(entry.get('wm_gate_episode_count', 0.0)), 1.0)),
                'mean_wm_residual_norm': (
                    float(entry.get('wm_residual_norm_sum', 0.0)) /
                    max(float(entry.get('wm_residual_episode_count', 0.0)),
                        1.0)),
            })
            total_success += success
            total_trials += trials

        summary = {
            'trials_per_object': self.trials_per_object,
            'num_objects': len(self.target_names),
            'total_success': total_success,
            'total_trials': total_trials,
            'micro_success_rate': (
                total_success / total_trials if total_trials > 0 else None),
            'complete': self.done(),
        }

        with open(str(self._save_dir / 'fixed_trials_result.pkl'), 'wb') as fp:
            pickle.dump({
                'summary': summary,
                'rows': rows,
                'counting': self.counting,
            }, fp)
        with open(str(self._save_dir / 'fixed_trials_summary.json'), 'w') as fp:
            json.dump(summary, fp, indent=2)
        phase_by_object = {}
        phase_global = {
            name: {
                'steps': 0.0, 'residual_norm_sum': 0.0,
                'residual_active_steps': 0.0, 'goal_delta_sum': 0.0,
            } for name in self.PHASE_NAMES}
        for object_name in self.target_names:
            entry = self.counting.get(object_name, {})
            object_phases = {}
            for phase_name in self.PHASE_NAMES:
                values = {
                    'steps': float(entry.get('phase_step_count', {}).get(
                        phase_name, 0.0)),
                    'residual_norm_sum': float(entry.get(
                        'phase_residual_norm_sum', {}).get(phase_name, 0.0)),
                    'residual_active_steps': float(entry.get(
                        'phase_residual_active_count', {}).get(
                            phase_name, 0.0)),
                    'goal_delta_sum': float(entry.get(
                        'phase_goal_delta_sum', {}).get(phase_name, 0.0)),
                }
                steps = max(values['steps'], 1.0)
                values['mean_residual_norm'] = (
                    values['residual_norm_sum'] / steps)
                values['residual_active_rate'] = (
                    values['residual_active_steps'] / steps)
                values['mean_goal_delta'] = values['goal_delta_sum'] / steps
                object_phases[phase_name] = values
                for key in phase_global[phase_name]:
                    phase_global[phase_name][key] += values[key]
            phase_by_object[object_name] = {
                'phases': object_phases,
                'lost_contact_count': float(entry.get(
                    'lost_contact_count', 0.0)),
            }
        for phase_name, values in phase_global.items():
            steps = max(values['steps'], 1.0)
            values['mean_residual_norm'] = values['residual_norm_sum'] / steps
            values['residual_active_rate'] = (
                values['residual_active_steps'] / steps)
            values['mean_goal_delta'] = values['goal_delta_sum'] / steps
        with open(str(self._save_dir / 'fixed_trials_phase.json'), 'w') as fp:
            json.dump({
                'phase_definitions': {
                    'near_goal': 'xy goal distance <= 1.5 * success radius',
                    'contact_moving': 'contact proxy (lateral force > 0.05 or tight sphere overlap +0.015m) and object speed > 0.01',
                    'contact_stalled': 'contact proxy (lateral force > 0.05 or tight sphere overlap +0.015m) and object speed <= 0.01',
                    'pre_contact': 'within radius margin 0.08 without contact',
                    'free_space': 'all remaining states',
                },
                'global': phase_global,
                'per_object': phase_by_object,
            }, fp, indent=2)
        with open(str(self._save_dir / 'fixed_trials_trials.jsonl'), 'w') as fp:
            for row in self.trial_rows:
                fp.write(json.dumps(row, sort_keys=True) + '\n')
        with open(str(self._save_dir / 'fixed_trials_result.csv'), 'w') as fp:
            writer = csv.DictWriter(
                fp,
                fieldnames=[
                    'object', 'success_count', 'reset_count', 'success_rate',
                    'timeout_count', 'timeout_rate',
                    'mean_final_goal_dist', 'mean_start_goal_dist',
                    'mean_min_goal_dist', 'mean_progress',
                    'mean_min_hand_obj_dist', 'contact_like_rate',
                    'no_contact_rate', 'contact_no_progress_rate',
                    'moved_away_rate', 'mean_action_norm',
                    'mean_action_goal_cos', 'mean_success_progress',
                    'mean_failure_progress',
                    'mean_success_final_goal_dist',
                    'mean_failure_final_goal_dist', 'mean_wm_gate',
                    'mean_wm_residual_norm'])
            writer.writeheader()
            writer.writerows(rows)


class Phase2Training(WrapperEnv):
    """
    In phase 2 we finetune the policy so that it can be transfered to the real robot
    1. Annealing reducing joint limit
    2. Annealing reducing residual scale (action)
    3. Environment randomization (friction, density)
    4. Torque randomize (add gaussian noise to the control signal)
    """

    @dataclass
    class Config(ConfigBase):
        # condition to update
        # 1. # reset after reset >= minimum # reset
        # 2. Success rate after update >= threshold
        min_reset_to_update: int = 4096
        update_threshold: float = 0.8

        adaptive_dof_pos_limit: bool = True
        # number of steps for joint position range annealing
        pos_num_steps: int = 10
        # FIXME:
        # should be `target_pos_threshold`
        pos_maximum_reduction: float = 0.03
        # FIXME:
        # should be `initial_pos_threshold`
        start_dof_pos_offset: float = 0.0

        adaptive_residual_scale: bool = True
        initial_residual_scale: Tuple[float, float] = (0.06, 0.1)
        target_residual_scale: Tuple[float, float] = (0.02, 0.03)
        # number of steps for residual scale annealing
        res_num_steps: int = 10
        epsilon: float = 1e-4
        log_period: int = 1024

    def __init__(self, cfg: Config, env: EnvIface):
        super().__init__(env)
        self.cfg = cfg

        # To prevent too frequent update
        self.nepi_from_prev_check = 0
        self.nsuc_from_prev_update = 0

        self.dof_pos_thresh = cfg.start_dof_pos_offset

        # delta of max_p = max_p - delta
        self.pos_reduction_step = cfg.pos_maximum_reduction / cfg.pos_num_steps

        target_pos_scale = (
            cfg.target_residual_scale[0] / cfg.initial_residual_scale[0])
        target_ori_scale = (
            cfg.target_residual_scale[1] / cfg.initial_residual_scale[1])

        # r of a_t+1 = r * a_t
        self.pos_multiplier = np.power(target_pos_scale,
                                       1. / cfg.res_num_steps)
        self.ori_multiplier = np.power(target_ori_scale,
                                       1. / cfg.res_num_steps)

        # TODO: It shoud also handle the case where xyz rpy has differne
        # scale
        self.target_pos_scale = target_pos_scale
        self.target_ori_scale = target_ori_scale
        if cfg.adaptive_residual_scale:
            self.pos_scale = 1.0
            self.ori_scale = 1.0
            self.scale = th.ones(env.action_space.shape[0],
                                 dtype=th.float,
                                 device=env.device)
        else:
            self.pos_scale = target_pos_scale
            self.ori_scale = target_ori_scale
            self.scale = th.ones(env.action_space.shape[0],
                                 dtype=th.float,
                                 device=env.device)
            self.scale[:3] = self.pos_scale
            self.scale[3:6] = self.ori_scale

        # Remember dof limits of robots
        dof_limits = env.robot.dof_limits
        self.dof_center = th.as_tensor(
            0.5 * (dof_limits[0] + dof_limits[1]),
            dtype=th.float,
            device=env.device)
        # scale=diameter, not radius
        self.dof_scale = th.as_tensor(
            (dof_limits[1] - dof_limits[0]),
            dtype=th.float,
            device=env.device)

        print("franka lower limit: " + str(dof_limits[0]))
        print("franka upper limit: " + str(dof_limits[1]))
        print("dof_scale: " + str(self.dof_scale))
        self.__step = 0

    def checkpoint_state_dict(self):
        return {
            'version': 1,
            'step': int(self.__step),
            'nepi_from_prev_check': int(self.nepi_from_prev_check),
            'nsuc_from_prev_update': int(self.nsuc_from_prev_update),
            'dof_pos_thresh': float(self.dof_pos_thresh),
            'pos_scale': float(self.pos_scale),
            'ori_scale': float(self.ori_scale),
            'target_pos_scale': float(self.target_pos_scale),
            'target_ori_scale': float(self.target_ori_scale),
        }

    def load_checkpoint_state_dict(self, state):
        if int(state.get('version', -1)) != 1:
            raise ValueError('unsupported Phase2Training checkpoint version')
        for key, expected in (
                ('target_pos_scale', self.target_pos_scale),
                ('target_ori_scale', self.target_ori_scale)):
            if not np.isclose(float(state[key]), float(expected)):
                raise ValueError(
                    f'Phase2Training config mismatch for {key}: '
                    f'{state[key]} != {expected}')

        self.__step = int(state['step'])
        self.nepi_from_prev_check = int(state['nepi_from_prev_check'])
        self.nsuc_from_prev_update = int(state['nsuc_from_prev_update'])
        self.dof_pos_thresh = float(state['dof_pos_thresh'])
        self.pos_scale = float(state['pos_scale'])
        self.ori_scale = float(state['ori_scale'])
        self.scale.fill_(1.0)
        self.scale[:3] = self.pos_scale
        self.scale[3:6] = self.ori_scale

    def step(self, actions: th.Tensor):
        cfg = self.cfg

        # NOTE: we apply a reduced action here.
        # self.scale starts from 1.0
        # and then anneals to (0.02/0.06, 0.03/0.1).
        obs, rew, done, info = self.env.step(
            actions * self.scale
        )

        # Terminate the episode if it exceed new pos limit.
        # TODO  May also need to adjust terminal reward,
        # if enabled.
        if cfg.adaptive_dof_pos_limit and self.dof_pos_thresh > 0.:
            dof_oob = self._check_dof_limit_violation()
            done = th.logical_or(done, dof_oob)
            self.env.buffers['done'][...] = done

        # Bookkeeping for check intervals and success rates
        self.nepi_from_prev_check += done.sum().item()
        if 'success' in info:
            self.nsuc_from_prev_update += info['success'].sum().item()
        cur_sr = (
            0.0 if (self.nepi_from_prev_check) <= 0
            else (self.nsuc_from_prev_update
                  / self.nepi_from_prev_check)
        )

        # Additional logging for tracking phase2
        if (self.__step % cfg.log_period == 0):
            if self._writer is not None:
                writer = self._writer
                step = self.__step
                writer.add_scalar('debug/sr_in_phase2',
                                  cur_sr, global_step=step)
                writer.add_scalar('debug/epi_count_phase2',
                                  self.nepi_from_prev_check, global_step=step)
                writer.add_scalar('debug/suc_count_phase2',
                                  self.nsuc_from_prev_update, global_step=step)
                writer.add_scalar(
                    'debug/pos_scale',
                    self.pos_scale * cfg.initial_residual_scale[0],
                    global_step=step)
                writer.add_scalar(
                    'debug/ori_scale',
                    self.ori_scale * cfg.initial_residual_scale[1],
                    global_step=step)
                writer.add_scalar('debug/joint_pos_reduce',
                                  self.dof_pos_thresh, global_step=step)

        # Step annealing when the number of success > threshold
        # and the success rate from previous update or recent 1000 episode >
        # threshold
        check: bool = (self.nepi_from_prev_check >= cfg.min_reset_to_update)
        if check:
            if cur_sr >= cfg.update_threshold:
                if cfg.adaptive_dof_pos_limit:
                    self._update_pos_thresh()
                if cfg.adaptive_residual_scale:
                    self._update_residual_scale()
            self.nepi_from_prev_check = 0
            self.nsuc_from_prev_update = 0

        self.__step += 1

        return (obs, rew, done, info)

    def _update_pos_thresh(self):
        if self.dof_pos_thresh >= self.cfg.pos_maximum_reduction:
            return False
        self.dof_pos_thresh += self.pos_reduction_step
        if self.dof_pos_thresh >= self.cfg.pos_maximum_reduction:
            self.dof_pos_thresh = self.cfg.pos_maximum_reduction
            print("dof_pos_limit is maximum " + str(self.dof_pos_thresh))
        else:
            print("dof_pos_limit is increased to " + str(self.dof_pos_thresh))
        return True

    def _update_residual_scale(self):
        """
            update_residual_scale by a_t+1 = r * a_t
        """

        self.pos_scale = (
            self.pos_scale * self.pos_multiplier)
        self.ori_scale = (
            self.ori_scale * self.ori_multiplier)

        # Ensure >= target scale
        if self.pos_scale <= self.target_pos_scale:
            self.pos_scale = self.target_pos_scale
        else:
            print(F"Residual pos. scale is reduced to {self.pos_scale}")

        if self.ori_scale <= self.target_ori_scale:
            self.ori_scale = self.target_ori_scale
        else:
            print(F"Residual ori. scale is reduced to {self.ori_scale}")

        # Update tensor
        # FIXME: NOT usable across all control modes
        self.scale[:3] = self.pos_scale
        self.scale[3:6] = self.ori_scale

        return True

    def _check_dof_limit_violation(self):
        """
            check the environment that exeeds the position limit
            threshold>cur_pos/max_pos or cur_pos/max_pos>=1-threshold
        """

        thresh = self.dof_pos_thresh
        dof_delta = (self.env.tensors['dof'][..., 0] - self.dof_center)
        # Ideally within range (-0.5 ~ 0.5) if within bounds
        normalized_dof_delta = (dof_delta / self.dof_scale)
        dof_pos_exceeded = th.abs(normalized_dof_delta) > (0.5 - thresh)
        return dof_pos_exceeded.any(dim=-1)


class ScenarioTest(WrapperEnv):
    def __init__(self, env: EnvIface):
        super().__init__(env)
        try:
            convert = partial(_array_from_map,
                              env.scene.keys,
                              dtype=th.float,
                              device=env.device)
            self.num_tests = convert({k: env.scene.meta.num_scenarios(k)
                                      for k in env.scene.keys})
        except BaseException:
            self.num_tests = 10 * th.ones(env.num_env, device=env.device)
        self.done_count = th.zeros_like(self.num_tests)
        self.success_count = th.zeros_like(self.num_tests)
        self.end_count = 0
        self.success_rates = {}

    def step(self, actions: th.Tensor):
        obs, rew, done, info = self.env.step(actions)
        self.done_count[done] += 1
        if 'success' in info:
            self.success_count += info['success']

        scenario_ended = (
            self.num_tests == self.done_count).nonzero().flatten().cpu().numpy()
        num_updated = 0
        for id in scenario_ended:
            if id not in self.success_rates:
                num_updated += 1
                self.success_rates[id] = \
                    (self.success_count[id] / self.done_count[id]).item()
        if num_updated > 0:
            self.end_count += num_updated
            print(f"{self.end_count} / {self.env.num_env} finished")
            if self.end_count == self.env.num_env:
                for id, v in self.success_rates.items():
                    obj_key = self.env.scene.keys[id]
                    print(f'scenario/env_{id}_{obj_key}={v}')

        # Step annealing when the number of success > threshold
        # and the success rate from previous update or recent 1000 episode >
        # threshold
        return (obs, rew, done, info)


def test_icp_emb_obs():
    class DummyEnv():
        def __init__(self):
            self.device = 'cuda:1'
            self.observation_space = spaces.Dict()

        def reset(self):
            obs = {
                'cloud': th.zeros((1, 512, 3), dtype=th.float32,
                                  device=self.device),
                'hand_state': th.zeros((1, 7), dtype=th.float32,
                                       device=self.device)
            }
            return obs
    env = DummyEnv()
    wrap = ICPEmbObs(env, ICPEmbObs.Config())
    obs = wrap.reset()
    print(wrap.observation_space['icp_emb'])
    print(obs['icp_emb'].shape)
    # Box(-inf, inf, (17, 128), float32)
    # torch.Size([1, 17, 128])


def main():
    test_icp_emb_obs()


if __name__ == '__main__':
    main()
