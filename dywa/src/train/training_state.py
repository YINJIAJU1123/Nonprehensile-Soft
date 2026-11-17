#!/usr/bin/env python3
"""Serialization helpers for resumable PPO training checkpoints."""

import random
from pathlib import Path

import numpy as np
import torch as th


def _load_trusted_checkpoint(path):
    """Load a locally produced training checkpoint across PyTorch versions."""
    try:
        return th.load(str(path), map_location='cpu', weights_only=False)
    except TypeError:
        # PyTorch releases before ``weights_only`` was introduced.
        return th.load(str(path), map_location='cpu')


def capture_rng_state():
    state = {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch_cpu': th.get_rng_state(),
    }
    if th.cuda.is_available():
        state['torch_cuda'] = th.cuda.get_rng_state_all()
    return state


def restore_rng_state(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    th.set_rng_state(state['torch_cpu'])
    if th.cuda.is_available() and 'torch_cuda' in state:
        th.cuda.set_rng_state_all(state['torch_cuda'])


def save_training_checkpoint(agent, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {'self': agent.state_dict()}
    if hasattr(agent, 'optimizer'):
        checkpoint_step = None
        if path.stem.startswith('step-'):
            checkpoint_step = int(path.stem.split('-', maxsplit=1)[1])
        checkpoint.update({
            'optimizer': agent.optimizer.state_dict(),
            'scheduler': agent.scheduler.state_dict(),
            'training_state': {
                'optim_step': int(agent.optim_step),
                'checkpoint_step': checkpoint_step,
            },
            'rng_state': capture_rng_state(),
        })
        if agent.scaler is not None:
            checkpoint['scaler'] = agent.scaler.state_dict()
    th.save(checkpoint, str(path))


def load_training_checkpoint(agent, path, strict=False,
                             load_training_state=False):
    checkpoint = _load_trusted_checkpoint(path)
    if 'self' not in checkpoint:
        raise KeyError(f'checkpoint has no model state: {path}')
    agent.load_state_dict(checkpoint['self'], strict=strict)

    if not load_training_state:
        return checkpoint
    required = {'optimizer', 'scheduler', 'training_state', 'rng_state'}
    missing = sorted(required.difference(checkpoint))
    if missing:
        raise RuntimeError(
            f'checkpoint cannot resume training; missing {missing}: {path}')
    if not hasattr(agent, 'optimizer'):
        raise RuntimeError('agent was constructed without training state')

    agent.optimizer.load_state_dict(checkpoint['optimizer'])
    agent.scheduler.load_state_dict(checkpoint['scheduler'])
    agent.optim_step = int(checkpoint['training_state']['optim_step'])
    if agent.scaler is not None:
        scaler_state = checkpoint.get('scaler')
        if scaler_state is None:
            raise RuntimeError('AMP resume checkpoint has no scaler state')
        agent.scaler.load_state_dict(scaler_state)
    restore_rng_state(checkpoint['rng_state'])
    return checkpoint
