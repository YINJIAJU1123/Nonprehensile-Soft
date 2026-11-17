"""Student-only deterministic rollout, shared by the two baseline policies."""
import torch as th


def run_fixed_trials(env, student, max_steps, use_amp=False):
    obs = env.reset()
    student.reset_state(th.ones(env.num_env, dtype=th.bool, device=env.device))
    with th.inference_mode(), th.cuda.amp.autocast(enabled=use_amp):
        for step in range(max_steps):
            output, _ = student.get_output(obs)
            if output.ndim != 3 or output.shape[1] != 2:
                raise ValueError('Expected policy output (batch, 2, action_dim)')
            if not bool(th.isfinite(output).all()):
                raise FloatingPointError('Non-finite student output')
            obs, _, done, _ = env.step(output[:, 0, :])
            student.reset_state(done)
            if env.done():
                return step + 1
    raise RuntimeError('Evaluation reached max_steps before completing all trials')


def load_policy_state(student, saved):
    """Strict parameters; temporal buffers depend on runtime batch size."""
    runtime = student.state_dict()
    ephemeral = {'history_tokenizer.history', 'aggregator.aggregator.memory'}
    if set(saved) - ephemeral != set(runtime) - ephemeral:
        raise ValueError('Policy checkpoint keys do not match architecture')
    for key in set(runtime) - ephemeral:
        if saved[key].shape != runtime[key].shape:
            raise ValueError('Policy checkpoint shape mismatch: ' + key)
        if not bool(th.isfinite(saved[key]).all()):
            raise FloatingPointError('Non-finite checkpoint tensor: ' + key)
        runtime[key] = saved[key]
    student.load_state_dict(runtime, strict=True)
