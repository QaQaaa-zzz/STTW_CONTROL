"""Synchronized, bounded MJX physics throughput measurement (not a PPO score)."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import time

import jax
import jax.numpy as jp
from .env import RecoveryEnv, load_config
from .runtime import configure_compilation_cache


def benchmark(task_path, output, *, num_envs=1024, steps=32, repeats=3, seed=61):
    for value in (num_envs, steps, repeats):
        if not isinstance(value, int) or value < 1:
            raise ValueError('environment count, steps and repeats must be positive integers')
    cache_dir=configure_compilation_cache()
    if not any(d.platform == 'gpu' for d in jax.devices()):
        raise RuntimeError('GPU benchmark requires a GPU JAX backend')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    task = load_config(task_path)
    declaration = {
        'role': 'engineering physics throughput; no policy or optimizer, not recovery evidence',
        'jax_compilation_cache_dir': cache_dir,
        'task': asdict(task), 'num_envs': num_envs, 'steps': steps,
        'repeats': repeats, 'seed': seed,
        'budget_control_steps': num_envs * steps * (repeats + 1),
        'sampling': 'each repetition starts from the same reset batch; no autoreset',
        'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted(Path(__file__).parent.glob('*.py'))},
    }
    (output / 'declaration.json').write_text(json.dumps(declaration, indent=2) + '\n')
    env = RecoveryEnv(task, backend='mjx')
    if steps >= env.horizon:
        raise ValueError('benchmark window must be shorter than the episode horizon')
    begin = time.monotonic()
    state = jax.jit(jax.vmap(env.reset))(jax.random.split(jax.random.PRNGKey(seed), num_envs))
    jax.block_until_ready(state)
    reset_seconds = time.monotonic() - begin
    step = jax.vmap(env.step)

    @jax.jit
    def rollout(s):
        def advance(s, _):
            return step(s, jp.zeros((num_envs, 2))), None
        return jax.lax.scan(advance, s, None, length=steps)[0]

    begin = time.monotonic()
    result = rollout(state)
    jax.block_until_ready(result)
    compile_seconds = time.monotonic() - begin
    seconds = []
    for _ in range(repeats):
        begin = time.monotonic()
        result = rollout(state)
        jax.block_until_ready(result)
        seconds.append(time.monotonic() - begin)
    finite = bool(jp.all(jp.isfinite(result.data.qpos)) & jp.all(jp.isfinite(result.data.qvel)))
    done = int(jp.sum(result.done))
    report = {
        'num_envs': num_envs, 'steps': steps,
        'reset_compile_seconds': reset_seconds,
        'compile_first_rollout_seconds': compile_seconds,
        'seconds': seconds,
        'scheduled_control_steps_per_second': num_envs * steps / (sum(seconds) / repeats),
        'finite': finite, 'done': done,
        'valid_throughput_sample': finite and done == 0,
        'device': str(jax.devices()[0]), 'memory_stats': jax.devices()[0].memory_stats(),
        'jax_version': jax.__version__,
    }
    (output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    if not report['valid_throughput_sample']:
        raise RuntimeError('nonfinite or terminated trajectories invalidate the throughput comparison')
    return report
