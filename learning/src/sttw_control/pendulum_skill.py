"""Frozen pendulum-bicycle skill adapter; uses the original JIT runtime.

Requires the read-only DVGC/JIT/src reference on PYTHONPATH. The caller owns
the physical state and observation FIFO; loading or changing the policy does
not reset either. This is not an STTW-to-jump state converter.
"""
from pathlib import Path
import json
import hashlib
from types import SimpleNamespace

import jax
from jax import numpy as jp


class PendulumSkill:
    def __init__(self, registry: Path, skill_id: str):
        from jit_dvgc.checkpoint import CheckpointIdentity, load_checkpoint
        from jit_dvgc.config import load_config
        from jit_dvgc.constants import ACTOR_OBSERVATION_SIZE, PRIVILEGED_OBSERVATION_SIZE
        from jit_dvgc.model import load_host_model
        from jit_dvgc.ppo import make_checkpoint_policy

        self.record = json.loads(Path(registry).read_text())["records"][skill_id]["policy"]
        raw = Path(self.record["formal_config"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != self.record["formal_config_file_sha256"]:
            raise ValueError("Frozen skill configuration changed")
        inputs = json.loads(raw)["inputs"]
        config = load_config(Path(inputs["up_config_path"]), runtime_only=True)
        if config.config_sha256 != inputs["up_config_sha256"]:
            raise ValueError("Frozen actuator configuration changed")
        self.bundle = load_host_model(config)
        identity = CheckpointIdentity(
            self.record["formal_config_sha256"], self.bundle.xml_sha256,
            tuple(self.record["actor_frame_fields"]), tuple(self.record["actor_task_fields"]),
            tuple(self.record["action_order"]),
        )
        payload = load_checkpoint(
            Path(self.record["checkpoint"]), expected=identity
        )
        # Inference only needs dimensions, not the training reset pool or PPO budget.
        dimensions = SimpleNamespace(actor_observation_size=ACTOR_OBSERVATION_SIZE,
            privileged_observation_size=PRIVILEGED_OBSERVATION_SIZE, action_size=4)
        self.policy = jax.jit(make_checkpoint_policy(dimensions, payload, deterministic=True))

    def infer(self, actor_observation, knee_position):
        """Return normalized action and native actuator targets, without stepping.

        Observation is the original 76-D FIFO plus jump signal. Targets are
        steering rad, rear-wheel rad/s, hip rad, knee rad, in that order.
        """
        from jit_dvgc.action_mapping import map_action

        observation = jp.asarray(actor_observation, dtype=jp.float32)
        if observation.shape != (76,):
            raise ValueError("Expected the original 76-D skill observation")
        action, _ = self.policy({"state": observation}, jax.random.PRNGKey(0))
        ctrl = map_action(action, jp.asarray(knee_position), self.bundle.action_mapping)
        return action, ctrl
