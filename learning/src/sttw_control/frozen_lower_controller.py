"""Frozen legacy Actor transferred to a causal governed-command path.

This adapter preserves the Actor's input units, ordering, history and return
state. It does not reproduce the old fixed-bend task: only published governed
commands construct its world-fixed path. Beyond the available endpoint a
forward tangent ray is used, explicitly flagged, without predicting a turn.
The caller owns physics, ESO advancement, total residual limits and V3 reward.
"""
import hashlib
import json
from pathlib import Path

from flax import struct
import jax.numpy as jp

from .network import load_policy
from .observation import (ObservationConfig, observation_fields, initial_history,
                          advance_history, make_frame)
from .teleop_reference import integrate
from .tracking_reward import (TrackingConfig, initial_return, return_observation,
                              transition)


@struct.dataclass
class FrozenLowerState:
    history: object
    return_state: object
    reference_pose: object
    path_points: object  # x, y, unwrapped yaw, curvature
    valid_count: object
    overflow: object


def path_features(state, pose):
    """Nearest valid historical segment, or endpoint's forward tangent ray."""
    points = state.path_points
    a, b = points[:-1], points[1:]
    v = b[:, :2] - a[:, :2]
    length2 = jp.sum(v*v, axis=-1)
    u = jp.clip(jp.sum((pose[:2]-a[:, :2])*v, axis=-1)
                / jp.maximum(length2, 1e-12), 0., 1.)
    foot = a[:, :2]+u[:, None]*v
    valid = (jp.arange(len(a)) < state.valid_count-1) & (length2 > 1e-12)
    distances = jp.where(valid, jp.sum((pose[:2]-foot)**2, axis=-1), jp.inf)
    index = jp.argmin(distances)
    yaw = a[index, 2]+u[index]*(b[index, 2]-a[index, 2])
    curvature = a[index, 3]+u[index]*(b[index, 3]-a[index, 3])
    end = points[jp.maximum(state.valid_count-1, 0)]
    tangent = jp.array([jp.cos(end[2]), jp.sin(end[2])])
    forward = jp.maximum(jp.dot(pose[:2]-end[:2], tangent), 0.)
    ray_foot = end[:2]+forward*tangent
    ray_distance = jp.sum((pose[:2]-ray_foot)**2)
    extension = ray_distance < distances[index]
    chosen_foot = jp.where(extension, ray_foot, foot[index])
    yaw = jp.where(extension, end[2], yaw)
    curvature = jp.where(extension, end[3], curvature)
    delta = pose[:2]-chosen_foot
    right = jp.sin(yaw)*delta[0]-jp.cos(yaw)*delta[1]
    heading = jp.arctan2(jp.sin(pose[2]-yaw), jp.cos(pose[2]-yaw))
    return jp.array([right, heading, curvature]), extension


class FrozenLowerController:
    """Config requires source_checkpoint and source_declaration.

    Optional export/sidecar paths must resolve to files in that checkpoint.
    Optional sidecar_sha256/payload_sha256 add explicit manifest assertions.
    alpha is fixed to 1; capacity defaults to the 3200-tick V3 episode plus reset.
    """
    def __init__(self, config):
        self.config = dict(config)
        checkpoint = Path(config['source_checkpoint']).resolve()
        export = Path(config.get('export', checkpoint/'actor.msgpack')).resolve()
        sidecar = Path(config.get('sidecar', checkpoint/'identity.json')).resolve()
        if export != checkpoint/'actor.msgpack' or sidecar != checkpoint/'identity.json':
            raise ValueError('frozen lower export/sidecar must belong to source_checkpoint')
        declaration = json.loads(Path(config['source_declaration']).read_text())
        metadata = json.loads(sidecar.read_text())
        sidecar_hash = hashlib.sha256(sidecar.read_bytes()).hexdigest()
        payload_hash = hashlib.sha256(export.read_bytes()).hexdigest()
        if sidecar_hash != declaration['policy']['checkpoint_sidecar_sha256']:
            raise ValueError('frozen lower evaluation/checkpoint sidecar mismatch')
        for key, actual in [('sidecar_sha256', sidecar_hash), ('payload_sha256', payload_hash)]:
            if key in config and config[key] != actual:
                raise ValueError('frozen lower '+key+' mismatch')
        if payload_hash != metadata['payload_sha256']:
            raise ValueError('frozen lower Actor payload mismatch')
        identity = metadata['identity']
        if any(declaration['policy'].get(k) != v for k, v in identity.items()):
            raise ValueError('frozen lower declaration policy identity mismatch')
        model_hash = hashlib.sha256(json.dumps(declaration['model'], sort_keys=True,
                                               allow_nan=False).encode()).hexdigest()
        if model_hash != identity['model_sha256']:
            raise ValueError('frozen lower source physical identity mismatch')
        if declaration.get('priority_override') != 1. or config.get('alpha', 1.) != 1.:
            raise ValueError('frozen lower requires source evaluation alpha 1')
        source = declaration['config']
        if source.get('learning_roll_reference') is not None:
            raise ValueError('frozen lower requires dynamic roll reference')
        self.observation_config = ObservationConfig(**source['observation'])
        if (self.observation_config.history_steps != 10
                or list(observation_fields(self.observation_config)) != metadata['fields']
                or len(metadata['fields']) != 27
                or metadata['hidden_sizes'] != [256, 128]
                or metadata.get('activation') != 'elu'):
            raise ValueError('unsupported frozen lower observation/network contract')
        self.policy = load_policy(checkpoint, expected=identity)
        self.tracking = TrackingConfig(**source['tracking'])
        self.dt = source['controller']['dt']
        self.wheelbase = source['controller']['wheelbase']
        self.caster = source['controller']['caster']
        if self.dt != .005:
            raise ValueError('frozen lower requires 200 Hz')
        self.capacity = int(config.get('path_capacity', 3201))
        if self.capacity < 2:
            raise ValueError('path_capacity must be at least 2')
        self.source = source
        self.identity = identity
        self.provenance = dict(checkpoint=str(checkpoint), sidecar_sha256=sidecar_hash,
                               payload_sha256=payload_hash, source_model=declaration['model'],
                               transfer='causal_governed_history_with_endpoint_tangent_ray')

    def initial(self, pose):
        points = jp.zeros((self.capacity, 4), dtype=pose.dtype)
        points = points.at[0, :3].set(pose)
        return FrozenLowerState(initial_history(self.observation_config), initial_return(),
                                pose, points, jp.int32(1), jp.bool_(False))

    def prepare(self, state, measurement, pose, governed, controller_output, previous_final):
        # At reset no segment exists; current published command supplies its
        # endpoint curvature. Historical points are immutable after publication.
        curvature = jp.cos(self.caster)*jp.tan(governed[1])/self.wheelbase
        points = state.path_points.at[0, 3].set(jp.where(
            state.valid_count == 1, curvature, state.path_points[0, 3]))
        state = state.replace(path_points=points)
        path, extension = path_features(state, pose)
        command = jp.array([governed[1], governed[0]])
        frame = make_frame(measurement, command, controller_output.reference_roll,
                           controller_output.steer_rate, previous_final,
                           controller_output.disturbance)
        frame = jp.concatenate([frame, path, jp.ones(1),
                                return_observation(state.return_state, self.tracking)])
        history, obs = advance_history(state.history, frame, self.observation_config)
        action = self.policy(obs)
        finite = jp.all(jp.isfinite(obs)) & jp.all(jp.isfinite(action)) & ~state.overflow
        return state.replace(history=history), action, obs, dict(
            finite=finite, path_endpoint_extension=extension, path_features=path,
            path_overflow=state.overflow)

    def after_step(self, state, governed, pose, measurement, true_speed, action, failed, tick):
        reference = integrate(state.reference_pose, governed, self.dt,
                              self.wheelbase, self.caster)
        curvature = jp.cos(self.caster)*jp.tan(governed[1])/self.wheelbase
        index = jp.minimum(state.valid_count, self.capacity-1)
        overflow = state.overflow | (state.valid_count >= self.capacity)
        points = state.path_points.at[index].set(jp.concatenate([reference, curvature[None]]))
        # Preserve all published points on overflow; caller observes a fault.
        points = jp.where(overflow, state.path_points, points)
        updated = state.replace(reference_pose=reference, path_points=points,
                                valid_count=jp.minimum(state.valid_count+1, self.capacity),
                                overflow=overflow)
        path, _ = path_features(updated, pose)
        returned, _ = transition(state.return_state, roll=measurement[0],
            roll_rate=measurement[1], speed_error=true_speed-governed[0],
            lateral_error=path[0], heading_error=path[1], action=action, alpha=1.,
            dt=self.dt, alive_rate=self.source['alive_reward_rate'],
            failure_penalty=self.source['failure_penalty'], failed=failed | overflow,
            enabled=tick*self.dt >= self.tracking.start_seconds, config=self.tracking)
        return updated.replace(return_state=returned)
