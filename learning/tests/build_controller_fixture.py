"""Create cross-environment controller evidence from the unchanged JAX source."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import jax
import jax.numpy as jp
import numpy as np
import mujoco
from sttw_control.isaac_asset import export_isaac_asset
from sttw_control.model import load_model
from sttw_control.controller import ControllerConfig, initial_controller, controller_step
import sttw_control.controller as source

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=False)
jax.config.update('jax_enable_x64', True)
rng = np.random.default_rng(20261104)
measurements = rng.normal(size=(1024, 8, 6)) * np.array([1., .1, .2, .05, .1, .15])
measurements[:, :, 0] += np.linspace(-1., 4., 8)
enable = np.broadcast_to(np.arange(1024)[:, None] > np.arange(8)[None, :]*100, (1024, 8))
reset = np.zeros((1024, 8), dtype=bool)
reset[512, [0, 2, 5]] = True
for label, fixed in [('dynamic', None), ('fixed', .12)]:
    c = ControllerConfig(fixed_roll_reference=fixed)
    initial = jax.vmap(lambda _: initial_controller(c))(jp.arange(8))
    def step(state, inputs):
        measurement, enabled, reset_mask = inputs
        state = jax.tree_util.tree_map(lambda x, y: jp.where(reset_mask.reshape((8,)+(1,)*(x.ndim-1)), y, x), state, initial)
        new, out = jax.vmap(lambda s, m, e: controller_step(s, m, e, c))(state, measurement, enabled)
        output = jp.stack((out.steer_rate, out.reference_roll, out.disturbance, out.equilibrium_shift), axis=-1)
        return new, (output, new.eso, new.disturbance, new.gains)
    _, trace = jax.jit(lambda: jax.lax.scan(step, initial, (jp.asarray(measurements), jp.asarray(enable), jp.asarray(reset))))()
    np.savez_compressed(args.output/(label+'.npz'), measurements=measurements, enable=enable, reset=reset,
                        output=np.asarray(trace[0]), eso=np.asarray(trace[1]), disturbance=np.asarray(trace[2]), gains=np.asarray(trace[3]))
    (args.output/(label+'.json')).write_text(json.dumps(asdict(c), indent=2))
(args.output/'identity.json').write_text(json.dumps(dict(source=str(Path(source.__file__).resolve()),
    sha256=hashlib.sha256(Path(source.__file__).read_bytes()).hexdigest(), steps=1024, lanes=8,
    dtype='float64', physics_steps=0, optimizer_updates=0), indent=2))
contract = export_isaac_asset(args.output/'asset')
model = load_model().model
names = [j['name'] for j in contract['joints']]
indices = [int(model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]) for n in names]
commands = rng.uniform(-1, 1, (512, 2))*np.array([4., 70.])
velocities = rng.uniform(-1, 1, (512, 4))
velocities[:, names.index('steering_joint')] = commands[:, 0]+rng.normal(0, .05, 512)
velocities[:, names.index('rearwheel_joint')] = -commands[:, 1]+rng.normal(0, .8, 512)
forces = []
for command, velocity in zip(commands, velocities):
    data = mujoco.MjData(model)
    data.qvel[indices] = velocity
    data.ctrl[:] = [0., -command[1], command[0], command[0]]
    mujoco.mj_forward(model, data)
    forces.append(data.qfrc_actuator[indices].copy())
np.savez_compressed(args.output/'servos.npz', command=commands, velocity=velocities, force=forces)
from sttw_control.env import RecoveryEnv
helper = object.__new__(RecoveryEnv)
helper.bundle = load_model()
qindices = [int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]) for n in names]
roots, positions, velocities, measurements, poses, speeds, passive = [], [], [], [], [], [], []
for _ in range(256):
    data = mujoco.MjData(model)
    data.qpos[:3] = rng.normal(size=3)
    quaternion = rng.normal(size=4)
    data.qpos[3:7] = quaternion/np.linalg.norm(quaternion)
    data.qpos[qindices] = rng.uniform(-.1, .1, len(qindices))
    data.qvel[:] = rng.normal(size=model.nv)
    mujoco.mj_forward(model, data)
    spatial = np.zeros(6)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, helper.bundle.chassis, spatial, 0)
    roots.append(np.r_[data.xpos[helper.bundle.chassis], np.roll(data.xquat[helper.bundle.chassis], -1), data.qvel[:3], spatial[:3]])
    positions.append(data.qpos[qindices].copy())
    velocities.append(data.qvel[indices].copy())
    measurements.append(np.asarray(helper.measure(data)))
    pose = np.asarray(helper.pose(data))
    poses.append(pose)
    speeds.append(data.qvel[0]*np.cos(pose[2])+data.qvel[1]*np.sin(pose[2]))
    passive.append(data.qfrc_passive[indices].copy())
np.savez_compressed(args.output/'signals.npz', root=roots, position=positions, velocity=velocities,
                    measurement=measurements, pose=poses, speed=speeds, passive=passive)
