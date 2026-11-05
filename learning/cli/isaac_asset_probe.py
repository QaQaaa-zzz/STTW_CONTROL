"""Bounded Isaac Gym asset and passive contact diagnostics; no training."""
import argparse
import json
import hashlib
import xml.etree.ElementTree as ET
from pathlib import Path
from isaacgym import gymapi
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--asset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=0)
    parser.add_argument('--ground', choices=('plane', 'box', 'trimesh'), default='plane')
    args = parser.parse_args()
    if not 0 <= args.steps <= 1000:
        parser.error('engineering probe permits at most 1000 physics steps')
    if args.output.exists() or args.output.with_suffix('.npz').exists():
        parser.error('output exists; preserve previous evidence')
    gym = gymapi.acquire_gym()
    params = gymapi.SimParams()
    params.dt = .0002
    params.substeps = 1
    params.up_axis = gymapi.UP_AXIS_Z
    params.gravity = gymapi.Vec3(0, 0, -9.81)
    params.physx.use_gpu = False
    sim = gym.create_sim(0, -1, gymapi.SIM_PHYSX, params)
    if sim is None:
        raise RuntimeError('Isaac simulation creation failed')
    options = gymapi.AssetOptions()
    options.fix_base_link = False
    options.collapse_fixed_joints = False
    options.override_com = False
    options.override_inertia = False
    path = args.asset.resolve()
    asset = gym.load_asset(sim, str(path.parent), path.name, options)
    if asset is None:
        gym.destroy_sim(sim)
        raise RuntimeError('Isaac asset import failed')
    result = dict(asset=str(path), asset_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  bodies=gym.get_asset_rigid_body_names(asset),
                  dofs=gym.get_asset_dof_names(asset),
                  shapes=gym.get_asset_rigid_shape_count(asset),
                  physics_steps=0, optimizer_updates=0,
                  scope='Import diagnostics only; not collision or closed-loop validation')
    if args.ground == 'plane':
        plane = gymapi.PlaneParams()
        plane.normal = gymapi.Vec3(0, 0, 1)
        gym.add_ground(sim, plane)
    elif args.ground == 'trimesh':
        vertices = np.array([[-2, -2, 0], [2, -2, 0], [2, 2, 0], [-2, 2, 0]], dtype=np.float32)
        triangles = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)
        mesh = gymapi.TriangleMeshParams()
        mesh.nb_vertices = 4
        mesh.nb_triangles = 2
        gym.add_triangle_mesh(sim, vertices.ravel(), triangles.ravel(), mesh)
    env = gym.create_env(sim, gymapi.Vec3(-3, -3, -1), gymapi.Vec3(3, 3, 3), 1)
    if args.ground == 'box':
        fixed = gymapi.AssetOptions()
        fixed.fix_base_link = True
        ground = gym.create_box(sim, 4., 4., .1, fixed)
        pose = gymapi.Transform()
        pose.p.z = -.05
        gym.create_actor(env, ground, pose, 'ground', 0, 0)
    pose = gymapi.Transform()
    root_body = ET.parse(path).find('worldbody/body')
    pose.p = gymapi.Vec3(*map(float, root_body.get('pos', '0 0 0').split()))
    actor = gym.create_actor(env, asset, pose, 'sttw', 0, 1)
    time_before_prepare = gym.get_sim_time(sim)
    before_dofs = gym.get_actor_dof_states(env, actor, gymapi.STATE_ALL).copy()
    before_prepare = gym.get_actor_rigid_body_states(env, actor, gymapi.STATE_ALL).copy()
    gym.prepare_sim(sim)
    result.update(time_before_prepare_s=time_before_prepare, time_after_prepare_s=gym.get_sim_time(sim), before_prepare_states=before_prepare.tolist())
    gym.set_actor_rigid_body_states(env, actor, before_prepare, gymapi.STATE_ALL)
    gym.set_actor_dof_states(env, actor, before_dofs, gymapi.STATE_ALL)
    initial = gym.get_actor_rigid_body_states(env, actor, gymapi.STATE_ALL).copy()
    initial_dofs = gym.get_actor_dof_states(env, actor, gymapi.STATE_ALL).copy()
    contacts = []
    trajectory = []
    for _ in range(args.steps):
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        contacts.append(len(gym.get_env_rigid_contacts(env)))
        states = gym.get_actor_rigid_body_states(env, actor, gymapi.STATE_ALL).copy()
        trajectory.append(states.view(np.float32).reshape(len(states), -1))
    final = gym.get_actor_rigid_body_states(env, actor, gymapi.STATE_ALL).copy()
    masses = [p.mass for p in gym.get_actor_rigid_body_properties(env, actor)]
    result.update(physics_steps=args.steps, ground=args.ground,
                  dt_s=params.dt, body_masses_kg=masses,
                  contact_steps=sum(n > 0 for n in contacts), max_contacts=max(contacts, default=0),
                  initial_states=initial.tolist(),
                  initial_dofs=initial_dofs.tolist(),
                  final_dofs=gym.get_actor_dof_states(env, actor, gymapi.STATE_ALL).tolist(),
                  shape_properties=[{k:getattr(p,k) for k in ('filter','friction','rolling_friction','torsion_friction','contact_offset','rest_offset')} for p in gym.get_actor_rigid_shape_properties(env, actor)],
                  shape_filters=[p.filter for p in gym.get_actor_rigid_shape_properties(env, actor)],
                  initial_body_z_m=initial['pose']['p']['z'].tolist(),
                  final_body_z_m=final['pose']['p']['z'].tolist(),
                  finite=bool(np.isfinite(final.view(np.float32)).all()),
                  scope='Passive contact engineering probe; no controller, policy or training')
    if args.steps:
        np.savez_compressed(args.output.with_suffix('.npz'), body_states=np.stack(trajectory), contact_counts=contacts, dt=params.dt)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    gym.destroy_sim(sim)


if __name__ == '__main__':
    main()
