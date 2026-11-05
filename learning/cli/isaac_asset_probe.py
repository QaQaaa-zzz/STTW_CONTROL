"""Bounded Isaac Gym asset and passive contact diagnostics; no training."""
import argparse
import json
import hashlib
import xml.etree.ElementTree as ET
from pathlib import Path
from isaacgym import gymapi, gymtorch
import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--asset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=0)
    parser.add_argument('--ground', choices=('plane', 'box', 'trimesh'), default='plane')
    parser.add_argument('--controller-config', type=Path)
    parser.add_argument('--drive-speed', type=float, default=2.3)
    parser.add_argument('--source-inertia', action='store_true')
    args = parser.parse_args()
    if not 0 <= args.steps <= (25000 if args.controller_config else 1000) or not 0 < args.drive_speed <= 3.5:
        parser.error('passive <=1000 / controlled <=25000 physics steps; speed in (0,3.5]')
    if args.output.exists() or args.output.with_suffix('.npz').exists():
        parser.error('output exists; preserve previous evidence')
    if args.source_inertia and args.steps:
        parser.error('runtime inertia setter changes link geometry in this asset; static audit only until import is repaired')
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
    if path.suffix == '.urdf':
        source_contract = json.loads(path.with_name('contract.json').read_text())
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
        if path.suffix == '.urdf':
            plane.static_friction = plane.dynamic_friction = source_contract['ground_friction'][0]
        gym.add_ground(sim, plane)
    elif args.ground == 'trimesh':
        vertices = np.array([[-2, -2, 0], [2, -2, 0], [2, 2, 0], [-2, 2, 0]], dtype=np.float32)
        triangles = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)
        mesh = gymapi.TriangleMeshParams()
        mesh.nb_vertices = 4
        mesh.nb_triangles = 2
        if path.suffix == '.urdf':
            mesh.static_friction = mesh.dynamic_friction = source_contract['ground_friction'][0]
        gym.add_triangle_mesh(sim, vertices.ravel(), triangles.ravel(), mesh)
    env = gym.create_env(sim, gymapi.Vec3(-3, -3, -1), gymapi.Vec3(3, 3, 3), 1)
    if args.ground == 'box':
        fixed = gymapi.AssetOptions()
        fixed.fix_base_link = True
        ground = gym.create_box(sim, 4., 4., .1, fixed)
        pose = gymapi.Transform()
        pose.p.z = -.05
        ground_actor = gym.create_actor(env, ground, pose, 'ground', 0, 0)
        if path.suffix == '.urdf':
            properties = gym.get_actor_rigid_shape_properties(env, ground_actor)
            for prop in properties:
                prop.friction = source_contract['ground_friction'][0]
            gym.set_actor_rigid_shape_properties(env, ground_actor, properties)
    pose = gymapi.Transform()
    if path.suffix == '.urdf':
        pose.p = gymapi.Vec3(*map(float, source_contract['root_translation'].split()))
        w, x, y, z = source_contract['root_quat_wxyz']
        pose.r = gymapi.Quat(x, y, z, w)
    else:
        root_body = ET.parse(path).find('worldbody/body')
        pose.p = gymapi.Vec3(*map(float, root_body.get('pos', '0 0 0').split()))
    actor = gym.create_actor(env, asset, pose, 'sttw', 0, 1)
    if path.suffix == '.urdf':
        materials = {g['body']: g['friction'] for g in source_contract['vehicle_geoms']}
        properties = gym.get_actor_rigid_shape_properties(env, actor)
        ranges = gym.get_actor_rigid_body_shape_indices(env, actor)
        for body_name, span in zip(result['bodies'], ranges):
            for index in range(span.start, span.start+span.count):
                properties[index].friction = materials[body_name][0]
        gym.set_actor_rigid_shape_properties(env, actor, properties)
        result['material_contract'] = 'source sliding friction on terrain and vehicle; PhysX rolling/torsion remain zero and differ from source'
    body_ids = {name: gym.get_actor_rigid_body_index(env, actor, i, gymapi.DOMAIN_ENV)
                for i, name in enumerate(result['bodies'])}
    vehicle_ids = set(body_ids.values())
    nonwheel_ids = {i for name, i in body_ids.items() if name not in ('frontwheel', 'rearwheel')}
    rigid_properties = gym.get_actor_rigid_body_properties(env, actor)
    if args.source_inertia:
        source_contract = json.loads(path.with_name('contract.json').read_text())
        if source_contract['exported_xml_sha256'] != result['asset_sha256']:
            raise ValueError('inertia source asset differs')
        for p, body in zip(rigid_properties, source_contract['bodies']):
            inertia = gymapi.Mat33()
            for axis, row in zip(('x', 'y', 'z'), body['inertia_body']):
                setattr(inertia, axis, gymapi.Vec3(*row))
            p.inertia = inertia
        gym.set_actor_rigid_body_properties(env, actor, rigid_properties, False)
        rigid_properties = gym.get_actor_rigid_body_properties(env, actor)
        result['inertia_contract'] = 'source principal inertias rotated into body frame; explicit setter, no recomputation'
    result['rigid_properties'] = [dict(mass=p.mass, com_position=[p.com.x, p.com.y, p.com.z],
        inverse_inertia=[[getattr(getattr(p.invInertia, row), column) for column in ('x', 'y', 'z')] for row in ('x', 'y', 'z')],
        inertia=[[getattr(getattr(p.inertia, row), column) for column in ('x', 'y', 'z')] for row in ('x', 'y', 'z')])
        for p in rigid_properties]
    if args.source_inertia:
        for source, actual in zip(source_contract['bodies'], result['rigid_properties']):
            np.testing.assert_allclose(actual['inertia'], source['inertia_body'], rtol=1e-5, atol=1e-9)
    result['body_environment_indices'] = body_ids
    if args.controller_config:
        import torch
        from sttw_control.torch_controller import TorchController, TorchVelocityServos
        from sttw_control.torch_observation import TorchVehicleSignals
        config = json.loads(args.controller_config.read_text())
        contract = json.loads(path.with_name('contract.json').read_text())
        if contract['exported_xml_sha256'] != result['asset_sha256'] or config['dt'] != .005 or contract['timestep_s'] != .0002:
            raise ValueError('source asset or control timing mismatch')
        joint_names = result['dofs']
        controller = TorchController(config, 1)
        servos = TorchVelocityServos(contract, joint_names)
        signals = TorchVehicleSignals(contract, joint_names)
        properties = gym.get_actor_dof_properties(env, actor)
        properties['driveMode'][:] = gymapi.DOF_MODE_EFFORT
        properties['stiffness'][:] = 0
        properties['damping'][:] = 0
        gym.set_actor_dof_properties(env, actor, properties)
        result.update(controller_config=config, drive_speed=args.drive_speed,
                      passive_force_contract='explicit source spring/damping; Isaac drive stiffness/damping zero',
                      contact_contract='imported friction and contact offsets unchanged from passive probe')
    time_before_prepare = gym.get_sim_time(sim)
    before_dofs = gym.get_actor_dof_states(env, actor, gymapi.STATE_ALL).copy()
    before_prepare = gym.get_actor_rigid_body_states(env, actor, gymapi.STATE_ALL).copy()
    gym.prepare_sim(sim)
    result.update(time_before_prepare_s=time_before_prepare, time_after_prepare_s=gym.get_sim_time(sim), before_prepare_states=before_prepare.tolist())
    if args.controller_config:
        before_prepare['vel']['linear']['x'][:] = args.drive_speed
        before_dofs['vel'][[signals.front, signals.rear]] = -args.drive_speed/.1
    import torch
    roots = gymtorch.wrap_tensor(gym.acquire_actor_root_state_tensor(sim))
    dof_tensor = gymtorch.wrap_tensor(gym.acquire_dof_state_tensor(sim))
    gym.refresh_actor_root_state_tensor(sim)
    gym.refresh_dof_state_tensor(sim)
    actor_index = gym.get_actor_index(env, actor, gymapi.DOMAIN_SIM)
    roots[actor_index] = torch.from_numpy(before_prepare.view(np.float32).reshape(len(before_prepare), -1)[0].copy())
    dof_tensor[:] = torch.from_numpy(before_dofs.view(np.float32).reshape(-1, 2).copy())
    gym.set_actor_root_state_tensor(sim, gymtorch.unwrap_tensor(roots))
    gym.set_dof_state_tensor(sim, gymtorch.unwrap_tensor(dof_tensor))
    result['reset_contract'] = 'Isaac root and DOF tensor APIs after prepare_sim; rigid-link caches refreshed by subsequent integration'
    initial = gym.get_actor_rigid_body_states(env, actor, gymapi.STATE_ALL).copy()
    initial_dofs = gym.get_actor_dof_states(env, actor, gymapi.STATE_ALL).copy()
    contacts = []
    trajectory = []
    control_trace = []
    contact_trace = []
    termination = 'budget_end'
    for tick in range(args.steps):
        if args.controller_config:
            dofs = gym.get_actor_dof_states(env, actor, gymapi.STATE_ALL).copy()
            position = torch.from_numpy(dofs['pos'].copy())[None]
            velocity = torch.from_numpy(dofs['vel'].copy())[None]
            if tick % 25 == 0:
                states = gym.get_actor_rigid_body_states(env, actor, gymapi.STATE_ALL).copy()
                root_state = torch.from_numpy(states.view(np.float32).reshape(len(states), -1)[0:1].copy())
                measurement, measured_pose, speed = signals.observe(root_state, position, velocity)
                if not torch.isfinite(measurement).all() or measurement[0, 0].abs() > .7:
                    termination = 'nonfinite_or_roll_domain'
                    break
                row = torch.stack((measurement[:, 5]*.1, measurement[:, 2], measurement[:, 3],
                                   measurement[:, 0], measurement[:, 1], torch.zeros(1)), dim=-1)
                output = controller.step(row, torch.tensor([tick*.0002 > 3.]))
                steer = output[:, 0].clamp(-3., 3.)
                steer = torch.maximum(torch.maximum(torch.tensor(-3.), (-.8-position[:, signals.steer])/.005),
                                      torch.minimum(torch.minimum(torch.tensor(3.), (.8-position[:, signals.steer])/.005), steer))
                command = torch.stack((steer, torch.full_like(steer, args.drive_speed/.1)), dim=-1)
                control_trace.append(np.r_[tick*.0002, measurement.numpy().ravel(), measured_pose.numpy().ravel(),
                                           speed.numpy(), command.numpy().ravel(), output.numpy().ravel()])
            torque = servos.torque(command, velocity)+signals.passive_torque(position, velocity)
            gym.apply_actor_dof_efforts(env, actor, torque.numpy().ravel())
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        contact = gym.get_env_rigid_contacts(env)
        contacts.append(len(contact))
        failed_contact = False
        for c in contact:
            first, second = int(c['body0']), int(c['body1'])
            contact_trace.append([tick, first, second, float(c['lambda'])])
            if c['lambda'] > 0 and (
                    (first in nonwheel_ids and second not in vehicle_ids) or
                    (second in nonwheel_ids and first not in vehicle_ids)):
                failed_contact = True
        states = gym.get_actor_rigid_body_states(env, actor, gymapi.STATE_ALL).copy()
        trajectory.append(states.view(np.float32).reshape(len(states), -1))
        if failed_contact:
            termination = 'nonwheel_ground_contact'
            break
        if not np.isfinite(trajectory[-1]).all():
            termination = 'nonfinite_state'
            break
    final = gym.get_actor_rigid_body_states(env, actor, gymapi.STATE_ALL).copy()
    masses = [p.mass for p in gym.get_actor_rigid_body_properties(env, actor)]
    result.update(physics_steps=len(trajectory), requested_physics_steps=args.steps, termination=termination, ground=args.ground,
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
                  scope='ECBC engineering probe without learned policies' if args.controller_config else 'Passive contact engineering probe; no controller, policy or training')
    if trajectory:
        np.savez_compressed(args.output.with_suffix('.npz'), body_states=np.stack(trajectory), contact_counts=contacts,
                            control_trace=np.asarray(control_trace), contact_trace=np.asarray(contact_trace), dt=params.dt)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    gym.destroy_sim(sim)


if __name__ == '__main__':
    main()
