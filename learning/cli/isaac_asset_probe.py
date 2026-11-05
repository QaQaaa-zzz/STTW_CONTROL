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
    parser.add_argument('--terrain-mesh', type=Path)
    parser.add_argument('--terrain-sha256')
    parser.add_argument('--controller-config', type=Path)
    parser.add_argument('--lower-actor', type=Path)
    parser.add_argument('--lower-sha256')
    parser.add_argument('--drive-speed', type=float, default=2.3)
    parser.add_argument('--path-bundle', type=Path)
    parser.add_argument('--alpha', type=float, choices=(0.,.25,.5,.75,1.), default=.5)
    parser.add_argument('--zero-middle', action='store_true')
    parser.add_argument('--contact-offset', type=float)
    parser.add_argument('--rest-offset', type=float)
    parser.add_argument('--policy-std', nargs=2, type=float)
    parser.add_argument('--seed', type=int, default=20261105)
    parser.add_argument('--source-inertia', action='store_true')
    parser.add_argument('--gpu-physics', action='store_true')
    args = parser.parse_args()
    if not 0 <= args.steps <= (25000 if args.controller_config else 1000) or not 0 < args.drive_speed <= 3.5:
        parser.error('passive <=1000 / controlled <=25000 physics steps; speed in (0,3.5]')
    if args.output.exists() or args.output.with_suffix('.npz').exists():
        parser.error('output exists; preserve previous evidence')
    if args.source_inertia and args.steps:
        parser.error('runtime inertia setter changes link geometry in this asset; static audit only until import is repaired')
    if bool(args.lower_actor) != bool(args.lower_sha256) or (args.lower_actor and not args.controller_config):
        parser.error('lower actor requires hash and controller configuration')
    if (args.path_bundle and not args.lower_actor) or (args.zero_middle and not args.path_bundle):
        parser.error('path bundle requires frozen lower; zero-middle requires path bundle')
    if bool(args.terrain_mesh) != bool(args.terrain_sha256) or (args.terrain_mesh and args.ground != 'trimesh'):
        parser.error('terrain mesh requires SHA256 and ground=trimesh')
    if (args.contact_offset is None) != (args.rest_offset is None) or (args.contact_offset is not None and not 0 <= args.rest_offset < args.contact_offset <= .02):
        parser.error('declare both contact/rest offsets with 0 <= rest < contact <= .02')
    if args.policy_std is not None and (not args.path_bundle or args.zero_middle or any(not 0 < x <= .5 for x in args.policy_std)):
        parser.error('sampling requires middle policy and std in (0,.5]')
    gym = gymapi.acquire_gym()
    params = gymapi.SimParams()
    params.dt = .0002
    params.substeps = 1
    params.up_axis = gymapi.UP_AXIS_Z
    params.gravity = gymapi.Vec3(0, 0, -9.81)
    params.physx.use_gpu = args.gpu_physics
    if args.contact_offset is not None:
        params.physx.contact_offset = args.contact_offset
        params.physx.rest_offset = args.rest_offset
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
    result['physx_config'] = {key:getattr(params.physx,key) for key in ('solver_type','num_position_iterations','num_velocity_iterations','contact_offset','rest_offset','bounce_threshold_velocity','max_depenetration_velocity')}
    result['physics_device'] = 'cuda:0' if params.physx.use_gpu else 'cpu'
    result['tensor_pipeline'] = 'gpu' if params.use_gpu_pipeline else 'cpu'
    if args.ground == 'plane':
        plane = gymapi.PlaneParams()
        plane.normal = gymapi.Vec3(0, 0, 1)
        if path.suffix == '.urdf':
            plane.static_friction = plane.dynamic_friction = source_contract['ground_friction'][0]
        result['ground_material'] = dict(static_friction=plane.static_friction,dynamic_friction=plane.dynamic_friction,restitution=plane.restitution,distance=plane.distance)
        gym.add_ground(sim, plane)
    elif args.ground == 'trimesh':
        vertices = np.array([[-2, -2, 0], [2, -2, 0], [2, 2, 0], [-2, 2, 0]], dtype=np.float32)
        triangles = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)
        if args.terrain_mesh:
            digest = hashlib.sha256(args.terrain_mesh.read_bytes()).hexdigest()
            if digest != args.terrain_sha256:
                raise ValueError('terrain mesh identity mismatch')
            terrain = np.load(args.terrain_mesh, allow_pickle=False)
            vertices = terrain['vertices']; triangles = terrain['triangles']
            if vertices.ndim != 2 or vertices.shape[1] != 3 or triangles.ndim != 2 or triangles.shape[1] != 3 or vertices.dtype != np.float32 or triangles.dtype != np.uint32 or not np.isfinite(vertices).all() or triangles.max() >= len(vertices):
                raise ValueError('invalid terrain mesh arrays')
            result['terrain_mesh'] = dict(path=str(args.terrain_mesh.resolve()), sha256=digest, vertices=len(vertices), triangles=len(triangles), max_height_m=float(vertices[:,2].max()))
        mesh = gymapi.TriangleMeshParams()
        mesh.nb_vertices = len(vertices)
        mesh.nb_triangles = len(triangles)
        if path.suffix == '.urdf':
            mesh.static_friction = mesh.dynamic_friction = source_contract['ground_friction'][0]
        result['ground_material'] = dict(static_friction=mesh.static_friction,dynamic_friction=mesh.dynamic_friction,restitution=mesh.restitution)
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
                if args.contact_offset is not None:
                    properties[index].contact_offset = args.contact_offset
                    properties[index].rest_offset = args.rest_offset
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
        lower = None
        if args.lower_actor:
            from sttw_control.torch_local_lower import TorchLocalLower
            lower = TorchLocalLower(args.lower_actor, args.lower_sha256, 1)
            result['lower_actor'] = dict(path=str(args.lower_actor.resolve()), sha256=args.lower_sha256, frequency_hz=200)
        previous_reference = torch.tensor([[args.drive_speed, 0.]])
        previous_command = torch.tensor([[0., args.drive_speed/.1]])
        previous_yaw = None
        applied = torch.zeros(1, 2)
        middle = None
        torch.manual_seed(args.seed)
        noise_std = None if args.policy_std is None else torch.tensor(args.policy_std)
        result['sampling'] = dict(seed=args.seed, policy_std=args.policy_std)
        if args.path_bundle:
            from sttw_control.torch_path_policy import TorchPathPolicy
            from sttw_control.torch_path_reference import TorchPathReference
            from sttw_control.torch_middle_control import TorchMiddleControl
            bundle = json.loads((args.path_bundle/'identity.json').read_text())
            for filename, digest in bundle['files'].items():
                if hashlib.sha256((args.path_bundle/filename).read_bytes()).hexdigest() != digest:
                    raise ValueError('path bundle identity mismatch: '+filename)
            path_config = json.loads((args.path_bundle/'config.json').read_text())
            policy = TorchPathPolicy(args.path_bundle/'actor.npz', bundle['files']['actor.npz'], path_config, 1)
            route = TorchPathReference(np.load(args.path_bundle/'route.npz'), path_config)
            middle = TorchMiddleControl(policy, route, bundle['frame_scales'], torch.tensor([args.alpha]), torch.tensor([args.drive_speed]))
            result['path_contract'] = dict(bundle=bundle, alpha=args.alpha, zero_middle=args.zero_middle)

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
    lower_trace = []
    middle_trace = []
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
                governed = previous_reference
                if middle is not None:
                    if tick == 0:
                        middle.reset(measurement, measured_pose, speed, previous_reference, previous_reference, previous_command, controller)
                    else:
                        middle.after(measurement, measured_pose, speed, previous_reference, previous_command, applied, controller)
                    governed = middle.before(measured_pose, args.zero_middle, noise_std)
                    if middle.fault.any() or middle.domain_exit.any():
                        result['middle_terminal'] = dict(fault=middle.fault.tolist(), domain_exit=middle.domain_exit.tolist(), preview_body=middle.follower['preview_body'].tolist(), projection_progress=middle.projection['progress'].tolist(), cross_track=middle.projection['cross_track'].tolist(), pose=measured_pose.tolist())
                        termination = 'middle_fault_or_path_domain'
                        break
                    middle_trace.append(np.r_[tick*.0002, middle.nominal.numpy().ravel(), governed.numpy().ravel(),
                        middle.projection['progress'].numpy(), middle.projection['cross_track'].numpy(),
                        middle.obs.numpy().ravel(), middle.latent.numpy().ravel()])
                if lower is not None:
                    if previous_yaw is not None:
                        difference = measured_pose[:, 2]-previous_yaw
                        lower.finish(applied, torch.atan2(torch.sin(difference), torch.cos(difference))/.005)
                    action, observation, finite = lower.prepare(measurement, speed, governed,
                        previous_reference, previous_command, controller)
                    if not finite.all():
                        termination = 'lower_policy_fault'
                        break
                    previous_yaw = measured_pose[:, 2].clone()
                    applied = action*torch.tensor([1.5, 10.])
                    lower_trace.append(np.r_[tick*.0002, observation.numpy().ravel(), action.numpy().ravel(), applied.numpy().ravel()])
                row = torch.stack((measurement[:, 5]*.1, measurement[:, 2], measurement[:, 3],
                                   measurement[:, 0], measurement[:, 1], governed[:, 1]), dim=-1)
                output = controller.step(row, torch.tensor([tick*.0002 > 3.]))
                steer = (output[:, 0]+applied[:, 0]).clamp(-3., 3.)
                steer = torch.maximum(torch.maximum(torch.tensor(-3.), (-.8-position[:, signals.steer])/.005),
                                      torch.minimum(torch.minimum(torch.tensor(3.), (.8-position[:, signals.steer])/.005), steer))
                command = torch.stack((steer, (governed[:, 0]/.1+applied[:, 1]).clamp(-60.,60.)), dim=-1)
                previous_command = command.clone()
                previous_reference = governed.clone()
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
                  scope=('Frozen LOCAL300 plus ECBC engineering probe' if args.lower_actor else 'ECBC engineering probe without learned policies') if args.controller_config else 'Passive contact engineering probe; no controller, policy or training')
    if args.path_bundle:
        result['scope'] = 'Fixed-path H75/zero-middle with frozen LOCAL300 engineering probe; no training'
    if trajectory:
        np.savez_compressed(args.output.with_suffix('.npz'), body_states=np.stack(trajectory), contact_counts=contacts,
                            control_trace=np.asarray(control_trace), lower_trace=np.asarray(lower_trace), middle_trace=np.asarray(middle_trace), contact_trace=np.asarray(contact_trace), dt=params.dt)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    gym.destroy_sim(sim)


if __name__ == '__main__':
    main()
