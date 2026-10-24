"""Offline model/command ABI audit on authentic full prefixes; no ROS or physics steps."""
import argparse
import json
import pickle
import shutil
import subprocess
from pathlib import Path

import numpy as np


def numpy_actor(parameters, observation, *, modulated=False, squash=False):
    """Portable deterministic reference; weights and normalization are external."""
    p = parameters['params']
    x = np.asarray(observation, dtype=np.float32)
    q = 2*x[..., 336:337]-1 if modulated else None
    count = 4 if modulated else 3
    for index in range(count):
        layer = p[f'Dense_{index}']
        x = x@np.asarray(layer['kernel'])+np.asarray(layer['bias'])
        if index == count-1:
            break
        if modulated and index > 0:
            x = (1+q*np.asarray(p[f'gain_{index+1}']))*x+q*np.asarray(p[f'shift_{index+1}'])
        negative = x <= 0
        x = x.copy()
        x[negative] = np.expm1(x[negative])
    return np.tanh(x) if squash else x


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    import jax
    import jax.numpy as j
    import yaml
    from sttw_control.path_command_training import load_environment
    from sttw_control.path_command_policy import make_path_actor, correction_tick
    from sttw_control.geometric_path import build_path
    from sttw_control.closed_loop_kernel import controls, preview_controls
    from sttw_control.actuator import ActuatorConfig, apply_residual, initial_actuator
    from sttw_control.lower_command import SCALES
    from sttw_control.preference_best import atomic_json, digest
    from sttw_control.feature_comparison_evaluation import authenticated_parent_record
    base = Path.cwd()
    declaration_file = base/'runs/reserve_feature_expansion_v1/post_fixed_final_v1/high_nominal_retention/H75/declaration.json'
    declaration = json.loads(declaration_file.read_text())
    models = json.loads((declaration_file.parents[2]/'models.json').read_text())
    model, _ = authenticated_parent_record(models['records']['H75'])
    cfg, source = declaration['config'], declaration['source']
    env, initial = load_environment(cfg, source, 'terminal_contract_v2')
    with Path(model['actor']).open('rb') as stream:
        upper = pickle.load(stream)
    lower_file = source['candidate']['actor']
    with Path(lower_file).open('rb') as stream:
        lower = pickle.load(stream)
    actor = make_path_actor(cfg)
    paths = {name: build_path(builder, np.asarray(env.physics.helpers.pose(initial.data)), cfg)
             for name, builder in [('left125_R2p2', 'left90_R2'), ('right125_R2p2', 'right90_R2')]}
    inputs, snapshots, weight_arrays = [], {}, {}
    for name, parameters in [('upper', upper), ('lower', lower)]:
        for key, value in parameters['params'].items():
            if isinstance(value, dict):
                for field, array in value.items():
                    weight_arrays[f'{name}_{key}_{field}'] = np.asarray(array)
            else:
                weight_arrays[f'{name}_{key}'] = np.asarray(value)
    np.savez_compressed(root/'weights.npz', **weight_arrays)
    obs_upper, obs_lower, latent_upper, actions_lower, governed_rows = [], [], [], [], []
    for run in ['alpha_physical_boundary_v1', 'alpha_physical_boundary_mirror_v1']:
        prefix_file = base/'runs'/run/'prefix_results.json'
        snapshots[str(prefix_file)] = digest(prefix_file)
        for row in json.loads(prefix_file.read_text())['cases']:
            if digest(row['snapshot']) != row['snapshot_sha256']:
                raise ValueError('Prefix snapshot changed')
            with Path(row['snapshot']).open('rb') as stream:
                state = jax.tree.map(j.asarray, pickle.load(stream))
            obs, _, fault = env.observation(state, paths[row['route']], row['speed'], 10.)
            z = actor.apply(upper, obs)
            np.testing.assert_allclose(numpy_actor(upper, np.asarray(obs), modulated=True), np.asarray(z), atol=5e-6, rtol=3e-5)
            _, _, governed, flags = correction_tick(state.filtered, state.offsets, z, state.physical.raw, cfg)
            measurement, pose, speed = env.physics.observe(state.physical.data)
            _, action, local_obs, local_flags = env.lower.prepare_local(state.lower, measurement, pose, speed, governed, state.physical, env.cc)
            portable = numpy_actor(lower, np.asarray(local_obs), squash=True)
            np.testing.assert_allclose(portable, np.asarray(action), atol=5e-6, rtol=3e-5)
            if bool(fault|flags['policy_fault']) or not bool(local_flags['finite']):
                raise ValueError('Real prefix observation fault')
            preview = preview_controls(state.physical.controller, measurement, governed, governed,
                                       state.physical.physical_tick*env.cc.dt>3., env.cc)
            _, _, command = controls(state.physical.controller, state.physical.actuator, measurement,
                governed, governed, state.physical.physical_tick*env.cc.dt>3., False,
                env.cc, env.ac, lower_action=action, previewed=preview)
            inputs.append(dict(case_id=row['state_id'], kind='authentic_prefix_next_control',
                               base=np.asarray(command['u_nom']).tolist(), action=np.asarray(action).tolist(),
                               steer=float(measurement[2]), expected=np.asarray(command['final_command']).tolist(),
                               upper_numpy_error=float(np.max(np.abs(numpy_actor(upper,np.asarray(obs),modulated=True)-np.asarray(z)))),
                               lower_numpy_error=float(np.max(np.abs(portable-np.asarray(action))))))
            snapshots[row['snapshot']] = row['snapshot_sha256']
            obs_upper.append(np.asarray(obs));obs_lower.append(np.asarray(local_obs))
            latent_upper.append(np.asarray(z));actions_lower.append(np.asarray(action));governed_rows.append(np.asarray(governed))
    # Finite end-stop and saturation probes exercise behavior absent from nominal
    # snapshots; these are synthetic command checks, not robot experiments.
    for i, (base_command, action, steer) in enumerate([
        ([2.9,59.8],[.8,.8],.799), ([-2.9,-59.8],[-.8,-.8],-.799),
        ([4.,30.],[0.,0.],0.), ([-4.,-30.],[0.,0.],0.),
        ([0.,30.],[.5,.5],.85), ([0.,30.],[-.5,-.5],-.85)]):
        _, expected = apply_residual(initial_actuator(env.ac),j.asarray(base_command),j.asarray(action),j.asarray(steer),env.ac)
        inputs.append(dict(case_id=f'synthetic_{i}', kind='synthetic_finite_limits', base=base_command,
                           action=action, steer=steer, expected=np.asarray(expected).tolist()))
    ros = base/'mujoco_ros_ws20260313/mujoco_ros'
    ros_cfg = yaml.safe_load((ros/'config/controller.yaml').read_text())
    header = ros/'include/mujoco_ros/ECBC/residual_command.h'
    shutil.copyfile(header, root/'residual_command.h')
    driver = root/'compose_audit.cpp'
    driver.write_text('''#include <iostream>
#include <iomanip>
#include "residual_command.h"
int main(){int id,profile;double b0,b1,a0,a1,steer,dt,p,r0,r1,s0,s1,strength;
while(std::cin>>id>>profile>>b0>>b1>>a0>>a1>>steer>>dt>>p>>r0>>r1>>s0>>s1>>strength){
sttw::Limits c;c.steer_position=p;c.steer_rate=r0;c.rear_rate=r1;c.steer_scale=s0;c.rear_scale=s1;c.strength=strength;
auto y=sttw::composeCommand({b0,b1},{a0,a1},steer,dt,c,true);
std::cout<<id<<" "<<profile<<" "<<std::setprecision(17)<<y.steer<<" "<<y.rear<<"\\n";}}
''')
    compile_result = subprocess.run(['g++','-std=c++17','-O2',str(driver),'-o',str(root/'compose_audit')],capture_output=True,text=True)
    (root/'compiler.log').write_text(compile_result.stdout+compile_result.stderr)
    compile_result.check_returncode()
    lines = []
    for i, row in enumerate(inputs):
        for profile in [0,1]:
            front, rear = (ros_cfg['steer_residual_scale'],ros_cfg['rear_residual_scale']) if profile==0 else (env.ac.steer_residual_scale,env.ac.rear_residual_scale)
            values = [i,profile,*row['base'],*row['action'],row['steer'],env.cc.dt,
                      env.ac.steer_limit,env.ac.steer_rate_limit,env.ac.rear_rate_limit,front,rear,env.ac.strength]
            lines.append(' '.join(map(str,values)))
    (root/'command_inputs.txt').write_text('\n'.join(lines)+'\n')
    result = subprocess.run([str(root/'compose_audit')],input='\n'.join(lines)+'\n',capture_output=True,text=True,check=True)
    (root/'command_outputs.txt').write_text(result.stdout)
    outputs = {(int(v[0]),int(v[1])): np.asarray(list(map(float,v[2:]))) for v in (line.split() for line in result.stdout.splitlines())}
    for i, row in enumerate(inputs):
        np.testing.assert_allclose(outputs[(i,1)],row['expected'],atol=1e-5,rtol=1e-6)
        row.update(cpp_training_profile=outputs[(i,1)].tolist(),
                   cpp_ros_default_profile=outputs[(i,0)].tolist(),
                   cpp_python_max_error=float(np.max(np.abs(outputs[(i,1)]-row['expected']))),
                   default_profile_delta=(outputs[(i,0)]-outputs[(i,1)]).tolist())
    np.savez_compressed(root/'real_prefix_inputs.npz', upper_observation=np.asarray(obs_upper),
                        lower_observation=np.asarray(obs_lower), upper_latent=np.asarray(latent_upper),
                        lower_action=np.asarray(actions_lower), governed_reference=np.asarray(governed_rows))
    contract = dict(upper_actor=model, lower_actor=source['candidate'],
        deterministic_upper='raw2D latent; alpha_modulated351; ELU128/128/64; no tanh until reference mapping',
        deterministic_lower='210D;ELU256/128;tanh2 normalized rate residual; no alpha',
        upper_frame_scales=np.asarray(env.frame_scales).tolist(),
        upper_context_fields=cfg['network']['context_fields'], lower_scales=np.asarray(SCALES).tolist(),
        upper_frame_fields=json.loads((Path(source['upper_run'])/'alpha0/frozen_config_stage52.json').read_text())['network']['frame_fields'],
        frame_update_dt_s=.005, upper_inference_dt_s=.02, lower_inference_dt_s=.005,
        upper_history_frames=16, lower_history_frames=10,
        inherited_upper_history_dt_metadata_s=cfg['network']['history_dt_s'],
        actual_upper_history_window_s=16*.005, actual_lower_history_window_s=10*.005,
        upper_reference_interface=cfg['residual_interface'], actuator=source['actuator'], controller=source['controller'],
        ros_defaults=dict(steer_residual_scale=ros_cfg['steer_residual_scale'],rear_residual_scale=ros_cfg['rear_residual_scale']),
        scope='Offline audit/export only. No ROS node, robot execution, sensor estimator, full ECBC parity, or onboard timing claim.')
    atomic_json(root/'CONTRACT.json', contract)
    source_files = [Path(__file__), declaration_file, header, ros/'config/controller.yaml',
                    ros/'msg/RobotState.msg', ros/'src/balance_control_node.cpp',
                    base/'learning/src/sttw_control/path_command_policy.py',base/'learning/src/sttw_control/frozen_local_lower.py']
    atomic_json(root/'receipt.json', dict(physical_steps=0, optimizer_updates=0, actual_real_prefix_cases=16,
        synthetic_command_cases=6, command_cases=inputs, weights_sha256=digest(root/'weights.npz'),
        snapshots_sha256=snapshots, source_sha256={str(p.resolve()):digest(p) for p in source_files},
        max_upper_numpy_error=max(x['upper_numpy_error'] for x in inputs[:16]),
        max_lower_numpy_error=max(x['lower_numpy_error'] for x in inputs[:16]),
        max_cpp_python_error=max(x['cpp_python_max_error'] for x in inputs),
        max_real_default_delta=np.max(np.abs([x['default_profile_delta'] for x in inputs[:16]]),axis=0).tolist(),
        scope=contract['scope']))
    print(root, '16real inputs+6synthetic commands; NumPy/JAX and C++/Python parity;0physics')


if __name__=='__main__':
    main()
