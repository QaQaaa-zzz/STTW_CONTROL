"""Prepare the frozen E2 phase/initial-speed grid and run it only after the replay gate passes."""
import argparse
from collections import Counter
import csv
import json
import os
import shutil
from pathlib import Path
import subprocess
import time

import numpy as np

from pcrp_replay_gate import REPO, read, sha, write


CENTERS = (3.9, 4.2, 4.5)
DS_CM = (-4, -2, -1, 0, 1, 2, 4)
DV_M_S = (-.02, -.01, .01, .02)
ALPHAS = (0., .25, .5, .75, 1.)
GYM = Path('/home/qy/miniconda3/envs/go2_isaacgym')
LAB = '/home/qy/miniconda3/envs/env_isaaclab/bin/python'


def shifted_velocity(state, increment):
    """Add only horizontal velocity parallel to the measured vehicle heading."""
    w, x, y, z = state['root_pose'][3:]
    yaw = np.arctan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z))
    velocity = state['root_velocity'].copy()
    velocity[:2] += increment * np.array([np.cos(yaw), np.sin(yaw)])
    return velocity


def build_ridges(stage, oat):
    geometry = {}
    for center in CENTERS:
        original = oat/'terrains'/f's{center}_length0.8_oblique0.25_h0.1'
        base_cfg = read(oat/'scene_configs'/f'{original.name}.json')
        with np.load(original/'terrain.npz') as a:
            base = a['height_field']
        for ds in DS_CM:
            if ds == 0:
                path = original
            else:
                path = stage/'terrains'/f's{center}_ds{ds:+d}cm'
                cfg = dict(base_cfg,center_s_m=round(center+ds/100, 3))
                scene = Path(str(path)+'.json')
                scene.parent.mkdir(parents=True,exist_ok=True)
                if path.exists():
                    if read(scene) != cfg:
                        raise ValueError(f'Mesh configuration mismatch at {path}')
                else:
                    write(scene,cfg)
                    subprocess.run([GYM/'bin/python','learning/cli/place_isaac_terrain.py',
                                    '--terrain','bend_bump','--scene-config',scene,'--output',path],
                                   cwd=REPO,env=dict(os.environ,LD_LIBRARY_PATH=str(GYM/'lib')+
                                   os.pathsep+os.environ.get('LD_LIBRARY_PATH','')),
                                   stdout=subprocess.DEVNULL,check=True)
            meta = read(path/'identity.json')
            if meta['sha256'] != sha(path/'terrain.npz') or meta['center_s_m'] != round(center+ds/100,3):
                raise ValueError(f'Mesh bytes/position mismatch at {path}')
            with np.load(path/'terrain.npz') as a:
                delta = a['height_field'].astype(int) - base
            if ds and not delta.any():
                raise ValueError(f'Grid quantization eliminated requested perturbation {center}, {ds}')
            geometry[center,ds] = dict(terrain=str(path),center_m=center,ds_cm=ds,
                                      actual_center_m=meta['center_s_m'],terrain_sha256=meta['sha256'],
                                      route_sha256=meta['route_sha256'],
                                      cells_changed_from_original=int(np.count_nonzero(delta)),
                                      max_height_change_m=float(np.abs(delta).max()*meta['vertical_scale_m']))
    if len(set(row['terrain_sha256'] for row in geometry.values())) != len(geometry):
        raise ValueError('Some distinct ridge placements produced identical meshes')
    return geometry


def prepare(root, matched=False):
    stage = root/'e2'
    stage.mkdir(parents=True,exist_ok=True)
    old_protocol = stage/'protocol.json'
    if old_protocol.exists():
        previous_matched = read(old_protocol).get('packing') == 'matched_packing'
        if previous_matched and not matched:
            raise ValueError('This run already registered a matched packing protocol')
        if matched and not previous_matched:
            archive = stage/'original_packing'
            archive.mkdir(exist_ok=True)
            for name in ('protocol.json','lanes.csv','verification.json'):
                old = stage/name
                if old.is_file() and not (archive/name).exists():
                    shutil.copyfile(old,archive/name)
    oat = root.parent/'h75_contact_oat'
    position = oat/'position'
    sources = {Path(c['output']).name:c for c in read(position/'batch_000.json')}
    proto = read(root/'step0/protocol.json')
    route = str(Path(proto['bundle'])/'route.npz')
    cfg = read(position/'flat_original_v2.6_seed401.json')
    start = REPO/cfg['initial_state']
    if (sha(Path(route))!=proto['identities']['route.npz']
            or sha(start)!=proto['identities']['initial_state']):
        raise ValueError('Original route or initial state does not match Step 0')
    geometry = build_ridges(stage,oat)
    write(stage/'geometry.json',list(geometry.values()))
    original_batch = read(Path(proto['original_batch'])/'contract.json')
    settings = dict(cfg,seed=original_batch['settings']['seed'])
    states = {}
    (stage/'initial_states').mkdir(exist_ok=True)
    with np.load(start) as a:
        state = dict(a)
    for dv in DV_M_S:
        path = stage/'initial_states'/f'dv{dv:+.2f}.npz'
        velocity = shifted_velocity(state,dv)
        if not path.exists():
            np.savez(path,**dict(state,root_velocity=velocity))
        with np.load(path) as a:
            for k, val in state.items():
                np.testing.assert_array_equal(a[k],velocity if k=='root_velocity' else val)
        states[dv] = path
    batches, lanes, physical_lanes = [], [], []
    contracts = {}
    # Unique original physical translations for all lanes within each batch.
    ds_values = (0, *(ds for ds in DS_CM if ds)) if matched else DS_CM
    dv_values = (0., *DV_M_S) if matched else DV_M_S
    shifts = [('ds',ds) for ds in ds_values]+[('dv',dv) for dv in dv_values]
    sweeps = [(axis,shift,idx) for axis,shift in shifts for idx in (range(5) if matched else (None,))]
    batch_root = stage/'matched' if matched else stage
    batch_root.mkdir(exist_ok=True)
    for axis,shift,alpha_idx in sweeps:
        group = f'ds{shift:+d}cm' if axis=='ds' else f'dv{shift:+.2f}'
        group_dir = batch_root/(f'{group}_a{ALPHAS[alpha_idx]:g}' if matched else group)
        indices = (alpha_idx,) if matched else range(5)
        cases,origins = [],[]
        for center in (CENTERS if axis=='ds' else (*CENTERS,None)):
            layout = 'flat_original' if center is None else f'position_s{center}'
            for seed in range(401,409 if axis=='ds' else 405):
                key = f'{layout}_v2.6_seed{seed}'
                c = sources[key]
                archive = Path(c['output'])
                if key not in contracts:
                    contract = read(archive/'contract.json')
                    if (contract['identity']['files']!=original_batch['identity']['files']
                            or contract['settings']['randomization']!=c['randomization']
                            or contract['settings']['initial_state']!=cfg['initial_state']
                            or c['alphas'][:5]!=list(ALPHAS)
                            or any(c['zero_middle'][:5]) or c['user_speed_m_s']!=2.6):
                        raise ValueError(f'Unmatched source identity for {key}')
                    archived_mesh=read(Path(c['terrain_path'])/'identity.json')
                    if (archived_mesh['sha256']!=contract['terrain'][0]['sha256'] or
                            archived_mesh['route_sha256']!=proto['identities']['route.npz']):
                        raise ValueError(f'Unmatched mesh or route in {key}')
                    contracts[key] = contract
                contract=contracts[key]
                terrain = geometry[center,shift]['terrain'] if axis=='ds' else c['terrain_path']
                reuse = not matched and axis=='ds' and shift==0
                output = archive if reuse else group_dir/key
                for i in indices:
                    alpha = ALPHAS[i]
                    row = dict(stage='E2a' if axis=='ds' else ('E2b' if center else 'E2c'),
                                      layout=layout,center_m=center,seed=seed,
                                      ds_cm=shift if axis=='ds' else 0,dv_m_s=shift if axis=='dv' else 0.,
                                      alpha=alpha,source='archive' if reuse else 'new',output=str(output),
                                      lane=0 if matched else i,route=route,terrain=terrain,base_output=str(archive))
                    if matched:
                        nominal = 'ds+0cm' if axis=='ds' else 'dv+0.00'
                        base = batch_root/f'{nominal}_a{alpha:g}'/key
                        slot = len(origins)
                        row.update(base_output=str(base),global_slot=slot,alpha_index=i)
                        physical_lanes.append(dict(**row,axis=axis,batch=group_dir.name,
                             zero_batch=base.parent.name,zero_output=str(base),zero_global_slot=slot,
                             alpha0_output=str(batch_root/f'{group}_a0'/key),alpha0_global_slot=slot,
                             world_origin=json.dumps(contract['origin'][0]),historical_archive=str(archive),
                             role='paired_zero_only' if axis=='dv' and shift==0 else 'primary'))
                    if not (matched and axis=='dv' and shift==0):
                        lanes.append(row)
                if not reuse:
                    case = dict(c,terrain_path=terrain,output=str(output))
                    for name in ('alphas','zero_middle','case_labels','zero_reference_channels'):
                        case[name]=[c[name][i] for i in indices]
                    cases.append(case)
                    origins.extend(contract['origin'][:len(indices)])
        if cases:
            group_dir.mkdir(exist_ok=True)
            if len(set(map(tuple,origins)))!=len(origins):
                raise ValueError(f'Archive world origins overlap in {group_dir}')
            write(group_dir/'cases.json',cases)
            write(group_dir/'settings.json',dict(settings,
                  initial_state=str(states[shift]) if axis=='dv' and shift else cfg['initial_state'],
                  plant_options=dict(cfg['plant_options'],replay_clone_origins=origins)))
            batches.append(dict(directory=str(group_dir),lanes=len(origins),
                                first_terrain=cases[0]['terrain_path'],
                                **(dict(axis=axis,alpha=ALPHAS[alpha_idx],shift=shift,
                                        zero_batch=base.parent.name,alpha0_batch=f'{group}_a0') if matched else {})))
    expected=dict(E2a=840,E2b=240,E2c=80)
    new_count = 1240 if matched else 1040
    unique={(r['stage'],r['layout'],r['seed'],r['alpha'],r['ds_cm'],r['dv_m_s']) for r in lanes}
    if (Counter(r['stage'] for r in lanes)!=expected or len(unique)!=1160 or len(lanes)!=1160
            or sum(b['lanes'] for b in batches)!=new_count):
        raise ValueError('E2 planned budget or Cartesian grid mismatch')
    with (stage/'lanes.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=lanes[0]);writer.writeheader();writer.writerows(lanes)
    if matched:
        verify_packing(physical_lanes,batches)
        with (stage/'paired_lanes.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=physical_lanes[0]);writer.writeheader();writer.writerows(physical_lanes)
        write(stage/'verification.json',dict(stage='CPU_checked',design='matched_packing',
              primary_conditions=len(lanes),physical_lanes=len(physical_lanes),dv_extra_zero_lanes=80,
              ds_batches=35,ds_lanes_per_batch=24,dv_batches=25,dv_lanes_per_batch=16,
              matched_fields='Each layout and group retains its original alpha0 world origin and identical physical slot in every alpha and perturbation batch. Within shift only alpha may change; within alpha only the designated perturbation may change.',
              alignment_file='paired_lanes.csv',rollouts_launched_by_prepare=0))
    write(stage/'protocol.json',dict(counts=expected,new=new_count,reused=0 if matched else 120,batches=batches,
          **(dict(packing='matched_packing',single_alpha_batches=True,extra_dv_zero=80,primary=1160,
                  wall_s_scope='shared GPU queue waiting time plus process startup, simulation, output, and shutdown',
                  historical_step0=read(root/'step0/status.json').get('passed'),
                  gate='Require user matched_packing decision, completed original step0 and successful 672/672 replay') if matched else
             dict(gate='Require step0/status.json stage=complete and passed=true before simulation')),
          bundle=proto['bundle'],input_state_sha256=sha(start),
          initial_speed_increment='Horizontal root-link velocity along the yaw direction used by TorchVehicleSignals.observe',
          initial_states=[dict(dv_m_s=dv,path=str(p),sha256=sha(p)) for dv,p in states.items()],
          archive_batch=proto['original_batch'],role='Previously examined physical groups 401-408 DEV',
          stop='One pass of the declared design; no automatic additional samples.'))
    return stage


def settings_except(settings,variable):
    omitted={'ds':('terrains',),'dv':('initial_state',),'alpha':('alphas','lane_alphas')}[variable]
    return {k:v for k,v in settings.items() if k not in omitted}


def verify_packing(physical_lanes,batches):
    from sttw_control.isaac_evaluation_batch import batch_settings
    if len(physical_lanes)!=1240 or Counter(row['role'] for row in physical_lanes)!=dict(primary=1160,paired_zero_only=80):
        raise ValueError('Invalid matched condition/control budget')
    if Counter((s['axis'],s['lanes']) for s in batches)!={('ds',24):35,('dv',16):25}:
        raise ValueError('Unexpected E2 group count or physical batch size')
    settings={}
    cases={}
    rows_by_batch={}
    for spec in batches:
        folder=Path(spec['directory'])
        manifest=read(folder/'cases.json')
        settings[folder.name]=batch_settings(read(folder/'settings.json'),manifest)
        cases[folder.name]=manifest
        rows_by_batch[folder.name]=[r for r in physical_lanes if r['batch']==folder.name]
    for spec in batches:
        name=Path(spec['directory']).name
        cfg=settings[name]
        rows=rows_by_batch[name]
        if len(rows)!=spec['lanes'] or cfg['alphas']!=[spec['alpha']]*len(rows):
            raise ValueError(f'Unexpected single-alpha lane count in {name}')
        for variable,base_name,column in [(spec['axis'],spec['zero_batch'],'zero_output'),
                                          ('alpha',spec['alpha0_batch'],'alpha0_output')]:
            if settings_except(cfg,variable)!=settings_except(settings[base_name],variable):
                raise ValueError(f'Unexpected settings change: {name} vs {base_name} ({variable})')
            for slot,(row,base,case) in enumerate(zip(rows,rows_by_batch[base_name],cases[name])):
                if (row[column]!=base['output'] or row['global_slot']!=slot or base['global_slot']!=slot
                        or row['lane']!=0 or case['lane_start']!=slot or case['lane_count']!=1
                        or row['alpha_index']!=ALPHAS.index(spec['alpha']) or row['source']!='new'
                        or any(row[k]!=base[k] for k in ('world_origin','layout','seed','route'))
                        or json.loads(row['world_origin'])!=cfg['plant_options']['replay_clone_origins'][slot]):
                    raise ValueError(f'Mispaired physical slot {slot}: {name} vs {base_name}')


def check_gate(root,matched):
    path = root/'step0/status.json'
    gate = read(path) if path.exists() else {}
    if matched:
        decision=root/'paired_authorization.json'
        recovery=root/'replay672/analysis/summary.json'
        replay=read(recovery) if recovery.exists() else {}
        if (not decision.is_file() or read(decision).get('decision')!='matched_packing'
                or gate.get('stage')!='complete' or replay.get('stage')!='complete'
                or replay.get('passed') is not True or replay.get('matched_lanes')!=672 or replay.get('total_lanes')!=672):
            raise SystemExit('Matched E2 held: requires user matched_packing decision, complete Step 0 and 672/672 recovery.')
    elif gate.get('stage')!='complete' or gate.get('passed') is not True:
        raise SystemExit('E2 held: Step 0 replay gate has not passed.')


def execute(root,matched=False):
    check_gate(root,matched)
    stage=root/'e2'
    protocol=read(stage/'protocol.json')
    if matched!=(protocol.get('packing')=='matched_packing'):
        raise ValueError('Execution mode does not match the prepared E2 protocol')
    env=dict(os.environ,PYTHONPATH=str(REPO/'learning/src')+os.pathsep+os.environ.get('PYTHONPATH',''))
    status=dict(stage='running',completed_new=0,reused=protocol['reused'],batches=[])
    if matched:
        status['wall_s_scope']=protocol['wall_s_scope']
    for spec in protocol['batches']:
        directory=Path(spec['directory'])
        status['active']=directory.name
        write(stage/'status.json',status)
        start=time.monotonic()
        with (directory/'simulation.log').open('x') as log:
            cmd=[LAB,'-u','learning/cli/evaluate_isaac_gpu.py','--backend','lab',
                 '--bundle',protocol['bundle'],'--config',directory/'settings.json',
                 '--case-manifest',directory/'cases.json','--terrain',spec['first_terrain'],
                 '--output',directory/'batch','--seconds','10']
            if matched:
                cmd=['/bin/bash',str(REPO/'learning/scripts/gpu_queue.sh'),*cmd]
            subprocess.run(cmd,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        batch=read(directory/'batch/status.json')
        contract=read(directory/'batch/contract.json')
        if batch['stage']!='complete' or batch['policy_steps']!=500 or contract['settings']['num_envs']!=spec['lanes']:
            raise ValueError(f'Incomplete physical rollout in {directory}')
        if matched:
            fields=('origin','terrain','lab_physics','randomization_readback','delay_steps','identity',
                    'backend','physics_dt','lower_dt','policy_dt','bodies')
            for variable,base_name in [(spec['axis'],spec['zero_batch']),('alpha',spec['alpha0_batch'])]:
                control=read(directory.parent/base_name/'batch/contract.json')
                differences=[key for key in fields if not (variable=='ds' and key=='terrain') and contract[key]!=control[key]]
                if (differences or settings_except(contract['settings'],variable)!=settings_except(control['settings'],variable)
                        or contract['origin']!=read(directory/'settings.json')['plant_options']['replay_clone_origins']
                        or contract['settings']['alphas']!=[spec['alpha']]*spec['lanes']):
                    raise ValueError(f'Runtime pairing mismatch in {directory} vs {base_name}: {differences}')
        status['completed_new']+=spec['lanes']
        status['batches'].append(dict(name=directory.name,new=spec['lanes'],wall_s=time.monotonic()-start,
                                      sampling_s=batch['elapsed_s'],export_s=batch['elapsed_with_output_s']-batch['elapsed_s']))
    status.update(stage='physics_complete',active=None)
    write(stage/'status.json',status)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    action=p.add_mutually_exclusive_group(required=True)
    action.add_argument('--prepare-only',action='store_true')
    action.add_argument('--execute',action='store_true')
    p.add_argument('--matched',action='store_true',help='Use newly simulated zero shifts in identical batch slots')
    args=p.parse_args()
    if args.prepare_only:
        prepare(args.run.resolve(),args.matched)
    else:
        execute(args.run.resolve(),args.matched)


if __name__=='__main__':
    main()
