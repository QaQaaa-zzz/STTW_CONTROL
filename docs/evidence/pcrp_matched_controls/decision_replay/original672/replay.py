"""Rerun exactly one registered 672-lane position batch using the pinned simulation tree."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

OUT = Path(__file__).resolve().parent
PIN = Path('/home/qy/STTW_CONTROL/runs/worktrees/pcrp-replay-3d90f818')
LAB = '/home/qy/miniconda3/envs/env_isaaclab/bin/python'
sys.dont_write_bytecode = True
sys.path.insert(0,str(PIN/'learning/src'))
from sttw_control.isaac_evaluation_batch import batch_settings


def read(p):
    return json.loads(p.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(p,value):
    p.write_text(json.dumps(value,indent=2)+'\n')


def prepare():
    registration=read(OUT/'registration.json')
    manifest=Path(registration['original_manifest'])
    assert digest(manifest)==registration['original_manifest_sha256']
    source=manifest.parent
    history=read(source/'batches/batch_000/contract.json')
    original=read(manifest)
    settings_path=source/(Path(original[0]['output']).name+'.json')
    output_cases=[dict(c,output=str(OUT/Path(c['output']).name)) for c in original]
    active= batch_settings(read(settings_path),[c.copy() for c in output_cases])
    active.update(diagnostics=True,episode_s=10.)
    if active!=history['settings']:
        raise ValueError(f"Resolved settings differ: {[k for k in set(active)|set(history['settings']) if active.get(k)!=history['settings'].get(k)]}")
    n=active['num_envs'];spacing=active['plant_options']['env_spacing']
    rows=int(np.ceil(n/int(np.sqrt(n))))
    cols=int(np.ceil(n/rows))
    src_origins=history['origin']
    origins=[[.5*spacing*(rows-1)-(i//cols)*spacing,(i%cols)*spacing-.5*spacing*(cols-1),0.] for i in range(n)]
    assert n==672 and src_origins==origins==history['lab_physics']['env_origins']==history['lab_physics']['terrain_layout']['clone_origins']
    assert 'replay_clone_origins' not in active['plant_options']
    assert len(original)==112 and len(active['terrains'])==14 and len(active['fixed_randomization_groups'])==8
    archive_trajectories=0
    for i,c in enumerate(original):
        folder=Path(c['output'])
        contract=read(folder/'contract.json')
        lo,hi=i*6,(i+1)*6
        assert contract['batch_source']==dict(output=str(source/'batches/batch_000'),lane_start=lo,lane_stop=hi)
        assert contract['origin']==origins[lo:hi] and contract['terrain']==history['terrain'][lo:hi]
        assert c['seed']==contract['settings']['seed'] and c['randomization']==contract['settings']['randomization']
        assert contract['settings']['alphas']==c['alphas']==[0,.25,.5,.75,1,0]
        assert c['zero_middle']==contract['settings']['zero_middle']==[False]*5+[True]
        assert c['case_labels']==contract['settings']['case_labels']==['fixed0','fixed0.25','fixed0.5','fixed0.75','fixed1','Zero']
        assert contract['identity']==history['identity']
        for name in ('randomization_readback','delay_steps'):
            assert all(value==history[name][k][lo:hi] for k,value in contract[name].items())
        assert read(folder/'status.json')['policy_steps']==500 and (folder/'trajectory.npz').stat().st_size>0
        archive_trajectories+=hi-lo
    for terrain in active['terrains'].values():
        folder=Path(terrain)
        assert digest(folder/'terrain.npz')==read(folder/'identity.json')['sha256']
    proto=read(OUT.parent/'step0/protocol.json')
    bundle=Path(history['bundle'])
    assert all(digest(bundle/f)==h for f,h in history['identity']['files'].items())
    for k,f in (('asset','asset'),('controller','controller_config'),('initial_state','initial_state'),('local300','lower_actor')):
        assert digest(PIN/active[f])==proto['identities'][k]
    head=subprocess.check_output(['git','-C',str(PIN),'rev-parse','HEAD'],text=True).strip()
    assert head==registration['code_for_comparison']
    subprocess.run(['git','-C',str(PIN),'diff','--quiet','HEAD','--','learning'],check=True)
    write(OUT/'cases.json',output_cases)
    launch=[LAB,'-u','learning/cli/evaluate_isaac_gpu.py','--backend','lab','--bundle',str(bundle),'--config',str(settings_path),
            '--case-manifest',str(OUT/'cases.json'),'--terrain',original[0]['terrain_path'],'--output',str(OUT/'batch'),'--seconds','10']
    check=dict(stage='cpu_verified',archive=str(source),archive_code=history['code_revision'],simulation_code=head,
               complete_resolved_settings_identical=True,original_case_order=True,implicit_grid_shape=[rows,cols],world_origins_matched=n,
               archives_verified=archive_trajectories,case_count=len(original),terrain_assets=14,physical_groups=8,
               original_config=str(settings_path),config_sha256=digest(settings_path),cases_sha256=digest(OUT/'cases.json'),
               command=launch,cwd=str(PIN),PYTHONPATH=str(PIN/'learning/src'))
    write(OUT/'preflight.json',check)


def execute():
    check=read(OUT/'preflight.json')
    assert digest(OUT/'cases.json')==check['cases_sha256'] and digest(Path(check['original_config']))==check['config_sha256']
    cmd=check['command']
    env=dict(os.environ,PYTHONPATH=str(PIN/'learning/src'),PYTHONDONTWRITEBYTECODE='1')
    before=time.monotonic()
    with (OUT/'simulation.log').open('x') as log:
        cp=subprocess.run(cmd,cwd=PIN,env=env,stdout=log,stderr=subprocess.STDOUT)
    wall=time.monotonic()-before
    timing=dict(stage='error' if cp.returncode else 'awaiting_validation',returncode=cp.returncode,wall_s=wall)
    write(OUT/'timing.json',timing)
    if cp.returncode:
        raise SystemExit(cp.returncode)
    meta=read(OUT/'batch/contract.json'); batch=read(OUT/'batch/status.json')
    old=read(Path(check['archive'])/'batches/batch_000/contract.json')
    if batch['stage']!='complete' or batch['policy_steps']!=500 or batch['physics_steps']!=672*500*100:
        raise ValueError('Incomplete physical batch')
    fields=('settings','identity','origin','bodies','terrain','randomization_readback','delay_steps','physics_dt','lower_dt','policy_dt','backend')
    differences=[k for k in fields if meta[k]!=old[k]]
    if differences:
        raise ValueError(f'Runtime differs from matched archive: {differences}')
    for c in read(OUT/'cases.json'):
        p=Path(c['output'])
        assert read(p/'status.json')['policy_steps']==500 and (p/'trajectory.npz').stat().st_size>0
    timing.update(stage='physics_complete',lanes=672,cases=112,runtime_match=list(fields),sampling_s=batch['elapsed_s'],
                  export_s=batch['elapsed_with_output_s']-batch['elapsed_s'],
                  initialization_and_shutdown_s=wall-batch['elapsed_with_output_s'])
    write(OUT/'timing.json',timing)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare-only',action='store_true')
    mode.add_argument('--execute',action='store_true')
    opts=parser.parse_args()
    if opts.prepare_only:
        prepare()
    else:
        execute()
