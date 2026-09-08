#!/usr/bin/env python3
"""Frozen circle-policy disturbance panel; explicit evaluation-only overrides."""
import argparse,json,hashlib
from pathlib import Path
from dataclasses import asdict,replace
import numpy as np
from sttw_control.env import RecoveryEnv,load_config
from sttw_control.network import make_policy_identity,load_policy
from sttw_control.evaluation import evaluate
from sttw_control.disturbance import recovery_metrics


def run(task_path,panel_path,training_run,checkpoint,output,seed=None):
    output.mkdir(parents=True,exist_ok=False)
    panel=json.loads(panel_path.read_text());cfg=load_config(task_path)
    if seed is not None:panel['seed']=seed
    source=json.loads((training_run/'declaration.json').read_text())
    # Only explicit event overrides and added neutral event defaults may differ.
    actual=asdict(cfg)
    for key,value in source['task'].items():
        if actual[key]!=value:raise ValueError('base task differs from frozen training task: '+key)
    nominal=replace(cfg,random_events=None,disturbance_force=0.,disturbance_steer_rate=0.)
    env=RecoveryEnv(nominal)
    expected=make_policy_identity(env.bundle.identity,source['task'],cfg.observation.history_steps)
    policy=load_policy(checkpoint,expected=expected)
    identity={**expected,'checkpoint_sidecar_sha256':hashlib.sha256((checkpoint/'identity.json').read_bytes()).hexdigest(),
              'evaluation_overrides':'event scheduling/waveform/timing/amplitude/frame/application point only; policy and observation unchanged'}
    scenarios=[('nominal',nominal)]
    event=replace(cfg,random_events=None,disturbance_start=panel['start_seconds'],disturbance_duration=panel['duration_seconds'],disturbance_force=0.,disturbance_steer_rate=0.)
    scenarios += [(f'steer_{i}',replace(event,disturbance_steer_rate=x)) for i,x in enumerate(panel.get('steer_rate_pulses',[]))]
    scenarios += [(f'force_{i}',replace(event,disturbance_force=x,disturbance_force_frame='heading_lateral',disturbance_force_point='vehicle_com')) for i,x in enumerate(panel.get('lateral_forces',[]))]
    for case in panel.get('cases',[]):
        scenarios.append((case['name'],replace(event,disturbance_steer_rate=case.get('steer_rate',0.),disturbance_force=case.get('force',0.),disturbance_duration=case['duration'],disturbance_waveform=case.get('waveform','constant'),disturbance_force_frame='heading_lateral',disturbance_force_point='vehicle_com')))
    (output/'declaration.json').write_text(json.dumps({'panel':panel,'checkpoint':str(checkpoint),'policy':identity,'scenarios':{k:asdict(c) for k,c in scenarios},'scope':'frozen policy engineering panel; task and checkpoint training provenance recorded explicitly'},indent=2)+'\n')
    rows=[];nom={}
    for name,c in scenarios:
        for label,p in [('baseline',None),('residual',policy)]:
            path=output/name/label
            summary=evaluate(RecoveryEnv(c),path,seed=panel['seed'],policy=p,policy_identity=identity if p else None)
            trace=dict(np.load(path/'trace.npz'))
            if name=='nominal':nom[label]=trace
            metrics={} if name=='nominal' else recovery_metrics(trace,nom[label],asdict(c),path_tolerance=panel['path_tolerance_m'],extra_tolerance=panel['extra_tolerance_m'],heading_tolerance=panel['heading_tolerance_rad'],hold_seconds=panel['hold_seconds'])
            row={'scenario':name,'policy':label,'summary':summary,'path_recovery':metrics};rows.append(row)
            with (output/'results.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            print(json.dumps(row),flush=True)
    (output/'status.json').write_text(json.dumps({'complete':True,'episodes':len(rows)})+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('task','panel','training-run','checkpoint','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--seed',type=int)
    a=p.parse_args();run(a.task,a.panel,a.training_run,a.checkpoint,a.output,a.seed)
