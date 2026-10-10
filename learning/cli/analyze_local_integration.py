#!/usr/bin/env python3
"""No simulation: existing fixed selection traces, phase score and yaw closure."""
import argparse,sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from sttw_control.local_integration_diagnostics import *
p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--output',required=True);a=p.parse_args()
r=Path(a.run);out=Path(a.output);out.mkdir(parents=True,exist_ok=True);cfg=json.loads((r/'config.json').read_text());result={}
for role,folder in [('300','validation_0300'),('350','validation_0350'),('400','validation_0400'),('R196','R196_reference')]:
    dest=out/role;dest.mkdir(exist_ok=True);cases={}
    original=json.loads((r/'training'/('validation_0300' if role=='R196' else folder)/'metrics.json').read_text())
    if role=='R196':original=original['R196_reference']
    for case in cfg['validation']['case_ids']:
        file=r/'training'/folder/(case+'.npz');d=dict(np.load(file,allow_pickle=False));small=small_window(d,case,cfg['validation']);rows=platform_statistics(d,small)
        # Validation reset explicitly anchors reference yaw to actual pose, epsi0=0.
        yaw0=float(d['reference_yaw_unwrapped'][0])-float(d['omega_c'][0])*DT
        y,arrays=yaw_decomposition(d,initial_yaw=yaw0,initial_error=0.)
        np.savez_compressed(dest/(case+'_yaw.npz'),**arrays)
        err=np.column_stack([d['actual_forward_speed'],d['actual_delta']]).astype(float)-d['limited_command']
        full={name:channel_stats(err[:,c],d['time'],np.ones(len(err),bool),tol) for name,c,tol in [('speed',0,.05),('steer',1,np.where(small,.01,.02))]}
        cases[case]=dict(source=str(file),old_Q=original['cases'][case]['physical_quality_score'],phase_consistent_score_v3=phase_score(d,small),platforms=rows,full=full,yaw_decomposition=y,initial_yaw_source='reset reference_pose=actual pose; first post-reference yaw minus dt*omega_c',small_start_s=float(d['time'][np.flatnonzero(small)[0]]) if small.any() else None,peak_roll=float(max(d['peak_roll'])),physical_failure=bool(np.any(d['physical_failure'])),final_clip_fraction=float(np.mean(d['final_command_clipped'])),force_limit_status='unavailable: no actuator force substep arrays in saved local validation',original_primary_failure=original['cases'][case]['primary_failure'])
    scored=[p['worst_normalized_rmse'] for c in cases.values() for p in c['platforms'] if p['worst_normalized_rmse'] is not None]
    result[role]=dict(old_Q=original['physical_quality_score'],phase_consistent_score_v3=float(np.mean([c['phase_consistent_score_v3'] for c in cases.values()])),worst_platform_normalized_RMSE=max(scored) if scored else None,cases=cases)
(out/'phase_diagnostics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
for role,row in result.items():print(role,row['old_Q'],row['phase_consistent_score_v3'],row['worst_platform_normalized_RMSE'])
print('No new physics. Maximum yaw closure:',max(c['yaw_decomposition']['closure_max_abs_error'] for r in result.values() for c in r['cases'].values()))
