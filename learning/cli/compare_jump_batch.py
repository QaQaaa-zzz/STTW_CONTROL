"""Compare completed batched connection lanes with their saved single-lane runs."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--run',type=Path,required=True)
a=p.parse_args()
batch=json.loads((a.run/'batch_results.json').read_text())
protocol=json.loads((a.run/'protocol.json').read_text());spec=protocol['spec']
fig,axes=plt.subplots(1,2,figsize=(13,6),layout='constrained')
rows=[];single_elapsed=0.
for lane in batch['cases']:
    i=lane['lane'];candidate=lane['candidate'];new=Path(lane['output'])
    reference_config=Path(spec['batch_reference_configs'][i])
    reference_spec=json.loads(reference_config.read_text());old=Path(reference_spec['output'])
    traces=[];summaries=[];durations=[]
    color={0.:'tab:blue',.5:'tab:orange',1.:'tab:green'}[candidate['alpha']]
    style='--' if candidate['speed_m_s']==1.8 else '-'
    for axis,folder in zip(axes,(old,new)):
        d=np.load(folder/'trajectory.npz');trace=d['trace'][d['valid'][:,0],0];traces.append(trace)
        summaries.append(json.loads((folder/'chain_summary.json').read_text()))
        r=json.loads((folder/'results.json').read_text());durations.append(r['cases'][0]['duration_s'])
        axis.plot(trace[:,1],trace[:,2],color=color,ls=style,label=f"v={candidate['speed_m_s']}, alpha={candidate['alpha']}")
        axis.plot(trace[-1,1],trace[-1,2],'x',color=color)
    single_elapsed+=json.loads((old/'results.json').read_text())['elapsed_s']
    x,y=traces;times,ix,iy=np.intersect1d(x[:,0],y[:,0],return_indices=True)
    delta=x[ix]-y[iy];xy=np.linalg.norm(delta[:,1:3],axis=1)
    changed=np.flatnonzero(np.max(abs(delta[:,1:]),axis=1)>1e-6)
    events={}
    for name in ('connection_decision','handoff','landing_ready','landing_handoff'):
        events[name]={label:json.loads((folder/(name+'.json')).read_text())['time'] if (folder/(name+'.json')).exists() else None for label,folder in (('single',old),('batch',new))}
    rows.append(dict(lane=i,candidate=candidate,reference_config=str(reference_config),single_run=str(old),batch_run=str(new),
        single_complete=summaries[0]['complete'],batch_complete=summaries[1]['complete'],
        single_checks=summaries[0]['checks'],batch_checks=summaries[1]['checks'],
        single_T_s=summaries[0]['success_time_s'],batch_T_s=summaries[1]['success_time_s'],
        single_duration_s=durations[0],batch_duration_s=durations[1],single_samples=len(x),batch_samples=len(y),
        matched_time_samples=len(times),matched_time_end_s=float(times[-1]) if len(times) else None,
        xy_rmse_m=float(np.sqrt(np.mean(xy**2))) if len(times) else None,
        xy_max_m=float(xy.max()) if len(times) else None,
        speed_rmse_m_s=float(np.sqrt(np.mean(delta[:,4]**2))) if len(times) else None,
        first_state_difference_over_1e_6_s=float(times[changed[0]]) if len(changed) else None,
        events=events))
route=np.load(a.run/'ground_route.npz')['xy']
for axis,title in zip(axes,('Saved single-lane execution','Mixed-phase GPU batch')):
    axis.plot(route[:,0],route[:,1],':',color='gray',label='Original route')
    axis.set(title=title,xlabel='World x (m)',ylabel='World y (m)',aspect='equal')
    axis.grid(alpha=.2);axis.legend(fontsize=8)
fig.suptitle(f"{spec['origin_id']} | frozen H75 + {spec['skills'][0]} | {1/protocol['physics_dt']:g}/{1/protocol['ground_dt']:g}/{1/protocol['jump_dt']:g} Hz\n{len(rows)} candidates; crosses are observed endpoints; original full-task criteria unchanged")
fig.savefig(a.run/'batch_vs_single_xy.png',dpi=150)
result=dict(role='Same-root engineering batch comparison; not independent task improvement or an isolated hardware benchmark',
    batch_elapsed_s=batch['elapsed_s'],saved_single_elapsed_sum_s=single_elapsed,
    observed_elapsed_ratio=single_elapsed/batch['elapsed_s'],
    differing_observation_windows=sum(abs(r['single_duration_s']-r['batch_duration_s'])>1e-5 for r in rows),
    timing_interpretation='Elapsed ratio only: early failures change workload; no speedup claim without comparable task behavior and windows',
    single_complete_count=sum(r['single_complete'] for r in rows),batch_complete_count=sum(r['batch_complete'] for r in rows),
    matching_full_task_labels=sum(r['single_complete']==r['batch_complete'] for r in rows),cases=rows,
    limits='Matched stored timestamps only; different observed windows remain explicit. Wall times collected under shared-machine load and different initialization scopes. Speed alone does not establish behavioral compatibility.')
(a.run/'batch_comparison.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('cases','limits')}))
