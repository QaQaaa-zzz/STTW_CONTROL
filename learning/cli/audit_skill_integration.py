"""Measure joint integration error with true velocities at 20ms versus 5ms."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=False,parents=True)
manifest=json.loads((a.data/'manifest.json').read_text());z=np.load(a.data/'sequences.npz');schema=json.loads((Path(manifest['rows'][0]['evaluation'])/'model_state_schema.json').read_text());qi=[schema['qpos_names'].index(n) for n in ['hip_joint','knee_joint']];vi=[schema['qvel_names'].index(n) for n in ['hip_joint','knee_joint']];dt=schema['model_dt_s'];raw_dt=schema['lower_dt_s'];stride=round(dt/raw_dt);rows=[]
for row in manifest['rows']:
 sl=slice(row['start'],row['start']+row['length']);q=z['qpos'][sl][:,qi];v=z['qvel'][sl][:,vi];k=int(np.flatnonzero(z['phase'][sl]==2)[0]);integrated=q[0]+np.sum(dt/2*(v[:k]+v[1:k+1]),axis=0)
 raw=np.load(row['path']);start=row['raw_start_step'];end=start+k*stride;rq=raw['qpos'][start:end+1][:,qi];rv=raw['qvel'][start:end+1][:,vi];fine=rq[0]+np.sum(raw_dt/2*(rv[:-1]+rv[1:]),axis=0)
 rows.append(dict(split=row['split'],map=row['map'],case=row['case_dir'],handoff_sample=k,coarse_error_rad=float(abs(integrated-q[k]).max()),fine_error_rad=float(abs(fine-rq[-1]).max()),matched_endpoint_error_rad=float(abs(q[k]-rq[-1]).max())))
(a.output/'cases.json').write_text(json.dumps(rows,indent=2)+'\n');summary={}
fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
for ax,split in zip(axes,['TRAIN','DEV']):
 selected=[x for x in rows if x['split']==split];coarse=np.array([x['coarse_error_rad'] for x in selected]);fine=np.array([x['fine_error_rad'] for x in selected]);order=np.arange(len(selected))
 ax.scatter(order,coarse,s=12,marker='x',label='20ms true-velocity integration');ax.scatter(order,fine,s=9,marker='o',label='5ms true-velocity integration');ax.axhline(.01,color='black',ls='--',label='Frozen joint tolerance');ax.set(xlabel='Declared condition index',ylabel='Max hip/knee endpoint error (rad)',title=f'{split}: {len(selected)} conditions',ylim=(0,None));ax.legend(fontsize=7)
 summary[split]=dict(cases=len(selected),coarse_median_rad=float(np.median(coarse)),coarse_max_rad=float(coarse.max()),coarse_outside=int(sum(coarse>=.01)),fine_median_rad=float(np.median(fine)),fine_max_rad=float(fine.max()),fine_outside=int(sum(fine>=.01)))
fig.savefig(a.output/'integration_error.png',dpi=160);plt.close(fig)
summary['scope']='Actual velocities, common phase2 sampled endpoint. Integrator representational diagnostic, not learned rollout or causal deployment. Shared roots are not independent samples; no smoothing.'
(a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary))
