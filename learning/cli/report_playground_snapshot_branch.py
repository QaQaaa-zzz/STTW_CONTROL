"""Compare original, same-command restore and counterfactual snapshot suffixes."""
import argparse,csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--evaluation',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
status=json.loads((a.evaluation/'status.json').read_text());cfg=json.loads((a.evaluation/'resolved_config.json').read_text())
rows=status['results'];kinds=sorted({r['skill'] for r in rows});original=next(k for k in kinds if not k.endswith('_replay') and '_memory' not in k);memory=original+'_memory1';branch=original+'_replay';kinds=[original,memory,branch]
index={(r['variant']['root_variant'],r['route'],r['variant']['requested_speed_m_s'],r['skill']):r for r in rows}
keys=sorted({k[:3] for k in index});records=[];outcomes=[];summary={}
panel_rows=(len(keys)+2)//3
figxy,axesxy=plt.subplots(panel_rows,3,figsize=(18,22),layout='constrained');figspeed,axesspeed=plt.subplots(panel_rows,3,figsize=(18,25),layout='constrained')
for key,ax,sv in zip(keys,axesxy.flat,axesspeed.flat):
 traces={k:dict(np.load(a.evaluation/index[*key,k]['case_dir']/'trajectory.npz')) for k in kinds}
 for k,color,style in zip(kinds,['#0072B2','#009E73','#D55E00'],['-','--','-.']):
  z=traces[k];r=index[*key,k];t=z['trace'];q=z['qpos'];vel=z['qvel'];label=k.removeprefix(original).lstrip('_') or 'original';label+=f" out={r['continuation_alpha']:g}"
  ax.plot(q[:,0],q[:,1],color=color,ls=style,lw=1,label=label);ax.scatter(*q[-1,:2],marker='x',s=10,color=color)
  speed=vel[:,0]*np.cos(t[:,6])+vel[:,1]*np.sin(t[:,6]);sv.plot(t[:,0],speed,color=color,ls=style,lw=1,label=label)
  for e in [r['rb_entry_s'],r['handoff_s']]:
   if e is not None:sv.axvline(e,color=color,alpha=.3,lw=.5)
  outcomes.append(dict(root=key[0],route=key[1],speed=key[2],method=k,continuation_alpha=r['continuation_alpha'],goal=r['goal'],termination=r['termination'],far_contact=r['far_contact'],handoff_s=r['handoff_s'],duration_s=r['duration_s'],source=str(a.evaluation/r['case_dir']/'trajectory.npz')))
 for candidate,baseline in [(memory,original),(branch,original),(branch,memory)]:
  z=traces[candidate];b=traces[baseline];r=index[*key,candidate];br=index[*key,baseline];n=min(len(z['trace']),len(b['trace']));zt=z['trace'][:n];bt=b['trace'][:n]
  if not np.array_equal(zt[:,0],bt[:,0]):raise ValueError('Saved physical times differ')
  h=[r['handoff_s'],br['handoff_s']];pre=zt[:,0]<min(h) if all(x is not None for x in h) else np.zeros(n,dtype=bool)
  rec=dict(root=key[0],route=key[1],speed=key[2],candidate=candidate,baseline=baseline,goal=r['goal'],baseline_goal=br['goal'],improved=r['goal'] and not br['goal'],lost=br['goal'] and not r['goal'],pre_handback_samples=int(pre.sum()),handoff_time_difference_s=h[0]-h[1] if all(x is not None for x in h) else None,common_end_s=float(zt[-1,0]))
  for field in ['qpos','qvel','controller_state','rb_state','applied_targets']:
   diff=np.abs(z[field][:n]-b[field][:n]);rec[field+'_full_common_max_difference']=float(np.nanmax(diff));rec[field+'_pre_handback_max_difference']=float(np.nanmax(diff[pre])) if pre.any() else None
  anchor=int(np.argmin(abs(zt[:,0]-cfg['snapshot_replay_s'])))
  rec['anchor_qpos_max_difference']=float(abs(z['qpos'][anchor]-b['qpos'][anchor]).max());rec['anchor_qvel_max_difference']=float(abs(z['qvel'][anchor]-b['qvel'][anchor]).max())
  rec['root_xyz_pre_handback_max_difference_m']=float(abs(z['qpos'][:n,:3]-b['qpos'][:n,:3])[pre].max()) if pre.any() else None;records.append(rec)
 ref=traces[original]['reference_xy'];ax.plot(ref[:,0],ref[:,1],':',color='.4',label='Original reference');ax.set_aspect('equal',adjustable='box');ax.set_xlim(1.3,21.5);ax.set_ylim(-1.4,1.4);ax.set_xlabel('World x (m)');ax.set_ylabel('World y (m)');ax.axvspan(10.3,10.6,color='.8',alpha=.4)
 sv.axhline(key[2],color='.4',ls=':',lw=.8,label='Requested');sv.set_xlim(0,10);sv.set_xlabel('Time (s)');sv.set_ylabel('Planar forward speed (m/s)')
 for panel in [ax,sv]:panel.set_title(f'root{key[0]} route{key[1]} v{key[2]} | '+ '/'.join(index[*key,k]['termination'] for k in kinds),fontsize=7);panel.grid(alpha=.2)
for name,fig,axes in [('xy',figxy,axesxy),('speed',figspeed,axesspeed)]:
 axes.flat[0].legend(fontsize=6);fig.suptitle(f"Known DEV: all{len(keys)} conditions, frozen ground/jump skills; anchor {cfg['snapshot_replay_s']}s\nOriginal and memory restore: out{cfg['ground_handoff']['continuation']['alpha']:g}; serialized branch: out{cfg['snapshot_branch_continuation_alpha']:g} applied at native handback. True endpoints, fixed original reference.",fontsize=11);fig.savefig(a.output/f'{name}.png',dpi=150);plt.close(fig)
for name,data in [('paired',records),('cases',outcomes)]:
 with (a.output/f'{name}.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=data[0],lineterminator='\n');w.writeheader();w.writerows(data)
for k in kinds:
 rr=[r for r in outcomes if r['method']==k];summary[k]=dict(cases=len(rr),goals=sum(r['goal'] for r in rr),far_contact=sum(r['far_contact'] for r in rr),handoffs=sum(r['handoff_s'] is not None for r in rr),failures={t:sum(r['termination']==t for r in rr) for t in sorted({r['termination'] for r in rr if not r['goal']})})
(a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary))

comparisons=[]
for candidate,baseline in sorted({(r['candidate'],r['baseline']) for r in records}):
 rr=[r for r in records if (r['candidate'],r['baseline'])==(candidate,baseline)]
 comparisons.append(dict(candidate=candidate,baseline=baseline,cases=len(rr),improved=sum(r['improved'] for r in rr),lost=sum(r['lost'] for r in rr),**{k:max(abs(r[k]) for r in rr if r[k] is not None) for k in rr[0] if k.endswith('difference') or k.endswith('difference_m') or k=='handoff_time_difference_s'}))
(a.output/'comparison_summary.json').write_text(json.dumps(comparisons,indent=2)+'\n')
