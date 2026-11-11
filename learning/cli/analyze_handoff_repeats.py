"""Compare same-lane cold repeats and measured-observer landing initialization."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
names=['cold_a','cold_b','measured'];runs={}
for name in names:
 folder=a.run/name;runs[name]=dict(summary=json.loads((folder/'evaluation/summary.json').read_text()),protocol=json.loads((folder/'protocol.json').read_text()),data=np.load(folder/'evaluation/trajectories.npz'),config=json.loads((folder/'config.json').read_text()))
rows=[];fig,axes=plt.subplots(4,2,figsize=(12,10),layout='constrained')
for lane in range(len(runs['cold_a']['summary']['cases'])):
 row=dict(lane=lane,source_root=runs['cold_a']['protocol']['cases'][lane]['source_root'],candidate=runs['cold_a']['protocol']['cases'][lane]['candidate'],cases={},comparisons={})
 panel,pa=plt.subplots(4,1,figsize=(10,10),layout='constrained')
 for name,color,style in zip(names,['#0072B2','#009E73','#D55E00'],['-','--','-.']):
  run=runs[name];d=run['data'];c=run['summary']['cases'][lane];row['cases'][name]=c;ids=np.flatnonzero(d['valid'][:,lane]);xy=d['root_xyzw'][ids,lane,:2];label=f'{name}: {"goal" if c["success"] else "end "+str(c["end_code"])}'
  for ax in [axes.flat[lane],pa[0]]:
   ax.plot(xy[:,0],xy[:,1],style,color=color,label=label);ax.scatter(*xy[-1],color=color,marker='o' if c['success'] else 'x',s=20)
  pa[1].plot(d['time'][ids],d['reward'][ids,lane],style,color=color,label=name)
  pa[2].plot(d['time'][ids],np.cumsum(d['reward'][ids,lane]),style,color=color,label=name)
  pa[3].plot(d['time'][ids],d['root_xyzw'][ids,lane,7],style,color=color,label=name)
  if c['handoff_s'] is not None:
   for ax in pa[1:]:ax.axvline(c['handoff_s'],color=color,ls=':',alpha=.6)
 config=runs['cold_a']['config'];bank=np.load(config['task_reference_path'])['xy']
 records=json.loads((a.run/'cold_a/evaluation/moving_planning.json').read_text())['records'];record=next((r for r in records if r['lane']==lane and r['status']=='published'),None)
 initial=bank[config['ground_handoff']['reference_route_ids'][lane]]
 if record is not None:
  chosen=bank[record['selected']];x=record['published_root'][0];road=np.concatenate((initial[initial[:,0]<=x],chosen[chosen[:,0]>x]))
 else:road=initial
 for ax in [axes.flat[lane],pa[0]]:
  ax.plot(road[:,0],road[:,1],':',color='black',alpha=.5,label='Original road');ax.set_aspect('equal',adjustable='datalim');ax.set(xlabel='World x (m)',ylabel='World y (m)');ax.legend(fontsize=7);ax.grid(alpha=.2)
 axes.flat[lane].set_title(f'Source {lane}: '+row['candidate'],fontsize=9)
 pa[1].set(xlabel='Time (s)',ylabel='Per-step task reward');pa[2].set(xlabel='Time (s)',ylabel='Cumulative task reward');pa[3].set(xlabel='Time (s)',ylabel='World vx (m/s)')
 panel.suptitle(f'Source {lane}, original root {row["source_root"]}: '+row['candidate']+'\nDotted vertical lines: actual handoff. True endpoints retained.',fontsize=10);panel.savefig(a.output/f'pair_{lane:02}.png',dpi=150);plt.close(panel)
 for other in ['cold_b','measured']:
  first,second=runs['cold_a'],runs[other];c1,c2=first['summary']['cases'][lane],second['summary']['cases'][lane];d1,d2=first['data'],second['data'];h1,h2=c1['handoff_s'],c2['handoff_s'];same_handoff_time=h1 is not None and h2 is not None and h1==h2
  result=dict(both_handoff=h1 is not None and h2 is not None,same_handoff_time=same_handoff_time,success_changed=c1['success']!=c2['success'])
  if result['both_handoff']:
   result['handoff_position_distance_m']=float(np.linalg.norm(d1['event_root_xyzw'][lane,3,:3]-d2['event_root_xyzw'][lane,3,:3]))
   result['handoff_root_equal']=bool(np.array_equal(d1['event_root_xyzw'][lane,3],d2['event_root_xyzw'][lane,3]))
   length=min(np.count_nonzero(d1['time']<=min(h1,h2)+1e-7),np.count_nonzero(d2['time']<=min(h1,h2)+1e-7));fields={}
   for field in ['root_xyzw','observation','action','controller_history','road_violation_history']:
    u,v=d1[field][:length,lane],d2[field][:length,lane];fields[field]=dict(equal=bool(np.array_equal(u,v)),max_abs_difference=float(np.max(np.abs(u.astype(float)-v.astype(float)))) if u.size else None)
   result['prefix_fields']=fields;result['recorded_prefix_equal']=same_handoff_time and result['handoff_root_equal'] and all(v['equal'] for v in fields.values())
  row['comparisons'][other]=result
 rows.append(row)
fig.suptitle('Same-lane prefix repeats and observer initialization | all 8 sources / 4 original roots',fontsize=12);fig.savefig(a.output/'xy_all.png',dpi=180);plt.close(fig)
aggregate={name:dict(n=len(run['summary']['cases']),handoffs=sum(c['handoff_s'] is not None for c in run['summary']['cases']),success=sum(c['success'] for c in run['summary']['cases']),elapsed_s=run['summary']['elapsed_s'],physics_lane_steps=run['summary']['physics_lane_steps']) for name,run in runs.items()}
result=dict(groups=aggregate,cases=rows,scope='Fresh-process same-lane replay diagnostic. Exact recorded prefix comparisons do not serialize hidden solver state. Historical handoff-conditioned DEV cohort;4 original roots.')
(a.output/'results.json').write_text(json.dumps(result,indent=2)+'\n')
lines=['# Same-lane landing initialization comparison','','![All actual/reference XY trajectories](xy_all.png)','','|Run|Handoffs|Complete task|Wall time (s)|','|---|---:|---:|---:|']
for name,g in aggregate.items():lines.append(f'|{name}|{g["handoffs"]}/{g["n"]}|{g["success"]}/{g["n"]}|{g["elapsed_s"]:.2f}|')
lines+=['','|Source|Cold A/B/measured complete|A/B recorded prefix equal|A/measured recorded prefix equal|','|---|---|---|---|']
for r in rows:lines.append(f'|{r["lane"]}|'+ '/'.join(str(int(r['cases'][n]['success'])) for n in names)+'|'+ '|'.join(str(r['comparisons'][n].get('recorded_prefix_equal','no paired handoff')) for n in ['cold_b','measured'])+'|')
lines+=['',result['scope'],'','All runs use identical lane layout, initial conditions, seed, frozen policies, candidate and timing protocol. Only measured initializes the observer at actual handoff. Cold A/B is the no-treatment repeat control. No new training; current handoff and goal criteria are unchanged. No outcome-selected exclusions or inferred capture labels.','', 'A missing handoff is retained and is not counted as a matched interface experiment. Equal logged prefixes cover observed state, action, controller history, road history and handoff root/time, not all PhysX contact/solver internals. Initialization effects cannot be isolated when the prefixes differ.','','Raw run: `'+str(a.run)+'`. Full pairing and numeric differences: [results.json](results.json).','']
lines += [f'- Source {r["lane"]}: [XY, per-step reward and speed](pair_{r["lane"]:02}.png)' for r in rows]
lines += ['', '[Interpretation and adoption decision](ANALYSIS.md).']
(a.output/'RESULTS.md').write_text('\n'.join(lines)+'\n');print(json.dumps(aggregate,indent=2))
