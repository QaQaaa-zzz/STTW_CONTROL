"""Paired stage-alpha physical coverage; no alpha-conditioned model claims."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sttw_control.playground_arx import Reference,states

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();manifest=json.loads((a.run/'manifest.json').read_text());a.output.mkdir(parents=True,exist_ok=True)
entries=manifest.get('existing_entries',[])+manifest['entries']
stage_pairs={(0.,0.)}
for entry in entries:
 ai=entry.get('approach_alpha',entry['alpha']);ao=entry.get('departure_alpha',entry['alpha']);stage_pairs.add((ai,ao))
 if 'diagonal_baseline' in entry:stage_pairs.add((ai,ai))
stage_pairs=sorted(stage_pairs)
colors=dict(zip(stage_pairs,['#0072B2','#D55E00','#009E73','#CC79A7','#E69F00','#56B4E9','#000000','#777777','#882255']));records=[];identities=[];paired=[];traces={}
for reused in manifest['reused']:
 mapid=reused['map'];baseline=Path(reused['evaluation']);sources={(0.,0.):baseline}
 for entry in entries:
  if entry['map']!=mapid:continue
  ai=entry.get('approach_alpha',entry['alpha']);ao=entry.get('departure_alpha',entry['alpha'])
  sources[ai,ao]=Path(entry['output'])
  if 'diagonal_baseline' in entry:sources[ai,ai]=Path(entry['diagonal_baseline'])
 old=json.loads((baseline/'declaration.json').read_text());model_hash=hashlib.sha256((baseline/'terrain.mjb').read_bytes()).hexdigest();original_config=json.loads((baseline/'resolved_config.json').read_text())
 for alpha,source in sources.items():
  declaration=json.loads((source/'declaration.json').read_text());cfg=json.loads((source/'resolved_config.json').read_text());normalized=json.loads(json.dumps(cfg));normalized['role']=original_config['role']
  normalized['ground_handoff']['approach']['alpha']=0.;normalized['ground_handoff']['continuation']['alpha']=0.
  match=dict(config=normalized==original_config,ground_actor=declaration['ground_model']==old['ground_model'],jump_actor=declaration['skills']==old['skills'],terrain=hashlib.sha256((source/'terrain.mjb').read_bytes()).hexdigest()==model_hash)
  if not all(match.values()):raise ValueError(f'{mapid} alpha{alpha}: unmatched baseline {match}')
  identities.append(dict(map=mapid,approach_alpha=alpha[0],departure_alpha=alpha[1],source=str(source),checks=match,terrain_sha256=model_hash))
  schema=json.loads((source/'model_state_schema.json').read_text());bank=dict(np.load(cfg['ground_handoff']['reference_path']));follower=json.loads((Path(cfg['ground_handoff']['bundle'])/'config.json').read_text())['follower']
  rows=json.loads((source/'status.json').read_text())['results']
  for row in rows:
   v=row['variant'];key=(mapid,v['root_variant'],row['route'],v['requested_speed_m_s']);z=np.load(source/row['case_dir']/'trajectory.npz');trace=z['trace'];q=z['qpos'];vel=z['qvel'];pose,xi=states(q,vel,schema);features=Reference(bank,row['route'],follower).features(pose,xi,speed=key[3]);phase=trace[:,-1].astype(int);ground=(phase==0)|(phase==2)
   planar=vel[:,0]*np.cos(trace[:,6])+vel[:,1]*np.sin(trace[:,6]);wheel=trace[:,17]*.07;goal=cfg['ground_handoff']['goal'];inside=(phase==2)&(q[:,0]>=goal['x_min_m'])&(q[:,0]<=goal['x_max_m'])
   gates=dict(y=np.abs(q[:,1])<=goal['abs_y_max_m'],path=trace[:,11]<=goal['path_tolerance_m'],heading=np.abs(features[:,10])<=goal['heading_tolerance_rad'],speed=np.abs(planar-trace[:,18])<=goal['speed_tolerance_m_s'],forward=vel[:,0]>=goal['min_vx_m_s'])
   traces[key,alpha]=dict(time=trace[:,0],xy=q[:,:2],speed=xi[:,0],planar_speed=planar,wheel_speed=wheel,requested=trace[:,18],lateral=features[:,9],heading=features[:,10],phase=phase,reference=z['reference_xy'],events=[row['rb_entry_s'],row['first_far_contact_s'],row['handoff_s']],initial_q=q[0],initial_v=vel[0],qpos=q,qvel=vel,goal=row['goal'])
   records.append(dict(map=mapid,data_split=v['data_split'],root=key[1],route=key[2],speed_request_m_s=key[3],approach_alpha=alpha[0],departure_alpha=alpha[1],goal=row['goal'],termination=row['termination'],failure_stage='complete' if row['goal'] else 'before_RB' if row['rb_entry_s'] is None else 'RB_or_handoff' if row['handoff_s'] is None else 'departure',far_contact=row['far_contact'],handoff_s=row['handoff_s'],duration_s=row['duration_s'],goal_hold_s=row['goal_hold_s'],final_x_m=float(q[-1,0]),final_y_m=float(q[-1,1]),ground_speed_mean_m_s=float(xi[ground,0].mean()),ground_speed_rmse_m_s=float(np.sqrt(np.mean((xi[ground,0]-trace[ground,18])**2))),ground_lateral_rmse_m=float(np.sqrt(np.mean(features[ground,9]**2))),ground_heading_rmse_rad=float(np.sqrt(np.mean(features[ground,10]**2))),max_roll_rad=float(abs(trace[:,4]).max()),goal_x_window_samples=int(inside.sum()),**{f'goal_window_{name}_violations':int((inside&~gate).sum()) for name,gate in gates.items()},source=str(source/row['case_dir']/'trajectory.npz')))
 keys=sorted(k for k,alpha in traces if k[0]==mapid and alpha==(0.,0.))
 comparisons=[(alpha,(0.,0.)) for alpha in colors if alpha!=(0.,0.)]
 comparisons += [(alpha,(alpha[0],alpha[0])) for alpha in colors if alpha[0]!=alpha[1] and alpha[0]!=0]
 comparisons=list(dict.fromkeys(comparisons))
 for key in keys:
  for alpha,base_alpha in comparisons:
   b=traces[key,base_alpha];t=traces[key,alpha];n=min(len(b['time']),len(t['time']));same=(b['phase'][:n]==t['phase'][:n])&np.isin(b['phase'][:n],[0,2]);xy=np.linalg.norm(t['xy'][:n]-b['xy'][:n],axis=1)
   handoffs=[b['events'][2],t['events'][2]];prefix=t['time'][:n]<min(handoffs) if all(h is not None for h in handoffs) else np.zeros(n,dtype=bool)
   paired.append(dict(map=mapid,root=key[1],route=key[2],speed_request_m_s=key[3],approach_alpha=alpha[0],departure_alpha=alpha[1],baseline_approach_alpha=base_alpha[0],baseline_departure_alpha=base_alpha[1],goal=t['goal'],baseline_goal=b['goal'],improved=bool(t['goal'] and not b['goal']),lost=bool(b['goal'] and not t['goal']),common_end_s=float(b['time'][n-1]),initial_q_max_difference=float(abs(t['initial_q']-b['initial_q']).max()),initial_v_max_difference=float(abs(t['initial_v']-b['initial_v']).max()),pre_handback_qpos_max_difference=float(abs(t['qpos'][:n][prefix]-b['qpos'][:n][prefix]).max()) if prefix.any() else None,pre_handback_qvel_max_difference=float(abs(t['qvel'][:n][prefix]-b['qvel'][:n][prefix]).max()) if prefix.any() else None,pre_handback_root_xyz_max_difference_m=float(abs(t['qpos'][:n][prefix,:3]-b['qpos'][:n][prefix,:3]).max()) if prefix.any() else None,pre_RB_root_xyz_max_difference_m=float(abs(t['qpos'][:n][(t['phase'][:n]==0)&(b['phase'][:n]==0),:3]-b['qpos'][:n][(t['phase'][:n]==0)&(b['phase'][:n]==0),:3]).max()),handoff_time_difference_s=handoffs[1]-handoffs[0] if all(h is not None for h in handoffs) else None,xy_rms_difference_m=float(np.sqrt(np.mean(xy**2))),same_ground_samples=int(same.sum()),speed_mean_difference_m_s=float((t['speed'][:n][same]-b['speed'][:n][same]).mean()),lateral_rmse_difference_m=float(np.sqrt(np.mean(t['lateral'][:n][same]**2))-np.sqrt(np.mean(b['lateral'][:n][same]**2)))))
 data_role='/'.join(sorted({x['data_split'] for x in records if x['map']==mapid}))
 for quantity in ['xy','speed','lateral']:
  fig,axes=plt.subplots(12,3,figsize=(18,22 if quantity=='xy' else 26),layout='constrained')
  for key,ax in zip(keys,axes.flat):
   reference=traces[key,(0.,0.)]['reference']
   for alpha,color in colors.items():
    t=traces[key,alpha];label=f'in{alpha[0]:g}/out{alpha[1]:g}'
    if quantity=='xy':
     ax.plot(t['xy'][:,0],t['xy'][:,1],color=color,label=label,lw=1,ls='--' if alpha[0]!=alpha[1] else '-');ax.scatter(*t['xy'][-1],color=color,marker='x',s=10)
    else:
     ax.plot(t['time'],t['speed' if quantity=='speed' else 'lateral'],color=color,label=label,lw=1)
     if quantity=='speed':ax.plot(t['time'],t['wheel_speed'],color=color,ls=':',lw=.65,alpha=.65)
     for event in t['events']:
      if event is not None:ax.axvline(event,color=color,lw=.45,alpha=.35)
   if quantity=='xy':
    ax.plot(reference[:,0],reference[:,1],':',color='.4',lw=.8,label='Original reference');ax.set_aspect('equal',adjustable='box');ax.set_xlim(1.3,21.5);ax.set_ylim(-1.4,1.4);ax.set_xlabel('World x (m)');ax.set_ylabel('World y (m)');ax.axvspan(10.3,10.6,color='.8',alpha=.4)
   else:
    ax.set_xlim(0,10);ax.set_xlabel('Time (s)');ax.set_ylabel('Speed (m/s)' if quantity=='speed' else 'Signed path error (m)');ax.axhline(key[3] if quantity=='speed' else 0,color='.3',ls='--',lw=.8,label='Requested' if quantity=='speed' else None)
   outcomes=[next(x for x in records if (x['map'],x['root'],x['route'],x['speed_request_m_s'],x['approach_alpha'],x['departure_alpha'])==(*key,*alpha))['termination'] for alpha in colors]
   ax.set_title(f'root{key[1]} route{key[2]} v{key[3]} | '+'/'.join({'goal':'G','timeout':'T','tracking_domain_exit':'D'}.get(o,o) for o in outcomes),fontsize=7);ax.grid(alpha=.2);ax.tick_params(labelsize=7)
  axes.flat[0].legend(fontsize=6,ncol=2);fig.suptitle(f'{mapid}: all36 matched conditions x {len(colors)} stage-alpha pairs (in/out); titles G=goal/T=timeout/D=domain exit follow legend order; actual traces end at their true goal/failure/timeout\nFrozen H75+ECBC/ESO and RB225; {data_role}. '+('XY travel +x, equal axes.' if quantity=='xy' else 'Speed solid: actual body-forward; dotted: rear rate x0.07m. Vertical lines: RB entry/contact/handback.' if quantity=='speed' else 'Signed lateral normal=(-sin heading,cos heading); fixed original path.'),fontsize=11);fig.savefig(a.output/f'{mapid}_{quantity}.png',dpi=150);plt.close(fig)
for name,data in [('cases',records),('paired',paired)]:
 with (a.output/(name+'.csv')).open('w') as f:
  w=csv.DictWriter(f,fieldnames=data[0],lineterminator='\n');w.writeheader();w.writerows(data)
summary=[]
for mapid in sorted({x['map'] for x in records}):
 for alpha in colors:
  rr=[x for x in records if x['map']==mapid and (x['approach_alpha'],x['departure_alpha'])==alpha];summary.append(dict(map=mapid,approach_alpha=alpha[0],departure_alpha=alpha[1],cases=len(rr),goals=sum(x['goal'] for x in rr),far_contact=sum(x['far_contact'] for x in rr),handoffs=sum(x['handoff_s'] is not None for x in rr),failures={k:sum(x['termination']==k for x in rr) for k in sorted({x['termination'] for x in rr if not x['goal']})}))
(a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');(a.output/'identity_receipt.json').write_text(json.dumps(identities,indent=2)+'\n')
(a.output/'analysis_contract.json').write_text(json.dumps(dict(source=str(a.run),data_roles=sorted({x['data_split'] for x in records}),scope=manifest.get('scope','Fixed stage alpha execution coverage; roles taken from source variants; not alpha-planning validation'),baseline='Alpha0 frozen H75, not pure ECBC; physical/actor/config identity checked per source before reuse',metrics='Task outcomes from runner. Ground error means use each actual ground window; paired differences use common actual duration and equal phase0/2 samples, no extrapolation after goal/failure. Same-approach comparisons measure raw qpos/qvel maximum differences strictly before the earlier measured handback; mixed component units, not a physical norm. Body-forward3D rotation; controller goal speed is planar heading projection. Lateral normal=(-sin,cos).',goal_diagnostics='Goal-window violation counts are overlapping diagnostic counts; do not replace actual status or attribute a unique cause',no_reward='No reward is defined by this physical coverage protocol; no fabricated reward curves'),indent=2)+'\n');print(json.dumps(summary))
