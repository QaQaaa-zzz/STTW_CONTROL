"""Audit same-slot incoming prefixes and changed downstream route outcomes."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);a=p.parse_args()
variants=['straight','left','right'];data={};cases={};protocol={};rows=[];wall=steps=0
for name in variants:
 d=a.run/name/'evaluation';data[name]=np.load(d/'trajectories.npz');summary=json.loads((d/'summary.json').read_text());cases[name]=summary['cases'];protocol[name]=json.loads((d.parent/'protocol.json').read_text());wall+=summary['elapsed_s'];steps+=summary['physics_lane_steps']
for name in variants:
 z=data[name]
 for c in cases[name]:
  i=c['lane'];hand=c['handoff_s'];root=z['event_root_xyzw'][i,3];k=round(hand/.02) if hand is not None else None
  observed=z['valid'][:,i];t=z['time'][observed];xy=z['root_xyzw'][observed,i,:2]
  rows.append(dict(c,variant=name,source_root=protocol[name]['cases'][i]['source_root'],handoff_root=root.tolist() if hand is not None else None,prior_road=bool(z['event_road_violation'][i,3]) if hand is not None else None,posthandoff_s=float(t[-1]-hand) if hand is not None else None,last_xy=xy[-1].tolist(),post_route_id=int(z['reference_route_id'][k+1,i]) if k is not None and k+1<len(t) else None))
pairs=[]
for name in variants[1:]:
 for i in range(len(cases[name])):
  l,r=cases['straight'][i],cases[name][i];z,w=data['straight'],data[name];lh,rh=l['handoff_s'],r['handoff_s'];both=lh is not None and rh is not None;same_time=both and lh==rh
  checks={};distance=None
  if same_time:
   k=round(lh/.02);distance=float(np.linalg.norm(z['event_root_xyzw'][i,3]-w['event_root_xyzw'][i,3]))
   for key in ['root_xyzw','observation','controller_history','action','reference_route_id']:
    checks[key]=bool(np.array_equal(z[key][:k,i],w[key][:k,i]))
   ticks=round(lh/.0025)
   for key in ['physics_root_xyzw','physics_contact_mode']:
    checks[key]=bool(np.array_equal(z[key][:ticks,i],w[key][:ticks,i]))
  pairs.append(dict(lane=i,variant=name,both_handoff=both,handoff_time_equal=bool(same_time),handoff_root_l2=distance,prefix_exact=checks,recorded_prefix_match=bool(same_time and distance==0 and all(checks.values())),straight_success=l['success'],variant_success=r['success']))
fig,axs=plt.subplots(2,4,figsize=(15,7),layout='constrained');rfig,rax=plt.subplots(2,4,figsize=(15,7),layout='constrained');bank=np.load(a.run/'roads.npz')['xy']
cfig,cax=plt.subplots(2,4,figsize=(15,7),layout='constrained')
colors=dict(straight='black',left='tab:blue',right='tab:orange')
for i,ax in enumerate(axs.flat):
 for name in variants:
  z=data[name];v=z['valid'][:,i];xy=z['root_xyzw'][v,i,:2];c=cases[name][i];ax.plot(xy[:,0],xy[:,1],color=colors[name],label=f"{name}: {'goal' if c['success'] else 'fail'}");ax.scatter(*xy[-1],color=colors[name],marker='o' if c['success'] else 'x',s=20)
  ref=bank[protocol[name]['downstream_route_id']];ax.plot(ref[:,0],ref[:,1],color=colors[name],ls='--',alpha=.4,lw=.7)
  rax.flat[i].plot(z['time'][v],z['reward'][v,i],color=colors[name],label=name)
  cax.flat[i].plot(z['time'][v],np.cumsum(z['reward'][v,i]),color=colors[name],label=name)
 ax.set(xlim=(8,19),ylim=(-2.5,2.5),xlabel='world x (m)',ylabel='world y (m)',title=f"Source {i},original root {protocol['straight']['cases'][i]['source_root']}");ax.set_aspect('equal');ax.grid(alpha=.2);ax.legend(fontsize=7)
 cax.flat[i].set(xlabel='time (s)',ylabel='Cumulative task reward',title=f'Source {i}');cax.flat[i].grid(alpha=.2)
 rax.flat[i].set(xlabel='time (s)',ylabel='Per-step task reward',title=f'Source {i}');rax.flat[i].grid(alpha=.2)
fig.suptitle('All8 sources; actual solid / prescribed downstream dashed; true endpoints. No physical state restore.')
fig.savefig(a.run/'xy_all.png',dpi=160);plt.close(fig);rfig.savefig(a.run/'reward_all.png',dpi=160);plt.close(rfig)
cfig.savefig(a.run/'cumulative_reward_all.png',dpi=160);plt.close(cfig)
groups=[]
for name in variants:
 rr=[r for r in rows if r['variant']==name];groups.append(dict(variant=name,n=len(rr),success=sum(r['success'] for r in rr),handoffs=sum(r['handoff_s'] is not None for r in rr),valid_handoffs=sum(r['handoff_s'] is not None and not r['prior_road'] for r in rr),road_violations=sum(r['road_violation'] for r in rr)))
result=dict(groups=groups,cases=rows,pairs=pairs,elapsed_s=wall,physics_lane_steps=steps,scope='Retrospective8branch/4root mechanism diagnostic;same-slot prefix reruns;recorded equality is not hidden solver serialization')
(a.run/'results.json').write_text(json.dumps(result,indent=2))
lines=['# Posthandoff route intervention','','![All actual/reference XY](xy_all.png)','','![Per-step rewards](reward_all.png)','','![Cumulative rewards](cumulative_reward_all.png)','','Full-prefix same-slot reruns of8historical branches from4underlying roots. All branches retained. Downstream reference switches only at measured handoff; speed1.5m/s,alpha0.5,cold initialization,goal,terrain and0.6m corridor half-width remain fixed. Left/right excursions reach +/-0.6m between x12 and16 and return to the same goal. No root-only state restore.','','|Variant|Goal/N|Handoffs|Valid handoffs|Road violations|','|---|---:|---:|---:|---:|']
for g in groups:lines.append(f"|{g['variant']}|{g['success']}/{g['n']}|{g['handoffs']}|{g['valid_handoffs']}|{g['road_violations']}|")
lines+=['','|Source|Variant|Both handoff|Recorded prefix exact|Handoff root L2|Straight goal|Variant goal|','|---|---|---|---|---:|---|---|']
for x in pairs:lines.append(f"|{x['lane']}|{x['variant']}|{x['both_handoff']}|{x['recorded_prefix_match']}|{x['handoff_root_l2']}|{x['straight_success']}|{x['variant_success']}|")
lines+=['',f'Actual physics lane steps{steps};summed wall time{wall:.2f}s;0training.','', 'Only pairs with equal recorded incoming prefixes and handoff states support the intended interface comparison. Hidden contact-solver state is not serialized. No-handoff or prior-road failures are not downstream recovery failures. These are inspected historical sources,not independent new TEST or a sampled recoverability boundary. Numerical equalities cover root,observation,ground history,action,reference index,physics roots and contact modes up to the switch.','', 'Raw complete trajectories and frozen config/protocol remain in straight/left/right; results.json retains per-case failures and prefix audit.']
(a.run/'RESULTS.md').write_text('\n'.join(lines)+'\n');print(json.dumps(dict(groups=groups,pairs=pairs,elapsed_s=wall),indent=2))
