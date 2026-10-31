"""Compare one lower candidate against authenticated cached and missing parent arms."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('candidate',type=Path);p.add_argument('parent_missing',type=Path);p.add_argument('parent_cached',type=Path);p.add_argument('output',type=Path)
a=p.parse_args();roots=[a.candidate,a.parent_missing,a.parent_cached];plans=[];tables=[]
sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
key=lambda row:(row['bank_index'],row['route'],row['alpha'],row['force_N'])
for root in roots:
 plan=json.loads((root/'plan.json').read_text());receipt=json.loads((root/'analysis/receipt.json').read_text())
 assert receipt['plan_sha256']==sha(root/'plan.json')
 assert json.loads((root/'evaluation/status.json').read_text())['phase']=='completed'
 rows=json.loads((root/'analysis/results.json').read_text());table={key(r):r for r in rows};assert len(table)==len(rows)
 plans.append(plan);tables.append(table)
for plan in plans[1:]:
 for field in ('config','models','replanning','horizon_s','goal_radius_m'):assert plan[field]==plans[0][field],field
 for field in ('reconnection','duration_s','force_frame','bank_state_fingerprints'):assert plan['protocol'][field]==plans[0]['protocol'][field],field
 assert {k:v for k,v in plan['initial_source'].items() if k!='candidate'}=={k:v for k,v in plans[0]['initial_source'].items() if k!='candidate'}
assert plans[1]['initial_source']==plans[2]['initial_source']
assert plans[0]['lower_transfer']['parent_checkpoint_sha256']==plans[1]['initial_source']['candidate']['checkpoint_sha256']
parent={k:(r,roots[i]) for i in (1,2) for k,r in tables[i].items() if k in tables[0]}
assert len(parent)==len(tables[0])==30
assert not (set(tables[1])&set(tables[2]))
for plan in plans[1:]:
 contexts={c['route']:c for c in plan['contexts']}
 for c in plans[0]['contexts']:assert c==contexts[c['route']]
candidate_update=plans[0]['initial_source']['candidate']['source_update'];parent_update=plans[1]['initial_source']['candidate']['source_update']
a.output.mkdir(parents=True,exist_ok=False);details=[];figures=[]
conditions=sorted({(k[0],k[1],k[3]) for k in tables[0]});alphas=sorted({k[2] for k in tables[0]})
colors=['#0072B2','#E69F00','#009E73','#D55E00','#CC79A7']
for index,(bank,route,force) in enumerate(conditions):
 fig,axes=plt.subplots(1,3,figsize=(17,5),layout='constrained')
 first=tables[0][bank,route,alphas[0],force]
 with np.load(roots[0]/'evaluation'/(first['path_key']+'_path.npz')) as z:path=dict(z)
 axes[0].plot(*path['xy'][path['s']<=path['goal']].T,'k:',label='Original reference')
 axes[0].scatter(*first['original_goal_xy'],c='k',marker='*',s=55)
 for alpha,color in zip(alphas,colors):
  k=(bank,route,alpha,force);candidate=tables[0][k];old,oldroot=parent[k]
  for field in ('speed','onset_s','duration_s','original_goal_xy'):assert candidate[field]==old[field],field
  with np.load(oldroot/'evaluation'/(old['path_key']+'_path.npz')) as z:
   for field in ('s','xy','heading','curvature','goal'):np.testing.assert_allclose(z[field],path[field],rtol=1e-6,atol=1e-6)
  pair=dict(bank=bank,route=route,alpha=alpha,force_N=force)
  for label,row,style in [('candidate',candidate,'-'),('parent',old,'--')]:
   assert sha(row['raw_file'])==row['raw_sha256']
   with np.load(row['raw_file']) as z:
    active=z['active_tick'].astype(bool);xy=z['actual_xy'][active];t=z['time'][active];v=z['actual_forward_speed'][active];roll=np.rad2deg(z['phi'][active])
   end=row['endpoint'];joint=end['goal_arrived'] and end['physical_outcome']=='survived'
   pair[label]=dict(endpoint=end,joint=joint,raw_file=row['raw_file'],raw_sha256=row['raw_sha256'],actual_turn_speed=row['actual_turn_speed'],lower_reference_tracking=row['lower_reference_tracking'],original_user_three_running_costs=row['original_user_three_running_costs'])
   axes[0].plot(*xy.T,c=color,ls=style,label=f'{label} a={alpha:g}');axes[0].scatter(*xy[-1],c=color,marker='s' if end['physical_outcome']=='survived' else 'x',s=18)
   axes[1].plot(t,v,c=color,ls=style);axes[2].plot(t,roll,c=color,ls=style)
  pair['joint_change']=int(pair['candidate']['joint'])-int(pair['parent']['joint']);details.append(pair)
 axes[0].set(xlabel='World X [m]',ylabel='World Y [m]');axes[0].set_aspect('equal',adjustable='datalim');axes[0].legend(fontsize=7)
 axes[1].set(xlabel='Task time [s]',ylabel='True forward speed [m/s]');axes[2].set(xlabel='Task time [s]',ylabel='Roll [deg]')
 for ax in axes:ax.grid(alpha=.2)
 if force:
  for ax in axes[1:]:ax.axvspan(first['onset_s'],first['onset_s']+first['duration_s'],color='gray',alpha=.15)
 fig.suptitle(f'{route} | bank{bank} | worldY {force:+g} N | middle {first["model_id"]} + candidate u{candidate_update} solid / parent u{parent_update} dashed | H10')
 name=f'case_{index:02d}_xy.png';fig.savefig(a.output/name,dpi=140);plt.close(fig);figures.append(name)
summary={}
for label in ('candidate','parent'):
 ends=[d[label]['endpoint'] for d in details]
 summary[label]=dict(n=len(ends),arrived=sum(e['goal_arrived'] for e in ends),survived=sum(e['physical_outcome']=='survived' for e in ends),joint=sum(d[label]['joint'] for d in details))
changes=dict(gained=sum(d['joint_change']>0 for d in details),lost=sum(d['joint_change']<0 for d in details))
(a.output/'comparison.json').write_text(json.dumps(dict(summary=summary,paired_changes=changes,conditions=details),indent=2)+'\n')
lines=['# Lower continuation paired development comparison','',f'Candidate update{candidate_update}; parent update{parent_update}. Same middle network, original target, physics and prepared state; all five alpha values.','']
for name in figures:lines+=['![]('+name+')','']
lines+=['| Model | Arrived | Survived | Both |','|---|---:|---:|---:|']
for label,s in summary.items():lines.append(f"| {label} | {s['arrived']}/30 | {s['survived']}/30 | {s['joint']}/30 |")
lines+=['',f"Paired joint gains: {changes['gained']}; losses: {changes['lost']}.",'','Arrival is entry into the original 1 m goal disk; survival covers H10. No heading/speed/hold arrival requirement. These previously inspected routes are development evidence. Running costs are retained per alpha and observed window; different-alpha weighted scores are not a physical ranking.','', '[Full numeric comparison](comparison.json)']
(a.output/'README.md').write_text('\n'.join(lines)+'\n');print(json.dumps(dict(summary=summary,paired_changes=changes)))
