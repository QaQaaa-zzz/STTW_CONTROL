from pathlib import Path
import argparse,json,numpy as np
p=argparse.ArgumentParser(description='Diagnose the requested-speed baseline and observed route-group ranking without new physics');p.add_argument('--base',type=Path,required=True);a=p.parse_args();b=a.base;out=b/'chapter3_interface_diagnosis';out.mkdir(exist_ok=True)
d=np.load(b/'chapter3_interface_all_routes/upstream_distance/dataset.npz');raw=d['x'];target=d['target'];m=d['dev']&d['hit'];tr=d['train']&d['hit'];v=raw[:,451]
print('requested speed unique',np.unique(v))
r=dict(scope='Existing grouped TRAIN/DEV only; route-group rankings use different observed handoffs and are not counterfactual route outcomes',vx_baselines={})
for role,mask in [('TRAIN',tr),('DEV',m)]:
 r['vx_baselines'][role]=dict(n=int(mask.sum()),requested_speed_rmse=float(np.sqrt(np.mean((v[mask]-target[mask,7])**2))),train_mean_rmse=float(np.sqrt(np.mean((target[tr,7].mean()-target[mask,7])**2))),correlation=float(np.corrcoef(v[mask],target[mask,7])[0,1]))
r['vx_by_request']=[dict(speed=float(s),n=int((m&(v==s)).sum()),observed=float(target[m&(v==s),7].mean())) for s in np.unique(v[m])]
model=b/'chapter3_frenet_condition_model';d=np.load(model/'dataset.npz');j=json.loads((model/'results.json').read_text());pred=np.load(model/'predictions.npz');groups={}
for i,row in enumerate(j['cases']):groups.setdefault((row['root'],row['candidate']),[]).append(i)
r['task_ranking']=[]
for role in ['TRAIN','DEV']:
 allgroups=[ix for ix in groups.values() if len(ix)==3 and j['cases'][ix[0]]['role']==role];valid=[ix for ix in allgroups if np.ptp(d['y'][ix])>0];exact=[ix for ix in allgroups if np.array_equal(d['x'][ix[0],:39],d['x'][ix[1],:39]) and np.array_equal(d['x'][ix[0],:39],d['x'][ix[2],:39])]
 r['task_ranking'].append(dict(role=role,groups=len(allgroups),exact_handoff_groups=len(exact),mixed_outcome_groups=len(valid),oracle_goals=sum(bool(d['y'][ix].max()) for ix in allgroups),selection={k:dict(goals=sum(bool(d['y'][ix[np.argmax(pred[k][ix])]]) for ix in allgroups),mixed_goals=sum(bool(d['y'][ix[np.argmax(pred[k][ix])]]) for ix in valid)) for k in pred.files if k!='y'}))
(out/'results.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
