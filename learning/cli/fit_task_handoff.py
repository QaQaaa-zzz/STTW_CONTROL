"""Fit task-conditioned handoff outcomes with grouped root splits and a state-only ablation."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from sttw_control.handoff_task_context import route_context,OFFSETS_M

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--collection',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
p.add_argument('--mount-config',type=Path,default=Path('learning/configs/rb_sttw_platform.json'))
p.add_argument('--context-representation',choices=['world_x','frenet'],default='world_x')
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);start=time.monotonic()
mount=json.loads(a.mount_config.read_text())['mount_rotation_wxyz']
contract=dict(context_representation=a.context_representation,question='Can downstream geometry distinguish full-goal continuation at a measured handoff?',budget='2feature sets x3seeds x200CPUepochs;0PPO/0newphysics;stop before30min',
 inputs='39D measured root/latest frame/time plus19D known downstream route context; no task ID input',context='Legacy world-x preview or19D Frenet: lateral/heading error,tangent/normal velocity,0.4s ballistic left/right margins,6heading/6curvature previews,width; ballistic margins not safety certificates',offsets_m=OFFSETS_M.tolist(),mount_wxyz=mount,
 target='Original full-goal outcome,conditional on handoff without prior road violation; missing/invalid handoffs excluded only from downstream conditional fit',split='Original source720 root split8TRAIN/4DEV; all3task variants of a root kept together; no independent TEST',
 model='64/32SiLU;BCE Adam lr.001 wd.0001;3seeds7101/7102/7103;200epochs;TRAIN normalization;DEV Brier selects epoch',source=str(a.collection),limitations='Fixed physical map,H75/jump/cold initialization/speed1.5/alpha.5;only downstream route changes;not a general viability margin')
(a.output/'declaration.json').write_text(json.dumps(contract,indent=2))
inputs=[];targets=[];rows=[];coverage=[]
bank=np.load(a.collection/'issued.npz')
for variant in ['straight','left','right']:
 run=a.collection/variant;cfg=json.loads((run/'config.json').read_text());pr=json.loads((run/'protocol.json').read_text());summary=json.loads((run/'evaluation/summary.json').read_text());z=np.load(run/'evaluation/trajectories.npz')
 physical=z['root_xyzw'];observations=z['observation'];event_roots=z['event_root_xyzw'];event_road=z['event_road_violation']
 rid=pr['downstream_route_id'];route={k:bank[k][rid] for k in ['xy','heading','curvature']};count=prior=0
 for case in summary['cases']:
  lane=case['lane'];t=case['handoff_s']
  if t is None:continue
  if event_road[lane,3]:prior+=1;continue
  k=round(t/.02);root=event_roots[lane,3];assert np.array_equal(root,physical[k-1,lane])
  context=route_context(root[None],route,cfg['ground_handoff']['road_half_width_m'],mount,representation=a.context_representation)[0]
  inputs.append(np.r_[root,observations[k,lane,50:75],t,context]);targets.append(case['success']);row=pr['cases'][lane]
  rows.append(dict(variant=variant,lane=lane,root=row['root'],role=row['role'],candidate=row['candidate'],success=case['success'],handoff_s=t,end_code=case['end_code']));count+=1
 coverage.append(dict(variant=variant,total=len(summary['cases']),goals=sum(c['success'] for c in summary['cases']),valid_handoffs=count,prior_road_handoffs=prior,physics_lane_steps=summary['physics_lane_steps'],elapsed_s=summary['elapsed_s']))
x=np.array(inputs,np.float32);y=np.array(targets,np.float32);train=np.array([r['role']=='TRAIN' for r in rows]);dev=np.array([r['role']=='DEV' for r in rows]);assert train.any() and dev.any()
assert not {r['root'] for r,m in zip(rows,train) if m}&{r['root'] for r,m in zip(rows,dev) if m}
mean=x[train].mean(0);std=x[train].std(0).clip(.01);xx=torch.tensor((x-mean)/std);yy=torch.tensor(y);selection={};pred={}
np.savez_compressed(a.output/'dataset.npz',x=x,y=y,train=train,dev=dev,mean=mean,std=std)
for name,columns in [('state_only',list(range(39))),('task_context',list(range(58)))]:
 values=[]
 for seed in [7101,7102,7103]:
  torch.manual_seed(seed);net=nn.Sequential(nn.Linear(len(columns),64),nn.SiLU(),nn.Linear(64,32),nn.SiLU(),nn.Linear(32,1));opt=torch.optim.Adam(net.parameters(),lr=.001,weight_decay=.0001);best=float('inf');curve=[]
  for epoch in range(1,201):
   opt.zero_grad();loss=nn.functional.binary_cross_entropy_with_logits(net(xx[train][:,columns])[:,0],yy[train]);loss.backward();opt.step()
   with torch.no_grad():prediction=torch.sigmoid(net(xx[:,columns])[:,0]).numpy()
   score=float(((prediction[dev]-y[dev])**2).mean());curve.append([epoch,float(loss.detach()),score])
   if score<best:best=score;chosen=epoch;state={k:v.detach().clone() for k,v in net.state_dict().items()}
  net.load_state_dict(state)
  with torch.no_grad():values.append(torch.sigmoid(net(xx[:,columns])[:,0]).numpy())
  torch.save(dict(state_dict=state,columns=columns,mean=mean,std=std,selected_epoch=chosen,seed=seed,target='goal',task_context=contract),a.output/f'{name}_{seed}.pt')
  (a.output/f'{name}_{seed}_curve.json').write_text(json.dumps(curve));selection[f'{name}_{seed}']=dict(epoch=chosen,dev_brier=best)
 pred[name]=np.mean(values,axis=0)
pred['constant']=np.full(len(y),y[train].mean());variant=np.array([r['variant'] for r in rows]);pred['task_prior']=np.array([y[train&(variant==v)].mean() for v in variant])
groups=[]
for role,mask in [('TRAIN',train),('DEV',dev)]:
 for v in [None,'straight','left','right']:
  m=mask if v is None else mask&(variant==v)
  groups.append(dict(role=role,variant=v,n=int(m.sum()),goals=int(y[m].sum()),brier={k:float(((p[m]-y[m])**2).mean()) for k,p in pred.items()}))
np.savez_compressed(a.output/'predictions.npz',**pred,y=y)
result=dict(contract=contract,coverage=coverage,groups=groups,cases=rows,selection=selection,elapsed_s=time.monotonic()-start)
(a.output/'results.json').write_text(json.dumps(result,indent=2));print(json.dumps(dict(coverage=coverage,groups=groups,elapsed_s=result['elapsed_s']),indent=2))
