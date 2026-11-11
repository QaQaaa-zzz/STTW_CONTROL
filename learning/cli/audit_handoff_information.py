"""Grouped feature ablations of the frozen-context handoff outcome dataset."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from sttw_control.handoff_continuation import mounted_features

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--source',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
start=time.monotonic();torch.set_num_threads(2)
contract=dict(question='Does observed handoff dynamics add information beyond time, position, yaw and speed?',
 budget='3 feature sets x3 seeds x200 CPU epochs;0physics/0PPO;stop before30min',
 split='Exact existing grouped TRAIN/DEV;historical source/RB transfer never used for checkpoint selection',
 features=['time_only','geometry_speed_time','geometry_speed_time_dynamics'],
 fit='64-32SiLU BCE Adam lr.001 wd.0001;TRAIN normalization;DEV Brier epoch selection',
 role='Retrospective feature ablation, not causal recovery boundary or new independent TEST',
 source=str(a.source),mount_wxyz=[.7071067811865476,-.7071067811865476,0.,0.])
(a.output/'declaration.json').write_text(json.dumps(contract,indent=2))
z=np.load(a.source/'dataset.npz');x,y=z['x'],z['y'];train,dev=z['train'],z['dev']
meta=json.loads((a.source/'results.json').read_text());rows=meta['cases']
angles=mounted_features(x[:,:13],contract['mount_wxyz'])[:,:3]
# Straight downstream reference: world x/y and heading relative to world x.
geometry=np.column_stack((x[:,0],x[:,1],angles[:,2],x[:,7],x[:,38]))
features=dict(time_only=x[:,38:39],geometry_speed_time=geometry,
 geometry_speed_time_dynamics=np.column_stack((geometry,angles[:,:2],x[:,8:13])))
old=np.load(a.source/'predictions.npz');pred={key:old[key] for key in ['train_constant','root_only','state_history','nearest_neighbor']}
selections={}
for name,raw in features.items():
 mean=raw[train].mean(0);std=raw[train].std(0).clip(.01);xx=torch.tensor((raw-mean)/std,dtype=torch.float32);yy=torch.tensor(y);outputs=[]
 for seed in [7101,7102,7103]:
  torch.manual_seed(seed);net=nn.Sequential(nn.Linear(raw.shape[1],64),nn.SiLU(),nn.Linear(64,32),nn.SiLU(),nn.Linear(32,1));opt=torch.optim.Adam(net.parameters(),lr=.001,weight_decay=.0001);best=float('inf');curve=[]
  for epoch in range(1,201):
   opt.zero_grad();loss=nn.functional.binary_cross_entropy_with_logits(net(xx[train])[:,0],yy[train]);loss.backward();opt.step()
   with torch.no_grad():v=torch.sigmoid(net(xx)[:,0]).numpy()
   score=float(((v[dev]-y[dev])**2).mean());curve.append([epoch,float(loss.detach()),score])
   if score<best:best=score;chosen=epoch;state={k:v.detach().clone() for k,v in net.state_dict().items()}
  net.load_state_dict(state)
  with torch.no_grad():outputs.append(torch.sigmoid(net(xx)[:,0]).numpy())
  torch.save(dict(state_dict=state,input_mean=mean,input_std=std,seed=seed,selected_epoch=chosen,feature_set=name),a.output/f'{name}_{seed}.pt')
  (a.output/f'{name}_{seed}_curve.json').write_text(json.dumps(curve));selections[f'{name}_{seed}']=dict(epoch=chosen,dev_brier=best)
 pred[name]=np.mean(outputs,axis=0)
groups=[]
for name,mask in [('TRAIN',train),('DEV',dev)]+[(ds,np.array([r['dataset']==ds for r in rows])) for ds in ['source240','rb400_240']]:
 groups.append(dict(role=name,n=int(mask.sum()),goals=int(y[mask].sum()),brier={k:float(((v[mask]-y[mask])**2).mean()) for k,v in pred.items()}))
root_groups=[]
for ds,root in sorted({(r['dataset'],r['root']) for r,m in zip(rows,dev) if m}):
 mask=dev & np.array([r['dataset']==ds and r['root']==root for r in rows]);root_groups.append(dict(dataset=ds,root=root,n=int(mask.sum()),brier={k:float(((v[mask]-y[mask])**2).mean()) for k,v in pred.items()}))
np.savez_compressed(a.output/'predictions.npz',**pred,y=y,train=train,dev=dev)
result=dict(contract=contract,groups=groups,dev_roots=root_groups,selection=selections,elapsed_s=time.monotonic()-start)
(a.output/'results.json').write_text(json.dumps(result,indent=2));print(json.dumps(dict(groups=groups,elapsed_s=result['elapsed_s']),indent=2))
