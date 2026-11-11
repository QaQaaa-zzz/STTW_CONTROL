"""CPU downstream continuation pilot; grouped source720 TRAIN/DEV."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
p=argparse.ArgumentParser();p.add_argument('--runs',type=Path,required=True);p.add_argument('--labels',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
torch.set_num_threads(2);start=time.monotonic()
sources={'source720':'chapter3_offroad_event_training/collection','source1440':'chapter3_offroad_distance_training/collection','source240':'chapter3_offroad_new_roots/source','rb400_240':'chapter3_offroad_new_roots/rb400'}
declaration=dict(input='39 values: physical root13 + latest measured observation frame25 + handoff time1. All available before switching. Fixed cold initialization,1.5m/s,alpha0.5,straight downstream. Not a complete simulator state.',target='Previously frozen observed2s continuation; no censored negatives',split='source720 original TRAIN roots fit; DEV roots select epochs; other batches historical transfer only',budget='CPU,3 seeds x2 feature sets x200 epochs;0PPO/0physics;25min wall cap',model='MLP39-64-32-1SiLU; BCE Adam lr.001 weight_decay.0001; TRAIN normalization; DEV Brier selection',scope='Exploratory downstream interface only; no independent TEST/planning benefit or disturbance margin')
(a.output/'declaration.json').write_text(json.dumps(declaration,indent=2))
labels=json.loads(a.labels.read_text())['cases'];rows=[];inputs=[];targets=[];alignment=0.
for name,rel in sources.items():
 run=a.runs/rel;z=np.load(run/'evaluation/trajectories.npz');pr=json.loads((run/'protocol.json').read_text())['cases']
 event=z['event_root_xyzw'];obs=z['observation'];roots=z['root_xyzw'];times=z['event_time'][:,3]
 for row in [r for r in labels if r['dataset']==name and r['continued_2s'] is not None]:
  i=row['lane'];step=round(float(times[i])/.02);error=float(abs(roots[step-1,i]-event[i,3]).max());alignment=max(alignment,error);assert error<1e-5,(name,i,error)
  inputs.append(np.r_[event[i,3],obs[step,i,50:75],times[i]])
  targets.append(float(row['continued_2s']));rows.append(dict(row,fit_role=pr[i]['role'] if name=='source720' else 'HISTORICAL_TRANSFER',input_step=step))
x=np.array(inputs,np.float32);y=np.array(targets,np.float32)
train=np.array([r['dataset']=='source720' and r['fit_role']=='TRAIN' for r in rows]);dev=np.array([r['dataset']=='source720' and r['fit_role']=='DEV' for r in rows]);assert train.any() and dev.any()
tr={r['root'] for r,m in zip(rows,train) if m};dr={r['root'] for r,m in zip(rows,dev) if m};assert not tr&dr
mean=x[train].mean(0);std=x[train].std(0).clip(.01);xn=(x-mean)/std
np.savez_compressed(a.output/'dataset.npz',x=x,y=y,train=train,dev=dev,mean=mean,std=std)
pred={'train_constant':np.full(len(y),y[train].mean())};dist=((xn[:,None]-xn[train][None])**2).sum(-1);pred['nearest_neighbor']=y[train][dist.argmin(1)];selection={}
for name,cols in [('state_history',list(range(39))),('root_only',list(range(13))+[38])]:
 values=torch.tensor(xn[:,cols]);truth=torch.tensor(y);outputs=[]
 for seed in [7101,7102,7103]:
  torch.manual_seed(seed);net=nn.Sequential(nn.Linear(len(cols),64),nn.SiLU(),nn.Linear(64,32),nn.SiLU(),nn.Linear(32,1));opt=torch.optim.Adam(net.parameters(),lr=.001,weight_decay=.0001);best=float('inf');curve=[]
  for epoch in range(1,201):
   opt.zero_grad();loss=nn.functional.binary_cross_entropy_with_logits(net(values[train]).squeeze(-1),truth[train]);loss.backward();opt.step()
   with torch.no_grad():v=torch.sigmoid(net(values).squeeze(-1)).numpy()
   score=float(((v[dev]-y[dev])**2).mean());curve.append([epoch,float(loss.detach()),score])
   if score<best:best=score;state={k:v.detach().clone() for k,v in net.state_dict().items()};chosen=epoch
  net.load_state_dict(state)
  with torch.no_grad():outputs.append(torch.sigmoid(net(values).squeeze(-1)).numpy())
  torch.save(dict(state_dict=state,columns=cols,mean=mean,std=std,selected_epoch=chosen,seed=seed),a.output/f'{name}_{seed}.pt')
  selection[f'{name}_{seed}']=dict(epoch=chosen,dev_brier=best)
  (a.output/f'{name}_{seed}_curve.json').write_text(json.dumps(curve))
 pred[name]=np.mean(outputs,axis=0);pred[name+'_std']=np.std(outputs,axis=0)
groups=[]
for name,mask in [('source720_TRAIN',train),('source720_DEV',dev)]+[(n,np.array([r['dataset']==n for r in rows])) for n in sources if n!='source720']:
 groups.append(dict(dataset=name,n=int(mask.sum()),positive=int(y[mask].sum()),brier={k:float(((v[mask]-y[mask])**2).mean()) for k,v in pred.items() if not k.endswith('_std')}))
np.savez_compressed(a.output/'predictions.npz',**pred,y=y)
result=dict(groups=groups,selection=selection,train_roots=sorted(tr),dev_roots=sorted(dr),max_handoff_alignment_error=alignment,elapsed_s=time.monotonic()-start,cases=rows,scope=declaration)
(a.output/'results.json').write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k not in ('cases','scope','selection')},indent=2))
