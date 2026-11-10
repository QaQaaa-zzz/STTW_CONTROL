"""Direct task-completion baseline using the same grouped observations and plans."""
import argparse,copy,json,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from sttw_control.ditch_event_model import EventPlanner
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--event-model',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
torch.set_num_threads(2);torch.manual_seed(20261110);rng=np.random.default_rng(20261110);started=time.monotonic();planner=EventPlanner(a.event_model);base=Path(planner.meta['collection']);cfg=json.loads((base/'config.json').read_text());protocol=json.loads((base/'protocol.json').read_text());summary=json.loads((base/'evaluation/summary.json').read_text());d=np.load(base/'evaluation/trajectories.npz');roles=np.array([c['role'] for c in protocol['cases']]);success=np.array([c['success'] for c in summary['cases']],np.float32)
state=planner.encode(d['initial_root_xyzw'],d['initial_observation'],d['controller_history'][0],np.zeros(len(success)));params=np.asarray(planner.meta['candidate_parameters'],np.float32)[np.array(cfg['ground_handoff']['reference_route_ids'])]
x=np.concatenate((state,planner.plan_features(state,params),planner.map.crop(state[:,:13])),-1);train=roles=='TRAIN';mean=x[train].mean(0);std=np.maximum(x[train].std(0),.02);x=torch.from_numpy((x-mean)/std);y=torch.from_numpy(success);train_ids=np.flatnonzero(train);dev_ids=np.flatnonzero(roles=='DEV')
model=nn.Sequential(nn.Linear(x.shape[1],64),nn.SiLU(),nn.Linear(64,32),nn.SiLU(),nn.Linear(32,1));opt=torch.optim.Adam(model.parameters(),lr=.001);best=float('inf');log=[]
for update in range(600):
 ids=rng.choice(train_ids,256,replace=True);loss=nn.functional.binary_cross_entropy_with_logits(model(x[ids]).flatten(),y[ids]);opt.zero_grad();loss.backward();opt.step()
 if (update+1)%100==0:
  with torch.no_grad():dev_loss=float(nn.functional.binary_cross_entropy_with_logits(model(x[dev_ids]).flatten(),y[dev_ids]))
  log.append(dict(update=update+1,train_loss=float(loss),dev_loss=dev_loss))
  if dev_loss<best:best=dev_loss;selected=update+1;weights=copy.deepcopy(model.state_dict())
model.load_state_dict(weights)
with torch.no_grad():prob=model(x).sigmoid().flatten().numpy()
n=len(protocol['candidates']);labels=success.reshape(-1,n).astype(bool);choice=prob.reshape(-1,n).argmax(1);fixed=int(labels[roles[::n]=='TRAIN'].sum(0).argmax());result={}
for role in ('TRAIN','DEV','TEST'):
 r=np.flatnonzero(roles[::n]==role);pred=labels[r,choice[r]];b=labels[r,fixed];result[role]=dict(roots=len(r),selected=int(pred.sum()),fixed=int(b.sum()),gains=int((pred&~b).sum()),losses=int((~pred&b).sum()),brier=float(((prob[roles==role]-success[roles==role])**2).mean()),choices=choice[r].tolist())
torch.save(dict(state_dict=weights,mean=mean,std=std),a.output/'model.pt');(a.output/'results.json').write_text(json.dumps(dict(comparison=result,selected_update=selected,logs=log,elapsed_s=time.monotonic()-started,scope='Direct completion baseline; same TRAIN/DEV splits, fixed600updates/100interval; offline comparison, TEST previously inspected in event-model work and not a fresh confirmatory test'),indent=2));np.save(a.output/'probability.npy',prob);print(json.dumps(result,indent=2))
