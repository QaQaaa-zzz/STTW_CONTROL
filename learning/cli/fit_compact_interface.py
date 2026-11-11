"""Fit a compact action-injected residual interface on the existing frozen root split."""
import argparse,json,time,os
from pathlib import Path
import numpy as np
import torch
from torch import nn
from sttw_control.interface_world_model import ActionInjectedNetwork,compact_inputs,handoff_baseline
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--routes',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);start=time.monotonic();torch.set_num_threads(2)
contract=dict(change='TRAIN-only16D observed-history PCA;13root+16history+entrydistance+16terrain+time+8action=55inputs;action concatenated to final64D hidden;vx/time physical baseline residual39D;no autoregressive PCA prediction',budget='3members x250CPUepochs;0physics/PPO;stop within25min',split='Unchanged1379TRAIN/540DEV;valid-handoff regression463TRAIN/168DEV;historical240 never selected',limits='Reference speed/time baseline is an approximation;full-goal masks unchanged;not guaranteed cure or action diversity')
(a.output/'declaration.json').write_text(json.dumps(contract,indent=2));d=np.load(a.source/'dataset.npz');raw=d['x'];train=d['train'];dev=d['dev'];hit=d['hit'];jump=d['jump'];goal=d['goal'];target=d['target'];meta=json.loads((a.source/'model.json').read_text())
h=raw[train,13:445];hm=h.mean(0);hs=h.std(0).clip(.01);_,_,components=np.linalg.svd((h-hm)/hs,full_matrices=False)
transform=dict(history_mean=hm,history_std=hs,history_components=components[:16],jump_start_x_m=9.,roads=np.load(a.routes)['xy'],parameters=np.array(meta['candidate_parameters'],np.float32))
x=compact_inputs(raw,transform);baseline=handoff_baseline(raw,transform);target_residual=target-baseline
xm=x[train].mean(0);xs=x[train].std(0).clip(.01);ym=target_residual[train&hit].mean(0);ys=target_residual[train&hit].std(0).clip(.01)
xx=torch.tensor((x-xm)/xs);yy=torch.tensor((target_residual-ym)/ys);hh=torch.tensor(hit.astype(np.float32));gg=torch.tensor(goal);selection={};means=[];ph=[];pg=[]
def objective(out,mask):
 h=mask&jump;g=mask&~jump;v=mask&hit;lv=out[:,39:78].clamp(-4,3)
 return nn.functional.binary_cross_entropy_with_logits(out[h,78],hh[h])+nn.functional.binary_cross_entropy_with_logits(out[g,79],gg[g])+.5*((out[v,:39]-yy[v]).square()*(-lv[v]).exp()+lv[v]).mean()
for seed in [71101,71102,71103]:
 torch.manual_seed(seed);net=ActionInjectedNetwork(x.shape[1],80);opt=torch.optim.Adam(net.parameters(),lr=.001,weight_decay=.0001);best=float('inf');curve=[]
 for epoch in range(1,251):
  opt.zero_grad();loss=objective(net(xx),train);loss.backward();opt.step()
  with torch.no_grad():out=net(xx);score=float(objective(out,dev))
  curve.append([epoch,float(loss.detach()),score])
  if score<best:best=score;chosen=epoch;state={k:v.detach().clone() for k,v in net.state_dict().items()}
 net.load_state_dict(state)
 with torch.no_grad():out=net(xx);means.append(out[:,:39].numpy()*ys+ym+baseline);ph.append(torch.sigmoid(out[:,78]).numpy());pg.append(torch.sigmoid(out[:,79]).numpy())
 torch.save(dict(state_dict=state,input_mean=xm,input_std=xs,target_mean=ym,target_std=ys,compact_transform=transform,selected_epoch=chosen,seed=seed),a.output/f'upstream_{seed}.pt');(a.output/f'upstream_{seed}_curve.json').write_text(json.dumps(curve));selection[str(seed)]=dict(epoch=chosen,dev_objective=best)
for path in a.source.glob('direct_*.pt'):os.symlink(path.resolve(),a.output/path.name)
(a.output/'model.json').write_text(json.dumps(meta,indent=2));mean=np.mean(means,0);groups=[]
for role,mask in [('TRAIN',train),('DEV',dev)]:
 m=mask&hit;fields=[]
 for name,j in [('x',0),('y',1),('vx',7),('vy',8),('wx',10),('time',38)]:
  fields.append(dict(field=name,rmse=float(np.sqrt(((mean[m,j]-target[m,j])**2).mean())),pred_std=float(mean[m,j].std()),observed_std=float(target[m,j].std())))
 groups.append(dict(role=role,n=int(m.sum()),fields=fields))
np.savez_compressed(a.output/'predictions.npz',predicted_handoff_mean=mean,valid_handoff_probability=np.mean(ph,0),ground_probability=np.mean(pg,0))
(a.output/'results.json').write_text(json.dumps(dict(contract=contract,selection=selection,groups=groups,elapsed_s=time.monotonic()-start),indent=2));print(json.dumps(dict(selection=selection,groups=groups,elapsed_s=time.monotonic()-start),indent=2))
