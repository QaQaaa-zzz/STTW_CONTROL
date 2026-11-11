"""Train upstream handoff distribution and compose with a frozen downstream goal head."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from sttw_control.interface_world_model import network,inputs,InterfaceWorldModel
from sttw_control.ditch_event_model import KnownHeightMap
p=argparse.ArgumentParser();p.add_argument('--runs',type=Path,required=True);p.add_argument('--downstream',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--include-distance-training',action='store_true');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);start=time.monotonic()
sources={'source720':'chapter3_offroad_event_training/collection','source1440':'chapter3_offroad_distance_training/collection','source240':'chapter3_offroad_new_roots/source'}
contract=dict(scope='Fixed known map,original jump,cold H751.5m/s alpha0.5; all four candidate routes; no RL or new physics',inputs='query root13,observation80,recorded ground history352,global time1,candidate8,known terrain crop128',outputs='valid handoff probability; conditional39D handoff Gaussian; separate direct-ground goal head; frozen downstream39D goal ensemble',labels='Invalid/missing handoffs included in Bernoulli training; state NLL only valid handoffs; full-goal labels unchanged',include_distance_training=a.include_distance_training,split='Original grouped TRAIN/DEV roles;source1440 optionally included for query-distance coverage;source240 historical transfer never selected',budget='CPU3seeds x250epochs for upstream and direct classifier;25min wall cap;0physics/0PPO',loss='BCE valid-handoff on jump lanes+BCE ground-goal on ground lanes+conditional diagonal Gaussian NLL; DEV same loss selects epoch. Direct goal BCE baseline same architecture/budget.',latency='Training forced candidates have measured query-to-publication latency recorded; online latency must be propagated and rechecked.',uncertainty='Diagonal Gaussian is an approximation; state/event errors and composed calibration reported, no safety claim')
(a.output/'declaration.json').write_text(json.dumps(contract,indent=2))
features=[];targets=[];hits=[];goals=[];jumps=[];rows=[];roots=[];observations=[];histories=[];times=[];params=[]
for name,rel in sources.items():
 run=a.runs/rel;cfg=json.loads((run/'config.json').read_text());pr=json.loads((run/'protocol.json').read_text())['cases'];cases=json.loads((run/'evaluation/summary.json').read_text())['cases'];records=json.loads((run/'evaluation/moving_planning.json').read_text())['records'];z=np.load(run/'evaluation/trajectories.npz')
 terrain=run/'evaluation/terrain/terrain.npz';mapping=KnownHeightMap(np.load(terrain)['vertices'],cfg['ditch_terrain'])
 physical_roots=z['root_xyzw'];obs=z['observation'];hist=z['controller_history'];event=z['event_root_xyzw'];et=z['event_time'][:,3];road=z['event_road_violation'];parameter=np.array(cfg['moving_event_planner']['candidate_parameters'],np.float32)
 if name=='source720':
  meta=dict(terrain_npz=str(terrain),terrain_spec=cfg['ditch_terrain'],downstream_directory=str(a.downstream),candidate_parameters=parameter.tolist(),scope=contract);(a.output/'model.json').write_text(json.dumps(meta,indent=2))
 for rec in records:
  if rec['status']!='published':continue
  i=rec['lane'];step=rec['query_step'];root=np.array(rec['query_root'],np.float32);par=parameter[rec['selected']];jump=par[-1]>.5;hit=bool(jump and np.isfinite(et[i]) and not road[i,3]);target=np.zeros(39,np.float32)
  if hit:
   k=round(float(et[i])/.02);assert np.array_equal(physical_roots[k-1,i],event[i,3])
   target=np.r_[event[i,3],obs[k,i,50:75],et[i]]
  r=root[None];ob=obs[step,i:i+1];hi=hist[step,i:i+1];tm=step*.02
  features.append(inputs(r,ob,hi,[tm],par[None],mapping.crop(r))[0]);targets.append(target);hits.append(hit);goals.append(cases[i]['success']);jumps.append(jump)
  roots.append(root);observations.append(ob[0]);histories.append(hi[0]);times.append(tm);params.append(par)
  rows.append(dict(dataset=name,lane=i,root=pr[i]['root'],role=pr[i]['role'] if name=='source720' or (name=='source1440' and a.include_distance_training) else 'HISTORICAL_TRANSFER',candidate=pr[i]['candidate'],query_x=pr[i]['x'],latency_s=rec['simulated_delay_s'],valid_handoff=hit,success=cases[i]['success']))
x=np.array(features,np.float32);target=np.array(targets,np.float32);hit=np.array(hits);goal=np.array(goals,np.float32);jump=np.array(jumps);train=np.array([r['role']=='TRAIN' for r in rows]);dev=np.array([r['role']=='DEV' for r in rows])
xm=x[train].mean(0);xs=x[train].std(0).clip(.01);ym=target[train&hit].mean(0);ys=target[train&hit].std(0).clip(.01)
xx=torch.tensor((x-xm)/xs);yy=torch.tensor((target-ym)/ys);hh=torch.tensor(hit.astype(np.float32));gg=torch.tensor(goal);pred={};selection={}
np.savez_compressed(a.output/'dataset.npz',x=x,target=target,hit=hit,goal=goal,jump=jump,train=train,dev=dev)
def objective(out,mask):
 h=mask&jump;g=mask&~jump;v=mask&hit
 mean=out[:,:39];lv=out[:,39:78].clamp(-4,3)
 return nn.functional.binary_cross_entropy_with_logits(out[h,78],hh[h])+nn.functional.binary_cross_entropy_with_logits(out[g,79],gg[g])+.5*((mean[v]-yy[v]).square()*(-lv[v]).exp()+lv[v]).mean()
for kind in ['upstream','direct']:
 outputs=[]
 for seed in [71101,71102,71103]:
  torch.manual_seed(seed);net=network(x.shape[1],80 if kind=='upstream' else 1);opt=torch.optim.Adam(net.parameters(),lr=.001,weight_decay=.0001);best=float('inf');curve=[]
  for epoch in range(1,251):
   opt.zero_grad();out=net(xx);loss=objective(out,train) if kind=='upstream' else nn.functional.binary_cross_entropy_with_logits(out[train,0],gg[train]);loss.backward();opt.step()
   with torch.no_grad():
    val=net(xx);score=float(objective(val,dev)) if kind=='upstream' else float(((torch.sigmoid(val[dev,0])-gg[dev])**2).mean())
   curve.append([epoch,float(loss.detach()),score])
   if score<best:best=score;state={k:v.clone().detach() for k,v in net.state_dict().items()};selected=epoch
  net.load_state_dict(state)
  torch.save(dict(state_dict=state,input_mean=xm,input_std=xs,target_mean=ym,target_std=ys,selected_epoch=selected,seed=seed),a.output/f'{kind}_{seed}.pt')
  selection[f'{kind}_{seed}']=dict(epoch=selected,dev_score=best)
  (a.output/f'{kind}_{seed}_curve.json').write_text(json.dumps(curve))
  if kind=='direct':
   with torch.no_grad():outputs.append(torch.sigmoid(net(xx)[:,0]).numpy())
 if outputs:pred['direct']=np.mean(outputs,axis=0)
model=InterfaceWorldModel(a.output);forecast=model.score(np.array(roots),np.array(observations),np.array(histories),np.array(times),np.array(params),particles=32)
pred['interface']=np.array(forecast['predicted_completion']);pred['constant']=np.full(len(goal),goal[train].mean());route=np.array([r['candidate'].split('_v')[0] for r in rows]);pred['route_prior']=np.array([goal[train&(route==r)].mean() for r in route])
groups=[]
for name,mask in [('pooled_TRAIN',train),('pooled_DEV',dev)]+[(n,np.array([r['dataset']==n for r in rows])) for n in sources if n!='source720']:
 valid=mask&hit;means=np.array(forecast['predicted_handoff_mean']);ph=np.array(forecast['valid_handoff_probability'])
 groups.append(dict(dataset=name,n=int(mask.sum()),goal=int(goal[mask].sum()),valid_handoffs=int(valid.sum()),brier={k:float(((v[mask]-goal[mask])**2).mean()) for k,v in pred.items()},valid_handoff_brier=float(((ph[mask&jump]-hit[mask&jump])**2).mean()),handoff_position_rmse_m=float(np.sqrt(((means[valid,:3]-target[valid,:3])**2).mean())),handoff_time_mae_s=float(abs(means[valid,38]-target[valid,38]).mean())))
np.savez_compressed(a.output/'predictions.npz',**pred,goal=goal,valid_handoff_probability=forecast['valid_handoff_probability'],predicted_handoff_mean=forecast['predicted_handoff_mean'])
result=dict(groups=groups,selection=selection,cases=rows,elapsed_s=time.monotonic()-start,scope=contract)
(a.output/'results.json').write_text(json.dumps(result,indent=2));print(json.dumps(dict(groups=groups,elapsed_s=result['elapsed_s']),indent=2))
