"""Fit a small event predictor with whole-root TRAIN/DEV/TEST separation."""
import argparse,copy,json,time
from pathlib import Path
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from sttw_control.ditch_event_model import EventTransitionNetwork,KnownHeightMap,EventPlanner,route_parameters


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--collection',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--updates',type=int,default=600);p.add_argument('--conditional-event-targets',action='store_true');p.add_argument('--prefix-stride',type=int,default=0);p.add_argument('--skip-offline-selection',action='store_true');p.add_argument('--ground-height-hold',action='store_true')
 a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);torch.manual_seed(20261110);rng=np.random.default_rng(20261110);start=time.monotonic()
 cfg=json.loads((a.collection/'config.json').read_text());protocol=json.loads((a.collection/'protocol.json').read_text());summary=json.loads((a.collection/'evaluation/summary.json').read_text());data=dict(np.load(a.collection/'evaluation/trajectories.npz'));routes=np.load(cfg['ground_handoff']['reference_path'])['xy']
 roles=np.array([c['role'] for c in protocol['cases']]);terrain_file=a.collection/'evaluation/terrain/terrain.npz';terrain=KnownHeightMap(np.load(terrain_file)['vertices'],cfg['ditch_terrain'])
 params=np.stack([route_parameters(xy,cfg['ground_handoff']['approach']['speed_m_s'],cfg['ground_handoff']['approach']['alpha']) for xy in routes])
 ground_only=np.asarray(cfg['ground_handoff'].get('ground_only_lanes',[False]*len(roles)));route_ids=np.asarray(cfg['ground_handoff']['reference_route_ids'])
 if 'event_candidate_parameters' in cfg:params=np.asarray(cfg['event_candidate_parameters'],np.float32)
 elif ground_only.any():params=np.column_stack((params,[float(not ground_only[np.flatnonzero(route_ids==i)[0]]) for i in range(len(params))])).astype(np.float32)
 lane_params=params[route_ids]
 hidden=np.concatenate((data['observation'],data['controller_history']),-1)
 # Compress only TRAIN histories, sampled at 10Hz; all data roles stay grouped.
 pool=hidden[::5,roles=='TRAIN'].reshape(-1,hidden.shape[-1]);mask=data['valid'][::5,roles=='TRAIN'].ravel();pool=pool[mask]
 hm=pool.mean(0);hs=np.maximum(pool.std(0),.01);normalized=(pool-hm)/hs
 _,_,v=np.linalg.svd(normalized,full_matrices=False);components=v[:16];np.savez_compressed(a.output/'codec.npz',hidden_mean=hm,hidden_std=hs,components=components)
 def state(lane,t,root,road):
  ix=min(int(np.floor((t+1e-6)/.02)),len(hidden)-1)
  return np.r_[root,((hidden[ix,lane]-hm)/hs)@components.T,float(road)].astype(np.float32)
 records=[];initial=[]
 for lane,case in enumerate(summary['cases']):
  terminal_ix=np.flatnonzero(data['physics_valid'][:,lane])[-1];end_t=float(data['physics_time'][terminal_ix]);end_root=data['physics_root_xyzw'][terminal_ix,lane]
  terminal=state(lane,end_t,end_root,data['road_violation'][lane]);first=state(lane,0.,data['initial_root_xyzw'][lane],False);initial.append(first)
  if ground_only[lane]:
   hit=float(case['success']);t=float(data['event_time'][lane,4]) if hit else end_t
   target=state(lane,t,data['event_root_xyzw'][lane,4],False) if hit else terminal
   records.append((lane,4,first,target,t,hit))
   if a.prefix_stride:
    for ix in range(a.prefix_stride,case['steps'],a.prefix_stride):
     at=ix*.02
     if at<t:records.append((lane,4,state(lane,at,data['root_xyzw'][ix-1,lane],data['road_violation_history'][ix-1,lane]),target,t-at,hit))
   continue
  points=[(0.,first)]
  t=case['entry_s']
  if t is None:points.append(None)
  else:
   k=max(0,min(round(t/.02)-1,len(hidden)-1));points.append((t,state(lane,t,data['root_xyzw'][k,lane],data['road_violation_history'][k,lane])))
  if a.prefix_stride:
   nt,ns=points[1] if points[1] is not None else (end_t,terminal)
   for ix in range(a.prefix_stride,case['steps'],a.prefix_stride):
    at=ix*.02
    if at<nt:records.append((lane,0,state(lane,at,data['root_xyzw'][ix-1,lane],data['road_violation_history'][ix-1,lane]),ns,nt-at,float(points[1] is not None)))
  for event in (0,2,3,4):
   t=float(data['event_time'][lane,event]);points.append((t,state(lane,t,data['event_root_xyzw'][lane,event],data['event_road_violation'][lane,event])) if np.isfinite(t) else None)
  for mode in range(5):
   if points[mode] is None:break
   t,s=points[mode];nxt=points[mode+1];hit=float(nxt is not None)
   nt,ns=nxt if nxt is not None else (end_t,terminal)
   if nt+1e-4<t:raise ValueError('Event ordering violated')
   records.append((lane,mode,s,ns,max(0,nt-t),hit))
   if not hit:break
  if points[1] is not None:
   t,s=points[1];nt=float(data['event_time'][lane,1]);hit=float(np.isfinite(nt));ns=state(lane,nt,data['event_root_xyzw'][lane,1],data['event_road_violation'][lane,1]) if hit else terminal
   records.append((lane,5,s,ns,max(0,(nt if hit else end_t)-t),hit))
 lanes=np.array([r[0] for r in records]);modes=np.array([r[1] for r in records]);s=np.stack([r[2] for r in records]);ns=np.stack([r[3] for r in records]);dt=np.array([r[4] for r in records],np.float32);hit=np.array([r[5] for r in records],np.float32)
 def plan_features(states,parameters):
  spec=cfg['ditch_terrain'];x=states[:,0];return np.column_stack((parameters,spec['front_x_m']-x,spec['front_x_m']+spec['width_m']-x,cfg['ground_handoff']['goal']['x_min_m']-x)).astype(np.float32)
 plans=plan_features(s,lane_params[lanes]);crop=terrain.crop(s[:,:13]);train=roles[lanes]=='TRAIN';dev=roles[lanes]=='DEV'
 residual=ns-s;residual[:,:3]-=s[:,7:10]*dt[:,None]
 if a.ground_height_hold:
  ground=(modes==0)|(modes==4);residual[ground,2]=ns[ground,2]-s[ground,2]
 constructor=dict(ground_height_hold=a.ground_height_hold,plan_dimension=plans.shape[1],conditional_event_targets=a.conditional_event_targets,dimension=s.shape[1],state_mean=s[train].mean(0),state_std=np.maximum(s[train].std(0),.02),plan_mean=plans[train].mean(0),plan_std=np.maximum(plans[train].std(0),.02),residual_mean=residual[train].mean(0),residual_std=np.maximum(residual[train].std(0),.02))
 model=EventTransitionNetwork(**constructor);opt=torch.optim.Adam(model.parameters(),lr=.001);tensors=[torch.from_numpy(x) for x in (s,plans,crop,modes,ns,dt,hit)];train_ids=np.flatnonzero(train);dev_ids=np.flatnonzero(dev);best=float('inf');logs=[]
 writer=SummaryWriter(str(a.output/'tensorboard'))
 for update in range(a.updates):
  idx=rng.choice(train_ids,256,replace=True);loss=model.loss(*(x[idx] for x in tensors));opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step()
  if (update+1)%100==0 or update==a.updates-1:
   with torch.no_grad():validation=float(model.loss(*(x[dev_ids] for x in tensors)))
   writer.add_scalar('loss/train',float(loss),update+1);writer.add_scalar('loss/dev',validation,update+1);writer.flush()
   logs.append(dict(update=update+1,train_loss=float(loss),dev_loss=validation));print(logs[-1],flush=True)
   if validation<best:best=validation;selected=update+1;weights=copy.deepcopy(model.state_dict())
 writer.close()
 model.load_state_dict(weights);torch.save(dict(constructor=constructor,state_dict=weights),a.output/'model.pt')
 meta=dict(ground_height_hold=a.ground_height_hold,prefix_stride_policy_steps=a.prefix_stride,inference_common_random_numbers=True,plan_skill_feature='last candidate parameter:1jump,0ground' if ground_only.any() else None,terrain_file=str(terrain_file),terrain_spec=cfg['ditch_terrain'],goal=cfg['ground_handoff']['goal'],candidate_parameters=params.tolist(),candidate_names=protocol['candidates'],selected_update=selected,training_logs=logs,collection=str(a.collection),scope='Exploratory same-map event model; grouped starts only; no map generalization or planning benefit claim',conditional_event_targets=a.conditional_event_targets,state='root xyzw13 + TRAIN-only PCA16 of causal80+352 history + prior road violation; history compression is not a Markov guarantee')
 (a.output/'model.json').write_text(json.dumps(meta,indent=2));np.savez_compressed(a.output/'transitions.npz',lane=lanes,mode=modes,state=s,next_state=ns,time=dt,hit=hit,role=roles[lanes])
 metrics={}
 with torch.no_grad():
  mean,logvar,logtime,logit=model(*tensors[:4]);pred_t=torch.expm1(logtime.clamp(0,np.log(11.)));pred=model.physical_baseline(tensors[0],pred_t,tensors[3])+mean*model.residual_std+model.residual_mean;pred=pred.numpy();prob=torch.sigmoid(logit).numpy()
 for role in ('TRAIN','DEV','TEST'):
  metrics[role]={}
  for mode in range(6):
   keep=(roles[lanes]==role)&(modes==mode);positive=keep&(hit>0)
   reference=hit[train&(modes==mode)].mean() if (train&(modes==mode)).any() else None
   metrics[role][str(mode)]=dict(train_event_rate=None if reference is None else float(reference),constant_rate_brier=float(np.mean((reference-hit[keep])**2)) if reference is not None and keep.any() else None,n=int(keep.sum()),events=int(positive.sum()),event_brier=float(np.mean((prob[keep]-hit[keep])**2)) if keep.any() else None,time_mae_s=float(np.mean(abs(pred_t.numpy()[positive]-dt[positive]))) if positive.any() else None,xy_error_m=float(np.linalg.norm(pred[positive,:2]-ns[positive,:2],axis=1).mean()) if positive.any() else None)
 if a.skip_offline_selection:
  result=dict(metrics=metrics,selected_update=selected,elapsed_s=time.monotonic()-start,scope='Prediction on actual published-state segments; no complete counterfactual panel, so no offline route-selection score')
  (a.output/'results.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2));return
 initial=np.stack(initial);planner=EventPlanner(a.output);scores=planner.score(initial,lane_params,particles=32)
 success=np.array([c['success'] for c in summary['cases']]).reshape(-1,len(params));selection=np.array(scores['score']).reshape(-1,len(params)).argmax(1);fixed=int(success[roles[::len(params)]=='TRAIN'].sum(0).argmax());comparison={}
 for role in ('TRAIN','DEV','TEST'):
  roots=np.flatnonzero(roles[::len(params)]==role);selected_success=success[roots,selection[roots]];fixed_success=success[roots,fixed]
  comparison[role]=dict(roots=len(roots),event_selected=int(selected_success.sum()),fixed_train_best=int(fixed_success.sum()),oracle_coverage=int(success[roots].any(1).sum()),gains=int((selected_success&~fixed_success).sum()),losses=int((~selected_success&fixed_success).sum()),selection=selection[roots].tolist())
 result=dict(metrics=metrics,offline_selection=comparison,fixed_train_best=fixed,selected_update=selected,elapsed_s=time.monotonic()-start,scope='Offline selection from stored physics outcomes, not online execution. TEST roots never used in fitting/checkpoint selection; same map only.')
 (a.output/'results.json').write_text(json.dumps(result,indent=2));(a.output/'predictions.json').write_text(json.dumps(scores));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
