"""One moving-state route decision with measured, physically propagated latency.

Road admission is a geometric hard check. Event scores are learned estimates,
not feasibility certificates. Controllers and physical state are never reset.
"""
import json,time
from pathlib import Path
import numpy as np
import torch
from .ditch_event_model import EventPlanner


def route_distances(position,roads):
    delta=position[:,None,None,:2]-roads[None,:,:-1,:]
    segment=roads[:,1:]-roads[:,:-1]
    fraction=((delta*segment[None]).sum(-1)/(segment*segment).sum(-1)[None]).clip(0,1)
    return np.linalg.norm(delta-fraction[:,:,:,None]*segment[None],axis=-1).min(-1)


class MovingDitchPlanner:
    def __init__(self,task,config,output):
        self.task=task;self.config=config;self.output=Path(output);self.planner=EventPlanner(config['model'])
        self.parameters=np.asarray(config.get('candidate_parameters',self.planner.meta['candidate_parameters']),np.float32)
        self.roads=np.load(task.env.cfg['task_reference_path'])['xy'][:len(self.parameters)]
        self.threshold=np.broadcast_to(np.asarray(config['decision_x_m']),task.env.num_envs)
        self.predictors={'event_model':self.planner}
        self.predictors.update({name:EventPlanner(path) for name,path in config.get('event_models',{}).items()})
        self.completion=None
        if 'direct_model' in config:
            from .ditch_completion_model import CompletionPlanner
            self.completion=CompletionPlanner(config['direct_model']);self.predictors['direct_success']=self.completion
        self.methods=config['methods'];self.queried=np.zeros(task.env.num_envs,bool);self.pending={};self.records=[]
    def admission(self,roots):
        distance=route_distances(roots,self.roads);allowed=distance<=self.task.road_half_width
        allowed[:,self.parameters[:,-1]>.5]&=roots[:,0,None]<self.task.approach['jump_start_x_m']
        return allowed,distance
    def publish(self,ids,selected):
        task=self.task;g=task.ground;device=task.env.device;ids=torch.as_tensor(ids,device=device)
        selected=torch.as_tensor(selected,device=device);changed=g.route.route_ids[ids]!=selected;g.route.route_ids[ids]=selected
        parameters=torch.as_tensor(self.parameters,device=device,dtype=g.speed.dtype)
        g.speed[ids]=parameters[selected,5];g.alpha[ids]=parameters[selected,6]
        ground=parameters[selected,-1]<.5
        task.ground_only[ids]=ground;task.phase[ids]=torch.where(ground,3,0)
        ids=ids[changed]
        if not len(ids):return
        # Reproject the chosen lane only; retain ESO, action filters and history.
        _,pose,velocity=g.plant.observe();points=g.route.xy[g.route.route_ids]
        nearest=(points-pose[:,None,:2]).square().sum(-1).argmin(-1)
        seed=g.route.s[nearest];projection=g.route.project(pose[:,:2],seed,torch.zeros_like(seed))
        follower=g.route.pursuit(projection['progress'],pose,velocity,g.speed)
        for key,value in projection.items():g.middle.projection[key][ids]=value[ids]
        for key,value in follower.items():g.middle.follower[key][ids]=value[ids]
        g.preference_progress[ids]=projection['progress'][ids]
    @torch.no_grad()
    def tick(self,step,observation):
        task=self.task;env=task.env;g=task.ground;roots=env.physical()[0].cpu().numpy()
        phase=task.phase.cpu().numpy();finished=task.finished.cpu().numpy()
        for lane in list(self.pending):
            record=self.pending[lane]
            if step<record['apply_at_step']:continue
            allowed,distance=self.admission(roots[lane:lane+1]);allowed&=np.asarray(record['eligible_candidates'])[None];scores=np.asarray(record['scores']);scores=np.where(allowed[0],scores,-np.inf)
            if finished[lane] or phase[lane]!=0:status='decision_window_closed';selected=None
            elif not allowed.any():status='no_admissible_connection_keep_current';selected=None
            else:
                selected=int(scores.argmax());self.publish([lane],[selected]);status='published'
            record.update(status=status,selected=selected,published_step=step,published_root=roots[lane].tolist(),admissible_at_publish=allowed[0].tolist(),distance_at_publish_m=distance[0].tolist(),feedback_changed_choice=selected is not None and selected!=record['proposed'],simulated_delay_s=(step-record['query_step'])*env.dt,motion_during_delay_m=float(np.linalg.norm(roots[lane,:2]-np.asarray(record['query_root'])[:2])))
            del self.pending[lane]
        ids=np.flatnonzero(~self.queried&~finished&(phase==0)&(roots[:,0]>=self.threshold))
        for lane in ids:
            started=time.monotonic();allowed,distance=self.admission(roots[lane:lane+1])
            eligible=np.ones(len(self.parameters),bool)
            if self.methods[lane]=='forced':
                eligible[:]=False;eligible[self.config['forced_candidates'][lane]]=True;allowed&=eligible[None]
                prediction=None;scores=eligible.astype(float)
            elif self.methods[lane] in self.predictors:
                history=torch.cat([v.flatten(1) for v in (g.controller.eso,g.controller.disturbance[:,None],g.controller.gains,g.previous_reference,g.previous_command,g.policy.filtered,g.policy.offsets,g.middle.frames,g.middle.mask)],-1)[lane:lane+1].cpu().numpy()
                predictor=self.predictors[self.methods[lane]]
                state=predictor.encode(roots[lane:lane+1],observation[lane:lane+1].cpu().numpy(),history,task.road_violation[lane:lane+1].cpu().numpy())
                prediction=predictor.score(np.repeat(state,len(self.parameters),0),self.parameters,particles=32)
                scores=np.asarray(prediction['score'])
            else:
                prediction=None;scores=np.zeros(len(self.parameters));scores[self.config['fixed_jump_candidate']]=.5;scores[self.parameters[:,-1]<.5]=1.
                if 'fixed_ground_candidate' in self.config:scores[self.config['fixed_ground_candidate']]=1.1
            if self.methods[lane]=='state_rule' and roots[lane,0]>=self.config['detour_commit_x_m']:
                scores[self.parameters[:,-1]<.5]=-1.
            masked=np.where(allowed[0],scores,-np.inf);proposed=int(masked.argmax()) if allowed.any() else None
            elapsed=time.monotonic()-started;delay=max(1,int(np.ceil(elapsed/env.dt)))
            record=dict(lane=int(lane),method=self.methods[lane],query_step=step,query_root=roots[lane].tolist(),query_seconds=elapsed,apply_at_step=step+delay,proposed=proposed,eligible_candidates=eligible.tolist(),scores=scores.tolist(),admissible_at_query=allowed[0].tolist(),distance_at_query_m=distance[0].tolist(),prediction=prediction,status='pending',scope='Inference wall latency is rounded up to policy steps; old control executes these physical steps before publication; corridor admission is geometric, outcome estimates are not guarantees')
            self.queried[lane]=True;self.pending[int(lane)]=record;self.records.append(record)
        if ids.size or step%50==0:self.save()
    def save(self):
        (self.output/'moving_planning.json').write_text(json.dumps(dict(records=self.records,unqueried=np.flatnonzero(~self.queried).tolist(),pending=list(self.pending)),indent=2))
