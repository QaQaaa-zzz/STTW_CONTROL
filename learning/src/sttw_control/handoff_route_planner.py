"""One measured-state downstream route query with physically propagated delay."""
import json
import math
import time
from pathlib import Path
import numpy as np
from .handoff_model import HandoffContinuationModel
from .handoff_task_context import route_context


class HandoffRoutePlanner:
    def __init__(self, task, config):
        self.task=task
        self.config=config
        self.route_ids=config['candidate_route_ids']
        self.methods=config['methods']
        if len(self.methods)!=task.env.num_envs or any(m not in ('task_context','state_only','fixed_straight','planned') for m in self.methods):
            raise ValueError('One declared downstream method is required per environment')
        route=task.ground.route
        if any(i<0 or i>=len(route.xy) for i in self.route_ids):raise ValueError('Invalid downstream candidate route')
        self.routes=[dict(xy=route.xy[i].cpu().numpy(),heading=route.heading[i].cpu().numpy(),curvature=route.curvature[i].cpu().numpy()) for i in self.route_ids]
        self.models={name:HandoffContinuationModel(config['model'],feature_set=name) for name in ('task_context','state_only')}
        self.mount=json.loads((Path(config['model'])/'declaration.json').read_text())['mount_wxyz']
        self.pending={}
        self.records=[]

    def query(self, ids, root, observation):
        e=self.task.env
        for lane in ids.cpu().tolist():
            begin=time.perf_counter();method=self.methods[lane]
            physical=root[lane:lane+1].detach().cpu().numpy()
            obs=observation[lane:lane+1].detach().cpu().numpy()
            now=float(e.episode_length_buf[lane])*e.dt
            context=np.concatenate([route_context(physical,route,self.task.road_half_width,self.mount) for route in self.routes])
            if method in ('fixed_straight','planned'):
                desired=self.route_ids[0] if method=='fixed_straight' else int(self.task.handoff_route_ids[lane])
                scores=np.asarray([float(r==desired) for r in self.route_ids])
            else:
                n=len(self.routes)
                scores=self.models[method].predict(np.repeat(physical,n,0),np.repeat(obs,n,0),np.full(n,now),context)['conditional_goal_probability']
            chosen=int(np.argmax(scores));elapsed=time.perf_counter()-begin
            delay=max(1,math.ceil(elapsed/e.dt))
            record=dict(lane=lane,method=method,query_s=now,query_root=physical[0].tolist(),query_observation=obs[0].tolist(),task_context=context.tolist(),scores=scores.tolist(),selected_route=self.route_ids[chosen],query_seconds=elapsed,simulated_delay_s=delay*e.dt,status='pending')
            self.pending[lane]=(int(e.episode_length_buf[lane])+delay,record)
            self.records.append(record)

    def publish_due(self, root):
        task=self.task;e=task.env
        for lane,(due,record) in list(self.pending.items()):
            if bool(task.finished[lane]):
                record['status']='terminated_before_publication';del self.pending[lane]
            elif int(e.episode_length_buf[lane])>=due:
                task.ground.route.route_ids[lane]=record['selected_route']
                record.update(status='published',publication_s=float(e.episode_length_buf[lane])*e.dt,publication_root=root[lane].detach().cpu().tolist())
                del self.pending[lane]

    def save(self, output):
        Path(output,'handoff_planning.json').write_text(json.dumps(dict(records=self.records,scope='One measured-state handoff query; default straight H75 runs during rounded-up query delay; publication changes route only,not physical/controller state'),indent=2))
