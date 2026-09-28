"""Persistent conservative accounting, committed BEFORE any physics work."""
import json
import os
import fcntl
from pathlib import Path

class BudgetExhausted(RuntimeError):pass

class Budget:
    def __init__(self,root,limit=80_000_000,substeps=25):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
        self.path=self.root/'budget.json';self.limit=min(int(limit),80_000_000);self.substeps=substeps
        if self.limit<0:raise ValueError('negative predictor budget')
        self.charge()
    def charge(self,predictor=0,plant=0,contracts=0,episodes=0,ablation=0,reason='initialize'):
        with (self.root/'.budget.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            d=json.loads(self.path.read_text()) if self.path.exists() else dict(
                predictor_ticks=0,plant_ticks=0,contract_rollouts=0,main_episodes=0,
                ablation_episodes=0,training_updates=0,training_transitions=0,events=[])
            limit=min(d.get('predictor_limit',self.limit),self.limit)
            if d['predictor_ticks']+predictor>limit or d['contract_rollouts']+contracts>128 or d['main_episodes']+episodes>42 or d['ablation_episodes']+ablation>4:
                raise BudgetExhausted('runtime_or_budget: no further work reserved')
            for k,v in [('predictor_ticks',predictor),('plant_ticks',plant),('contract_rollouts',contracts),('main_episodes',episodes),('ablation_episodes',ablation)]:
                if v<0:raise ValueError('negative budget charge')
                d[k]+=int(v)
            d['predictor_limit']=limit;d['predictor_physics_steps']=d['predictor_ticks']*self.substeps
            d['plant_physics_steps']=d['plant_ticks']*self.substeps
            d['accounting']='reserved before dispatch; includes padding; interrupted reservations are not refunded'
            if predictor or plant or contracts or episodes or ablation:
                d['events'].append(dict(reason=reason,predictor=predictor,plant=plant,contracts=contracts,episodes=episodes,ablation=ablation))
            tmp=self.path.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2)+'\n');os.replace(tmp,self.path)
            return d
