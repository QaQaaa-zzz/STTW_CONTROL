"""Persistent experiment compute ledger: stages and reservations never reset on restart."""
from contextlib import contextmanager
from pathlib import Path
import fcntl
import json
import os
import signal
import time
import math

class BudgetStop(RuntimeError):
    pass

class ComputeBudget:
    def __init__(self, root, spec):
        self.root=Path(root); self.root.mkdir(parents=True,exist_ok=True)
        self.path=self.root/'budget.json'; self.spec=spec
        e=spec['budget']
        self.limits={'compile':e['compile_wall_seconds'],'smoke':e['engineering_wall_seconds'],
                     'pilot':e['training_wall_seconds'],'review':e['evaluation_wall_seconds'],'checks':e['additional_compute_wall_seconds']}
        self.total_limit=e['additional_compute_wall_seconds']
        # V5.1's declared transition ceiling describes PPO sampling only.
        # Evaluation is a separate evidence pass and must never terminate after
        # a fully completed learner because training consumed its exact budget.
        self.tick_limit=(e['default_control_transitions_upper'] if spec.get('priority_recovery_v51')
                         else e['default_control_transitions_upper']+5600+1024+1600+19200)
        self._update(lambda d: d.update(wall_limits_enabled=spec['budget'].get('wall_limits_enabled',True)))
    def _update(self, fn):
        with (self.root/'.compute.lock').open('a') as f:
            fcntl.flock(f,fcntl.LOCK_EX)
            d=json.loads(self.path.read_text()) if self.path.exists() else dict(
                schema='direct_command_v3_budget',seconds={},control_ticks_reserved=0,events=[],active=None)
            fn(d)
            tmp=self.path.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2)+'\n');os.replace(tmp,self.path)
            return d
    def remaining(self, stage):
        if not self.spec['budget'].get('wall_limits_enabled',True):return float('inf')
        d=json.loads(self.path.read_text())
        return max(0.,min(self.total_limit-sum(d['seconds'].values()),self.limits[stage]-d['seconds'].get(stage,0.)))
    def reserve(self, ticks, reason):
        def change(d):
            if d['control_ticks_reserved']+ticks>self.tick_limit:raise BudgetStop('control tick limit')
            d['control_ticks_reserved']+=int(ticks)
            d['physics_steps_reserved']=d['control_ticks_reserved']*25
            d['events'].append(dict(kind='tick_reservation',ticks=int(ticks),reason=reason))
        self._update(change)
    @contextmanager
    def measure(self,stage,reason,estimate=0.,cap=None):
        left=self.remaining(stage)
        if cap is not None and self.spec['budget'].get('wall_limits_enabled',True):left=min(left,cap)
        if left<=max(0.01,estimate):raise BudgetStop(f'{stage}: insufficient wall budget ({left:.3f}s)')
        start=time.monotonic(); old=signal.getsignal(signal.SIGALRM)
        def alarm(*_):raise BudgetStop(f'{stage}: wall budget reached during {reason}')
        self._update(lambda d:d.update(active=dict(stage=stage,reason=reason,started_epoch=time.time(),reserved_s=left if math.isfinite(left) else None)))
        signal.signal(signal.SIGALRM,alarm)
        if math.isfinite(left):signal.setitimer(signal.ITIMER_REAL,left)
        outcome='complete'
        try:yield
        except BaseException:
            outcome='interrupted';raise
        finally:
            signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old)
            elapsed=time.monotonic()-start
            def change(d):
                d['seconds'][stage]=d['seconds'].get(stage,0.)+elapsed
                d['total_compute_seconds']=sum(d['seconds'].values())
                d['events'].append(dict(kind='compute',stage=stage,reason=reason,seconds=elapsed,outcome=outcome))
                d['active']=None
            self._update(change)
    def recover_interrupted(self):
        def change(d):
            if d['active']:
                a=d['active']; elapsed=max(0.,time.time()-a['started_epoch'])
                d['seconds'][a['stage']]=d['seconds'].get(a['stage'],0.)+elapsed
                d['events'].append(dict(kind='unclean_exit_charge',**a,seconds=elapsed))
                d['active']=None
        self._update(change)
