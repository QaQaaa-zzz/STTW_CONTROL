"""Batched fixed-path DEV, preserving paths and every active lower-step log.

Cases with different path array shapes compile separately rather than padding or
modifying geometry. One host synchronization per 50-policy-step chunk replaces
per-policy-step synchronization. This module never launches training by itself.
"""
from pathlib import Path
import time
import numpy as np
import jax
import jax.numpy as j
from .preference_best import atomic_json


def _signature(tree):
    return tuple((tuple(x.shape),str(x.dtype)) for x in jax.tree.leaves(tree))


class BatchedPanelEvaluator:
    def __init__(self,env,snapshot,config,check_budget=lambda:None,
                 status=lambda **kw:None,account=lambda ticks:None):
        self.env=env;self.snapshot=snapshot;self.config=config
        self.check_budget=check_budget;self.status=status;self.account=account
        self.compiled={}

    def _compiled_group(self,paths,speeds,alpha,params,horizon):
        from .path_command_policy import PathCommandActor
        env=self.env
        key=(_signature(paths),_signature(speeds),_signature(params),float(horizon))
        if key not in self.compiled:
            def reset(paths,speeds,alpha):
                return jax.vmap(lambda p,v:env.reset(self.snapshot,p,v,alpha))(paths,speeds)
            reset_fn=jax.jit(reset).lower(paths,speeds,alpha).compile()
            states=reset_fn(paths,speeds,alpha);jax.block_until_ready(states)
            example=jax.tree.map(lambda x:x[0],states)
            path=jax.tree.map(lambda x:x[0],paths)
            env.set_log_template(example,path,speeds[0])
            template=jax.eval_shape(lambda state:env.policy_step(state,j.zeros(2),path,speeds[0],horizon),example)[1]
            zero_logs=jax.tree.map(lambda x:j.zeros(x.shape,x.dtype),template)
            horizon_ticks=round(horizon/self.config['timing']['lower_dt_s'])
            actor=PathCommandActor()
            def chunk(states,paths,speeds,params):
                def one(state,path,v):
                    active=(state.tick<horizon_ticks)&~state.physical.failed&~state.fault&~state.domain_exit
                    def advance(state):
                        obs,_,fault=env.observation(state,path,v,horizon)
                        state=state.replace(fault=state.fault|fault)
                        z=actor.apply(params,obs)
                        return env.policy_step(state,z,path,v,horizon)
                    return jax.lax.cond(active,advance,lambda state:(state,zero_logs),state)
                def step(states,_):return jax.vmap(one)(states,paths,speeds)
                return jax.lax.scan(step,states,None,length=50)
            chunk_fn=jax.jit(chunk).lower(states,paths,speeds,params).compile()
            self.compiled[key]=(reset_fn,chunk_fn)
        return self.compiled[key]

    def evaluate(self,params,alpha,output,cached_names=None):
        from .path_command_selection import cases
        from .geometric_path import build_path
        from .direct_command_training import flatten_logs
        started=time.perf_counter();out=Path(output)
        if out.exists() and cached_names is None:raise FileExistsError('validation output immutable')
        out.mkdir(parents=True,exist_ok=cached_names is not None)
        declared=cases(self.config);names={name for name,_,_ in declared}
        cached=set(cached_names or ())
        if not cached<=names:raise ValueError('unknown cached validation case')
        horizon=self.config['training_future_phase_C']['episode_s']
        upper_dt=self.config['timing']['upper_dt_s'];intervals=round(horizon/upper_dt)
        if abs(upper_dt-.02)>1e-12 or intervals%50:
            raise ValueError('batched evaluator requires existing 50Hz whole-second horizon')
        traces={};groups={};pose=np.asarray(self.env.physics.helpers.pose(self.snapshot.data))
        timing=dict(schema='path_batched_evaluation_v1',compile_seconds=0.,physics_seconds=0.,
            transfer_seconds=0.,file_seconds=0.,reset_seconds=0.,cached_cases=sorted(cached),
            chunk_policy_steps=50,lower_dt_s=self.config['timing']['lower_dt_s'],groups=[],completed=False)
        def write_timing():
            timing['total_seconds']=time.perf_counter()-started
            atomic_json(out/'evaluation_timings.json',timing)
        def save_group(rows,paths,chunks):
            before=time.perf_counter()
            if not chunks:return
            flat=flatten_logs(jax.tree.map(lambda *xs:np.concatenate(xs,axis=0),*chunks))
            for i,(name,_,_) in enumerate(rows):
                # scan-time, case, lower-tick, remaining value axes.
                data={k:v[:,i].reshape((-1,)+v.shape[3:]) for k,v in flat.items()}
                active=np.asarray(data['active_tick'],bool);data={k:v[active] for k,v in data.items()}
                path=jax.tree.map(lambda x:np.asarray(x[i]),paths)
                data['goal_progress']=np.full(len(data['time']),float(path.goal))
                np.savez_compressed(out/f'{name}.npz',**data)
                np.savez_compressed(out/f'{name}_path.npz',s=path.s,xy=path.xy,
                    heading=path.heading,curvature=path.curvature,goal=path.goal,turn_end=path.turn_end)
                traces[name]=data
            timing['file_seconds']+=time.perf_counter()-before
        try:
            for name,route,speed in declared:
                if name in cached:
                    before=time.perf_counter()
                    with np.load(out/f'{name}.npz') as archive:traces[name]={k:archive[k] for k in archive.files}
                    timing['file_seconds']+=time.perf_counter()-before
                    continue
                path=build_path(route,pose,self.config)
                groups.setdefault(_signature(path),[]).append(((name,route,speed),path))
            for entries in groups.values():
                self.check_budget();rows=[row for row,_ in entries]
                paths=jax.tree.map(lambda *xs:j.stack(xs),*[path for _,path in entries])
                speeds=j.asarray([row[2] for row in rows],j.float32);a=j.asarray(alpha,j.float32)
                before=time.perf_counter()
                reset,chunk=self._compiled_group(paths,speeds,a,params,horizon)
                timing['compile_seconds']+=time.perf_counter()-before
                before=time.perf_counter();states=reset(paths,speeds,a);jax.block_until_ready(states)
                timing['reset_seconds']+=time.perf_counter()-before
                names_group=[row[0] for row in rows];chunks=[]
                group=dict(cases=names_group,path_points=int(paths.s.shape[-1]),active_lower_ticks=0,completed_intervals=0)
                timing['groups'].append(group)
                if bool(j.any(states.fault|states.domain_exit)):
                    # Reset-domain invalidity is engineering fault, not a policy sample.
                    raise RuntimeError('fixed validation reset invalid; no best selection')
                for offset in range(0,intervals,50):
                    self.check_budget()
                    self.status(state='running',stage='validation',case=','.join(names_group),completed_intervals=offset)
                    before=time.perf_counter();states,logs=chunk(states,paths,speeds,params);jax.block_until_ready(states)
                    timing['physics_seconds']+=time.perf_counter()-before
                    before=time.perf_counter();host_logs,failed,domain,fault,ticks=jax.device_get((logs,states.physical.failed,states.domain_exit,states.fault,states.tick))
                    timing['transfer_seconds']+=time.perf_counter()-before
                    chunks.append(host_logs);active_ticks=int(np.sum(host_logs['active_tick']))
                    self.account(active_ticks);group['active_lower_ticks']+=active_ticks;group['completed_intervals']=offset+50
                    if np.any(fault):
                        save_group(rows,paths,chunks)
                        raise RuntimeError('fixed validation engineering fault; no best selection')
                    if np.all(failed|domain|(ticks>=round(horizon/self.config['timing']['lower_dt_s']))):break
                save_group(rows,paths,chunks)
            timing['completed']=True
            return traces
        finally:write_timing()
