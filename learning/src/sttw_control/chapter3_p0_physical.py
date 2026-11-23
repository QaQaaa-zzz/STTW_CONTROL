"""Matched physical branches, handback evidence, and delayed rolling-route publication."""
import json
import pickle
import time

import jax
import numpy as np
import torch
from mujoco.mjx.warp.types import DATA_NON_VMAP

from .chapter3_continuous_exit import ContinuousExitPlanner, committed_progress
from .platform_online import velocity_state
from .playground_snapshot import capture


def lane_tensors(target, prefix, count):
    fields = {}
    for k, x in target.items():
        if k in ('weights', 'policy', 'route', 'config', 'controller'):
            continue
        name = f'{prefix}_{k}'
        if isinstance(x, dict):
            fields.update(lane_tensors(x, name, count))
        elif torch.is_tensor(x) and x.ndim and len(x)==count:
            fields[name] = x
    return fields


def rejoin_default(route, rows, default, committed):
    """After the immutable prefix, join the original straight-exit curve over 1.6 m."""
    s = route.s
    xy = route.xy[rows]
    initial = route.xy[default]
    x, y = xy.unbind(-1)
    gx, gy = initial.unbind(-1)
    i = torch.searchsorted(gx.contiguous(), x.contiguous(), right=True).sub(1).clamp(0,len(s)-2)
    weight = ((x-gx.gather(1,i))/(gx.gather(1,i+1)-gx.gather(1,i)).clamp_min(1e-8)).clamp(0.,1.)
    target = torch.lerp(gy.gather(1,i),gy.gather(1,i+1),weight)
    end = torch.searchsorted(s,committed.contiguous()).clamp_max(len(s)-1)
    u = ((x-x.gather(1,end[:,None]))/1.6).clamp(0.,1.)
    yy = torch.lerp(y,target,u**3*(10.-15.*u+6.*u**2))
    # Use the published arc grid again. Anchor the new arc at the last immutable point.
    arc = torch.cat((torch.zeros_like(x[:,:1]),torch.hypot(x.diff(dim=1),yy.diff(dim=1)).cumsum(1)),1)
    arc = arc-arc.gather(1,end[:,None])+s[end,None]
    at = s.expand(len(rows),-1).contiguous()
    i = torch.searchsorted(arc.contiguous(),at,right=True).sub(1).clamp(0,len(s)-2)
    w = ((at-arc.gather(1,i))/(arc.gather(1,i+1)-arc.gather(1,i)).clamp_min(1e-8)).clamp(0,1)
    new_x = torch.lerp(x.gather(1,i),x.gather(1,i+1),w)+(at-arc[:,-1:]).clamp_min(0.)
    new_y = torch.lerp(yy.gather(1,i),yy.gather(1,i+1),w)
    # Arc grid is uniform. atan2(dy, dx) is independent of the sample spacing.
    h = torch.atan2(torch.gradient(new_y,dim=(1,))[0],torch.gradient(new_x,dim=(1,))[0])
    k = torch.gradient(h,dim=(1,))[0]/(s[1]-s[0])
    prefix = torch.arange(len(s),device=s.device)[None,:]<=end[:,None]
    return (torch.where(prefix[:,:,None],xy,torch.stack((new_x,new_y),-1)),
            torch.where(prefix,route.heading[rows],h),torch.where(prefix,route.curvature[rows],k))


class P0Branches:
    """The seven lanes of one prefix always share one native physics batch."""
    def __init__(self, cfg, out, query, route, contracts, q, v):
        self.cfg, self.out, self.query = cfg, out, query
        self.n = len(q)
        self.g = cfg['group_size']
        self.arm = torch.arange(self.n, device=q.device) % self.g
        self.section = q.new_tensor([c['section_x'] for c in contracts])
        self.side = q.new_tensor([c['side'] for c in contracts])
        self.next = torch.full_like(self.arm, -1)
        self.due = torch.full_like(self.arm, -1)
        self.received = torch.full_like(self.arm, -1)
        self.valid_until = torch.full_like(self.arm, -1)
        self.cover_lost = torch.zeros_like(self.arm, dtype=torch.bool)
        self.default_rows = torch.full_like(self.arm,-1)
        self.active_plan = torch.zeros_like(self.arm,dtype=torch.bool)
        self.version = torch.zeros_like(self.arm)
        self.pending_version = self.version.clone()
        self.changed_tick = torch.full_like(self.arm,-1)
        self.route_observed_tick = self.changed_tick.clone()
        self.route_observed_distance = q.new_zeros(self.n)
        self.await_actor = self.active_plan.clone()
        self.await_pursuit = self.active_plan.clone()
        self.queries = torch.zeros_like(self.arm)
        self.skipped_pending = torch.zeros_like(self.arm)
        self.uncovered_ticks = torch.zeros_like(self.arm)
        self.initial_uncovered_ticks = torch.zeros_like(self.arm)
        self.expired_uncovered_ticks = torch.zeros_like(self.arm)
        self.gate = torch.zeros_like(self.arm)
        self.xy = q[:, :2].clone()
        self.distance = torch.zeros_like(q[:, 0])
        self.start_distance = self.distance.clone()
        self.committed = self.distance.clone()
        self.deadline = cfg['deadline_ms']
        self.admitted = torch.zeros_like(self.arm, dtype=torch.bool)
        self.in_time = self.admitted.clone()
        self.pending_until = self.valid_until.clone()
        self.pair_ok = torch.ones(self.n//self.g, dtype=torch.bool, device=q.device)
        self.audit_done = torch.zeros_like(self.pair_ok)
        self.first_bad = torch.full_like(self.pair_ok, -1, dtype=torch.int64)
        self.suffix_ref = len(route.xy)
        for field in ('xy', 'heading', 'curvature'):
            old = getattr(route, field)
            setattr(route, field, torch.cat((old, old.new_zeros((self.n, *old.shape[1:])))))
        self.work = {k: torch.empty_like(getattr(route, k)[-self.n:]) for k in ('xy','heading','curvature')}
        budget = cfg.get('candidates', 16)
        self.latencies = q.new_tensor(cfg['latency_ms'])
        self.models = ContinuousExitPlanner(query, route, candidates=budget, iterations=2,
                                            robots_per_chunk=cfg.get('robots_per_chunk', 8))
        init_twist = velocity_state(q[:1],v[:1])
        self.models.plan(q[:1],init_twist,init_twist,route,q.new_tensor([1.]),self.section[:1],
                         self.side[:1],5.,rows=route.route_ids[:1])
        torch.cuda.synchronize()
        self.log = (out/'p0_events.jsonl').open('w', buffering=1)
        self.anchor_dir = out/'p0_handbacks'
        self.anchor_dir.mkdir()
        self.ref_dir = out/'p0_published'
        self.ref_dir.mkdir()

    def event(self, **values):
        self.log.write(json.dumps(values, allow_nan=False)+'\n')

    @staticmethod
    def runtime(env):
        names = ('phase','alive','recovering','stable_time','upper_ready_time','airborne','flags','far_time',
                 'entry_time','handoff_time','hold','lock','target','codes','durations','ownership_error',
                 'substep_loads','rb_events','pulse_start','pulse_ticks','query_pass','query_ms','release',
                 'pending_route','applied_tick','proposed','directions','velocity_history')
        return {k:env[k] for k in names}

    def audit(self, tick, env, just_locked):
        """Compare every executed prefix tick and capture full batched state for deterministic replay."""
        if tick==0:
            snapshot = capture(env['data'],env['ground_adapter'],(env['history'],env['obs'],env['last']),
                self.runtime(env),tick,dict(world_map=env['world'],terrain=env['terrain'],ground_identity=env['identity'],
                    rb_identity=env['skill'].record,physics_model=str(self.out/'terrain.mjb'),
                    resolved_config=str(self.out/'resolved_config.json')))
            with (self.out/'p0_replay_from_start.pkl').open('wb') as stream:
                pickle.dump(snapshot,stream,protocol=pickle.HIGHEST_PROTOCOL)
            del snapshot
        if self.audit_done.all():
            return
        state = {}
        for key, obj in (('middle',env['middle']),('policy',env['middle'].policy),('controller',env['ctl'])):
            state.update(lane_tensors(vars(obj),key,self.n))
        state.update(lane_tensors(self.runtime(env),'runtime',self.n))
        state['route'] = env['route'].route_ids
        # Action target is the actual held four-actuator request from the preceding physical transition.
        for k in ('qpos','qvel','act','qacc_warmstart','ctrl','qfrc_applied','eq_active','time'):
            state['physics_'+k] = torch.from_dlpack(getattr(env['data'],k))
        for k in ('history','obs','last','rb_events'):
            state.update({f'{k}_{i}':torch.from_dlpack(x) for i,x in enumerate(jax.tree.leaves(env[k]))})
        # Direction labels, unpublished query data and already-selected future route IDs differ by
        # construction before the robot moves. They are not actuator, plant or controller state.
        for k in ('proposed','directions','query_pass','query_ms','pending_route','release','applied_tick'):
            state.pop('runtime_'+k,None)
        columns = [v.reshape(self.n,-1).contiguous().view(torch.uint8) for v in state.values() if v.numel()]
        bits = torch.cat(columns,1).reshape(self.n//self.g,self.g,-1)
        same = (bits==bits[:,:1]).all((1,2))
        self.first_bad = torch.where(~self.audit_done & ~same & (self.first_bad<0),tick,self.first_bad)
        self.pair_ok &= same | self.audit_done
        locked = just_locked.reshape(-1,self.g)
        # Keep all seven rows when a group diverges. Such a group cannot be pooled as one handback.
        if bool(locked.any()):
            groups = torch.where(locked.any(1))[0]
            ids = (groups[:,None]*self.g + torch.arange(self.g,device=groups.device)).flatten()
            self.archive(tick, ids, env, state, just_locked)
        completed = locked.any(1) | (~env['alive']).reshape(-1,self.g).all(1)
        self.audit_done |= completed

    def archive(self, tick, ids, env, state, just_locked):
        # Exact native checkpoint is the common start plus the frozen pre-branch controls. Save
        # the entire MJX solver state at each real handback too, including packed global contacts.
        data = env['data']
        entries = {k:v[ids].detach().cpu().numpy() for k,v in state.items()}
        entries.update(lanes=ids.cpu().numpy(),locked=just_locked[ids].cpu().numpy())
        ordered, _ = jax.tree_util.tree_flatten_with_path(data)
        owner = torch.from_dlpack(data._impl.contact__worldid)
        used = torch.arange(len(owner),device=owner.device)<torch.from_dlpack(data._impl.nacon).reshape(-1)[0]
        contact_idx = torch.where(used & torch.isin(owner,ids))[0]
        entries['packed_contact_indices'] = contact_idx.cpu().numpy()
        for path, leaf in ordered:
            name = path[-1].name
            value = torch.from_dlpack(leaf)
            if name not in DATA_NON_VMAP:
                value = value[ids]
            elif name.startswith('contact__'):
                value = value[contact_idx]
            entries['native_'+name] = value.cpu().numpy()
        for i,(q,v) in enumerate(env['velocity_history']):
            entries[f'query_pose_history_{i}']=q[ids].cpu().numpy()
            entries[f'query_velocity_history_{i}']=v[ids].cpu().numpy()
        np.savez_compressed(self.anchor_dir/f'tick_{tick:04d}.npz', **entries)

    def step(self,tick,env,just_locked,previous):
        q, v, route, middle = (env[k] for k in ('q','v','route','middle'))
        alive, handoff = env['alive'], env['handoff_time']
        rolling = (self.arm>=3) & alive & (handoff>=0)
        step_xy = q[:,:2]
        old_xy = self.xy
        self.distance += (step_xy-old_xy).norm(dim=1)
        cross = rolling & (self.gate==0) & (old_xy[:,0]<self.section) & (self.section<=q[:,0])
        ratio = (self.section-old_xy[:,0])/(q[:,0]-old_xy[:,0]).clamp_min(1.e-7)
        signed = self.side*(old_xy[:,1]+ratio*(q[:,1]-old_xy[:,1]))
        self.gate = torch.where(cross,torch.where((signed>=.06)&(signed<=3.9),1,-1),self.gate)
        self.xy = step_xy.clone()
        locked_roles = just_locked & (self.arm>=3)
        if bool(locked_roles.any()):
            ids = torch.where(locked_roles)[0]
            current = velocity_state(q[ids],v[ids])
            # Score the old three-exit controller under each of the four actual gate contracts.
            candidates = (len(env['approach_x']) + (env['approach_ids'][ids,None]*3
                          +torch.arange(3,device=q.device))*len(env['arc64'])+env['join'][ids,None].clamp_max(len(route.s)-1))
            straight = torch.ones_like(ids)
            choice, passed, inc = self.query.choose(q[ids],current[:,0],env['nominal_continuation_speed'][ids],
                route,candidates,self.section[ids],self.side[ids],straight,min(5.1,10.-tick*.005),
                current,previous[ids])
            choice = torch.where(passed,choice,straight)
            self.event(kind='old_selection',tick=tick,lanes=ids.tolist(),choice=choice.tolist(),
                       default_route_id=route.route_ids[ids].tolist(),old_candidates=candidates.tolist(),
                       admitted=passed.tolist(),incumbent_admitted=inc.tolist())
            self.default_rows[ids]=candidates[:,1]
            self.next[ids]=tick
        # Exposed route arrays contain lane-private rows, so publication does not reset H75 history.
        due = rolling & (self.due==tick)
        if bool(due.any()):
            ids = torch.where(due)[0]
            required = (middle.projection['progress'][ids]+middle.follower['lookahead'][ids]+middle.v_user[ids]*.02).clamp_max(route.s[-1])
            publish = (self.admitted[ids] & self.in_time[ids] & (tick<=self.pending_until[ids])
                       & (self.version[ids]==self.pending_version[ids]) & (required<=self.committed[ids]+1e-6))
            rows = ids[publish]
            if len(rows):
                for key in self.work:
                    getattr(route,key)[self.suffix_ref+rows]=self.work[key][rows]
                route.route_ids[rows]=self.suffix_ref+rows
                self.valid_until[rows]=self.pending_until[rows]
                self.active_plan[rows]=True
                self.record_change(tick,rows,self.received[rows],self.start_distance[rows])
                np.savez_compressed(self.ref_dir/f'tick_{tick:04d}.npz',lanes=rows.cpu().numpy(),
                    **{k:self.work[k][rows].cpu().numpy() for k in self.work})
            self.event(kind='publication',tick=tick,lanes=ids.tolist(),sample_tick=self.received[ids].tolist(),
                       published=publish.tolist(),in_time=self.in_time[ids].tolist(),model_admitted=self.admitted[ids].tolist(),
                       pending_valid_until=self.pending_until[ids].tolist(),route_version=self.version[ids].tolist(),
                       route_id=route.route_ids[ids].tolist(),travel_m=(self.distance[ids]-self.start_distance[ids]).tolist())
            self.due[ids]=-1
        expired = rolling & self.active_plan & (tick>self.valid_until)
        if bool(expired.any()):
            ids = torch.where(expired)[0]
            required = (middle.projection['progress'][ids]+middle.follower['lookahead'][ids]+middle.v_user[ids]*.02).clamp_max(route.s[-1])
            fields = rejoin_default(route,route.route_ids[ids],self.default_rows[ids],required)
            for field,value in zip(('xy','heading','curvature'),fields):
                getattr(route,field)[self.suffix_ref+ids]=value
            route.route_ids[ids]=self.suffix_ref+ids
            self.active_plan[ids]=False
            self.record_change(tick,ids,tick,self.distance[ids])
            np.savez_compressed(self.ref_dir/f'fallback_{tick:04d}.npz',lanes=ids.cpu().numpy(),
                **{key:v.cpu().numpy() for key,v in zip(('xy','heading','curvature'),fields)})
            self.event(kind='expired_to_default',tick=tick,lanes=ids.tolist(),valid_until=self.valid_until[ids].tolist(),
                       route_version=self.version[ids].tolist(),default_route_id=self.default_rows[ids].tolist(),
                       committed_s=required.tolist(),route_id=route.route_ids[ids].tolist())
        query = rolling & (self.next==tick)
        self.next = torch.where(query,tick+self.cfg['schedule_ticks'],self.next)
        self.skipped_pending += (query & (self.due>=0)).long()
        query &= self.due<0
        if bool(query.any()):
            ids = torch.where(query)[0]
            progress = committed_progress(middle.projection['progress'][ids],middle.follower['lookahead'][ids],
                                          middle.v_user[ids],self.deadline/1000).clamp_max(route.s[-1])
            current = velocity_state(q[ids],v[ids])
            start=time.monotonic()
            proposed=self.models.plan(q[ids],current,previous[ids],route,progress,self.section[ids],self.side[ids],
                       10.-tick*.005,gate_status=self.gate[ids],hold_s=env['hold'][ids],rows=route.route_ids[ids])
            torch.cuda.synchronize()
            batch_ms=1000*(time.monotonic()-start)
            # Corresponding arms of the same physical prefix see identical latency samples.
            number = (ids//self.g + self.queries[ids]) % len(self.latencies)
            ms=self.latencies[number]
            self.received[ids]=tick
            self.pending_version[ids]=self.version[ids]
            # The path publication is visible only at the next absolute 20 ms policy boundary.
            physical_due = ((tick+torch.ceil(ms/5.).long()+3)//4)*4
            self.due[ids]=physical_due
            self.in_time[ids]=(physical_due-tick)*5<=self.deadline
            self.pending_until[ids]=tick+torch.floor(proposed.valid_for_s/.005).long()
            self.admitted[ids]=proposed.admitted
            self.committed[ids]=progress
            self.queries[ids]+=1
            self.start_distance[ids]=self.distance[ids]
            for k in self.work:
                self.work[k][ids]=getattr(proposed,k)
            self.event(kind='query',tick=tick,lanes=ids.tolist(),ms=ms.tolist(),due=physical_due.tolist(),
                       admitted=proposed.admitted.tolist(),incumbent=proposed.incumbent_admitted.tolist(),
                       sample_index=number.tolist(),offline_batch_wall_ms=batch_ms)
        uncovered = rolling & (tick>self.valid_until)
        self.cover_lost |= uncovered
        self.uncovered_ticks += uncovered.long()
        self.initial_uncovered_ticks += (uncovered & (self.version==0)).long()
        self.expired_uncovered_ticks += (uncovered & (self.version>0)).long()

    def record_change(self,tick,ids,observed_tick,observed_distance):
        replaced = ids[self.await_actor[ids] | self.await_pursuit[ids]]
        if len(replaced):
            self.event(kind='superseded_before_consumption',tick=tick,lanes=replaced.tolist(),
                       version=self.version[replaced].tolist(),actor_not_read=self.await_actor[replaced].tolist(),
                       pursuit_not_read=self.await_pursuit[replaced].tolist())
        self.changed_tick[ids]=tick
        self.route_observed_tick[ids]=observed_tick
        self.route_observed_distance[ids]=observed_distance
        self.version[ids]+=1
        self.await_actor[ids]=True
        self.await_pursuit[ids]=True

    def actor_command(self,tick,alive,phase,middle,target):
        # before() selects a new H75 latent exactly on this per-lane clock. Log the
        # actuator target actually sent; its value may coincide with that of the old route.
        consume = self.await_actor & alive & (phase==2) & (middle.lane_ticks%4==0)
        if bool(consume.any()):
            ids=torch.where(consume)[0]
            self.event(kind='h75_actor_consumed',tick=tick,lanes=ids.tolist(),
                       publication_tick=self.changed_tick[ids].tolist(),version=self.version[ids].tolist(),
                       observation_tick=self.route_observed_tick[ids].tolist(),
                       observed_to_applied_travel_m=(self.distance[ids]-self.route_observed_distance[ids]).tolist(),
                       applied_steer_rear=target[ids,:2].tolist())
            self.await_actor[ids]=False

    def pursuit_read(self,tick,alive,phase,middle):
        # after() has already refreshed the pursuit preview using the updated progress.
        consume = self.await_pursuit & alive & (phase==2) & (middle.lane_ticks%4==0)
        if bool(consume.any()):
            ids=torch.where(consume)[0]
            self.event(kind='pursuit_consumed',tick=tick,lanes=ids.tolist(),
                       publication_tick=self.changed_tick[ids].tolist(),version=self.version[ids].tolist(),
                       observation_tick=self.route_observed_tick[ids].tolist(),
                       observed_to_read_travel_m=(self.distance[ids]-self.route_observed_distance[ids]
                           +(middle.pose[ids,:2]-self.xy[ids]).norm(dim=1)).tolist(),
                       follower_request=middle.follower['command'][ids].tolist())
            self.await_pursuit[ids]=False

    def finish(self):
        self.log.close()
        np.savez_compressed(self.out/'p0_branch_audit.npz',pair_ok=self.pair_ok.cpu().numpy(),
             first_bad_tick=self.first_bad.cpu().numpy(),queries=self.queries.cpu().numpy(),
             ever_without_coverage=self.cover_lost.cpu().numpy(),uncovered_ticks=self.uncovered_ticks.cpu().numpy(),
             before_first_accepted_ticks=self.initial_uncovered_ticks.cpu().numpy(),
             after_expiry_uncovered_ticks=self.expired_uncovered_ticks.cpu().numpy(),
             skipped_pending=self.skipped_pending.cpu().numpy(),unpublished_due_tick=self.due.cpu().numpy(),
             gate_status=self.gate.cpu().numpy())
