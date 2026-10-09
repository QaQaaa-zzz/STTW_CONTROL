"""V3 bounded campaign on reused teleop physics and task-local RSL PPO."""
from pathlib import Path
import json
import os
import time
import pickle
import random
import hashlib
import subprocess
import numpy as np
from .direct_command_budget import ComputeBudget,BudgetStop


def write(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');os.replace(tmp,path)

def clean_numbers(value):
    if isinstance(value,dict):return {k:clean_numbers(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [clean_numbers(x) for x in value]
    if isinstance(value,(float,np.floating)):return float(value) if np.isfinite(value) else None
    if isinstance(value,(np.integer,np.bool_)):return value.item()
    return value

def notify(title,body):
    try:subprocess.run(['notify-send',title,body],timeout=2,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    except (OSError,subprocess.TimeoutExpired):pass

def load_spec(path):
    def unique(pairs):
        result={}
        for k,v in pairs:
            if k in result:raise ValueError('duplicate schema field '+k)
            result[k]=v
        return result
    spec=json.loads(Path(path).read_text(),object_pairs_hook=unique)
    canonical=json.loads((Path(__file__).resolve().parents[2]/'configs/STTW_Direct_Command_V3.json').read_text())
    # A separately frozen, user-authorized run may change its training budget.
    # Method, reward, evaluation, initialization and physical fields stay frozen.
    mutable={'ppo':('default_updates','future_total_updates_only_after_user_approval'),
        'budget':('default_policy_transitions','default_control_transitions_upper',
                  'additional_compute_wall_seconds','training_wall_seconds','compile_wall_seconds',
                  'engineering_wall_seconds','evaluation_wall_seconds','full_episode_wall_seconds')}
    expected=json.loads(json.dumps(canonical))
    if 'lower_controller' in spec:
        lower=spec['lower_controller']
        required={'source_checkpoint','source_declaration','sidecar_sha256','payload_sha256','alpha','path_capacity'}
        if set(lower)!=required or lower['alpha']!=1. or lower['path_capacity']!=3201:
            raise ValueError('invalid frozen lower transfer contract')
        expected['lower_controller']=lower
    for section,fields in mutable.items():
        for field in fields:
            value=spec.get(section,{}).get(field)
            if type(value) is not int or value<=0:raise ValueError('invalid run budget '+field)
            expected[section][field]=value
    if spec!=expected:raise ValueError('unknown or altered frozen V3 method field')
    updates=spec['ppo']['default_updates'];b=spec['budget']
    samples=updates*spec['ppo']['num_envs']*spec['ppo']['rollout_policy_steps']
    if (updates>spec['ppo']['future_total_updates_only_after_user_approval'] or
        b['default_policy_transitions']!=samples or
        b['default_control_transitions_upper']!=samples*spec['plant']['control_ticks_per_action'] or
        b['training_wall_seconds']>b['additional_compute_wall_seconds']):
        raise ValueError('inconsistent declared training budget')
    return spec

def validate_run_limits(spec,updates=None,compute_wall_budget=None):
    maximum=spec['ppo']['default_updates'];wall=spec['budget']['additional_compute_wall_seconds']
    updates=maximum if updates is None else updates
    compute_wall_budget=wall if compute_wall_budget is None else compute_wall_budget
    if type(updates) is not int or not 1<=updates<=maximum:
        raise ValueError(f'updates must be1..{maximum} for this frozen run config')
    if not 0<compute_wall_budget<=wall:raise ValueError(f'compute budget must be0..{wall}')
    return updates,compute_wall_budget

def flatten_logs(logs):
    flat={}
    for name,value in logs.items():
        if isinstance(value,dict):
            prefix={'raw_components':'raw_cost','effective_components':'effective_cost','scored_effective_components':'scored_cost'}.get(name,name)
            for key,array in value.items():flat[prefix+'_'+key]=array
        else:flat[name]=value
    return flat

def conditional_batch_reset(end,done,reset_one):
    """Skip all reset computation only when the entire batch has no done."""
    import jax
    import jax.numpy as jp
    return jax.lax.cond(jp.any(done),
        lambda args:jax.vmap(lambda state,finished:jax.lax.cond(finished,reset_one,lambda x:x,state))(*args),
        lambda args:args[0],(end,done))


class Campaign:
    def __init__(self,config,output):
        self.config=Path(config);self.spec=load_spec(config);self.out=Path(output)
        if (self.out/'manifest.json').exists():raise FileExistsError('immutable run output already exists')
        self.budget=ComputeBudget(output,self.spec);self.budget.recover_interrupted()
        self.status=dict(run_id=self.out.name,state='initializing',stage='interface',completed_updates=0,declared_updates=self.spec['ppo']['default_updates'],
            policy_transitions=0,control_ticks=0,initialization='fresh',training_seed=self.spec['ppo']['seed'])
        self.seen_cases=set();self._status()
    def _status(self,**kwargs):
        self.status.update(kwargs);self.status['last_update_epoch']=time.time();write(self.out/'status.json',self.status)
    def compile(self,name,fn,*args):
        import jax
        self._status(stage='compile',detail=name,state='running')
        with self.budget.measure('compile',name):return jax.jit(fn).lower(*args).compile()
    def initialize(self):
        import jax
        import jax.numpy as jp
        import torch
        from dataclasses import asdict
        from .direct_command_env import DirectCommandEnv
        from .runtime import configure_compilation_cache
        from .actuator import initial_actuator
        if jax.config.x64_enabled:raise ValueError('V3 preserves float32, no global x64 change')
        torch.set_num_threads(2);configure_compilation_cache()
        with self.budget.measure('compile','load original MJX model and initial forward'):
            self.env=DirectCommandEnv(self.spec);base=self.env.physics.initial();jax.block_until_ready(base)
        e=self.env;cc=e.cc
        expected=self.spec['reference']
        for field,key in [('wheelbase','wheelbase_m_expected'),('gravity','gravity_expected'),('cg_forward','cg_forward_m_expected'),('cg_height','cg_height_m_expected'),('trail','trail_m_expected')]:
            if not np.isclose(getattr(cc,field),expected[key],rtol=0,atol=1e-12):raise ValueError('frozen controller mismatch '+field)
        if not np.isclose(cc.caster,expected['caster_deg_expected']*np.pi/180,atol=1e-12):raise ValueError('caster mismatch')
        write(self.out/'manifest.json',dict(parent_revision='92afde6',source_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            reused={'physics':'teleop_env.py','control':'closed_loop_kernel.py','reference':'teleop_reference.py','PPO':'RSL3.2 task adapter'},
            shared_diff='optional exact-governed passthrough and zero-actuator preview logging; old defaults unchanged',
            config_sha256=hashlib.sha256(self.config.read_bytes()).hexdigest(),model=e.physics.bundle.identity,
            controller=asdict(e.cc),actuator=asdict(e.ac),physics_timestep=e.physics.model.opt.timestep,substeps=e.physics.substeps,
            identity=dict(actor=345,critic=346,latent=2,action_semantics='raw-relative speed/steer reference corrections',alpha_index=336),
            observation_mode='simulation_state_assisted',old_checkpoint_loaded=False,
            frozen_lower=None if e.lower is None else e.lower.provenance,
            initialization='fresh_upper_actor_critic_optimizer',
            preparation='original ECBC prepared bank; lower history/reset starts with each task'))
        write(self.out/'frozen_config.json',self.spec);write(self.out/'observation_action_schema.json',dict(network=self.spec['network'],action=self.spec['action']))
        conditions=jp.asarray(self.spec['initialization']['initial_conditions_speed_roll'])
        def initial(condition):
            speed,angle=condition;q=base.data.qpos;w,x,y,z=q[3:7];ca,sa=jp.cos(-angle/2),jp.sin(-angle/2)
            q=q.at[3:7].set(jp.array([ca*w-sa*x,ca*x+sa*w,ca*y-sa*z,ca*z+sa*y]))
            b=e.physics.bundle;vel=base.data.qvel.at[0].set(speed).at[b.rear_dof].set(-speed/.1).at[b.front_dof].set(-speed/.1)
            data=base.data.replace(qpos=q,qvel=vel,ctrl=base.data.ctrl.at[1].set(-speed/.1))
            data=e.physics.helpers._mjx.forward(e.physics.helpers.mjx_model,data)
            pose=e.physics.helpers.pose(data);raw=jp.array([speed,0.])
            return base.replace(data=data,raw=raw,actuator=initial_actuator(e.ac,speed/.1),reference_pose=pose,yaw_wrapped=pose[2],yaw_unwrapped=pose[2],
                governor=base.governor.replace(current_reference=raw,last_goal=raw))
        create=self.compile('eight declared initial conditions',jax.vmap(initial),conditions)
        with self.budget.measure('smoke','make eight complete states'):
            bank=create(conditions);jax.block_until_ready(bank)
        def prep(s):return jax.vmap(lambda s:e.physics._step(s,s.raw,True))(s)
        prep=self.compile('eight prepared-state physics',prep,bank)
        self.budget.reserve(5600,'eight x 700 prepare ticks');self._status(stage='preparation')
        with self.budget.measure('smoke','3.5s bank physical preparation'):
            for tick in range(700):
                bank,log=prep(bank)
                if tick%25==0:
                    jax.block_until_ready(bank)
                    if bool(jp.any(bank.failed)):raise RuntimeError('prepared state failed')
            jax.block_until_ready(bank);last=jax.device_get(log)
            passed=(~np.asarray(bank.failed))&(np.abs(last['phi'])<=.05)&(np.abs(last['phi_dot'])<=.10)&(np.abs(last['actual_delta'])<=.03)&(np.abs(last['actual_forward_speed']-np.asarray(conditions[:,0]))<=.15)
            write(self.out/'prepared_metrics.json',dict(passed=passed.tolist(),conditions=np.asarray(conditions).tolist(),
                roll=last['phi'].tolist(),roll_rate=last['phi_dot'].tolist(),speed=last['actual_forward_speed'].tolist(),steer=last['actual_delta'].tolist()))
            if not passed.all():raise RuntimeError('prepared state acceptance failed')
            self.bank=bank
            with (self.out/'prepared_bank.pkl').open('wb') as f:pickle.dump(jax.device_get(bank),f)
        self.sample=jax.tree.map(lambda x:x[3],bank) # prescribed 2.3m/s zero-roll prepared state
        with self.budget.measure('compile','trace direct command observation and log schema'):
            self.sample_state=e.reset(self.sample,jp.int32(0),jp.int32(0),jp.asarray(0.));e.set_log_template(self.sample_state)
    def preflight(self):
        import jax
        import jax.numpy as jp
        e=self.env
        rows=jp.zeros((2,16,3)).at[:,:,0].set(99.).at[:,0,:].set(jp.array([[0.,2.3,0.],[0.,2.3,.04]]))
        def init(rows):return jax.vmap(lambda r:e.reset(self.sample,jp.int32(0),jp.int32(0),jp.asarray(0.),r,jp.array([.5,.3])))(rows)
        reset=self.compile('two zero-policy interface resets',init,rows)
        with self.budget.measure('smoke','interface resets'):states=reset(rows);jax.block_until_ready(states)
        def chunk(s):
            def step(s,_):
                end,r,d,logs,obs=jax.vmap(lambda s:e.policy_step(s,jp.zeros(2),False))(s)
                return end,logs
            return jax.lax.scan(step,s,None,length=50)
        run=self.compile('one-second zero-policy interface chunk',chunk,states)
        self.budget.reserve(1600,'two 4s maximum zero-policy interface traces')
        with self.budget.measure('smoke','zero straight and mild-turn physical interfaces'):
            chunks=[]
            for _ in range(4):
                states,log=run(states);chunks.append(jax.device_get(log))
                if bool(jp.any(states.physical.failed)):raise RuntimeError('zero-policy interface physical failure')
            residual=np.concatenate([x['normalized_residual'] for x in chunks],axis=0)
            governed=np.concatenate([x['governed'] for x in chunks],axis=0);raw=np.concatenate([x['limited_command'] for x in chunks],axis=0)
            if e.lower is None:
                if np.max(np.abs(residual))>1e-6:raise RuntimeError('zero policy changed baseline interface')
            else:
                if any(np.asarray(x['lower_fault']).any() for x in chunks):raise RuntimeError('frozen lower nonfinite interface')
                if np.max(np.abs(residual))>1.+1e-6:raise RuntimeError('combined residual exceeds original authority')
            if np.max(np.abs(governed-raw))>1e-6:raise RuntimeError('zero upper policy changed reference')
            flat=flatten_logs(jax.tree.map(lambda *x:np.concatenate(x,axis=0),*chunks))
            np.savez_compressed(self.out/'interface_traces.npz',**flat)
            write(self.out/'interface_checks.json',dict(passed=True,episodes=2,seconds_each=4,max_residual=float(np.max(np.abs(residual))),
                max_reference_difference=float(np.max(np.abs(governed-raw))),final_steer=np.asarray(log['actual_delta'][-1,:,-1]).tolist(),
                final_yaw=np.asarray(log['yaw_unwrapped'][-1,:,-1]).tolist()))
    def setup_batch(self,n,log_mode="evaluation_full",reset_guard=False):
        if log_mode not in ("training_summary","evaluation_full"):raise ValueError(log_mode)
        import jax
        import jax.numpy as jp
        e=self.env;ids=jp.arange(n,dtype=jp.int32)
        fixed_alpha=self.spec.get('upper_alpha')
        alpha=(jp.full((n,),float(fixed_alpha)) if fixed_alpha is not None else jp.concatenate([jp.zeros(n//2),jp.ones(n-n//2)]))
        def reset_one(env_id,episode,alpha):
            key=jax.random.fold_in(jax.random.fold_in(jax.random.PRNGKey(self.spec['commands']['random_seed']),env_id),episode)
            index=jax.random.randint(jax.random.fold_in(key,812),(),0,8)
            if self.spec.get('preference_v5') and self.spec.get('training_stage')==1:
                from .direct_command_scenarios import schedule
                _,family,_=schedule(env_id,episode,2.6,self.spec)
                high=jp.asarray(self.spec['prepared_high_speed_indices'])
                choice=jax.random.randint(jax.random.fold_in(key,813),(),0,len(self.spec['prepared_high_speed_indices']))
                index=jp.where(family==1,high[choice],index)
            snap=jax.tree.map(lambda x:x[index],self.bank)
            return e.reset(snap,env_id,episode,alpha)
        reset=self.compile(f'{n} independent initial task cases',jax.vmap(reset_one),ids,jp.zeros(n,jp.int32),alpha)
        phase='smoke' if n==8 else 'pilot'
        with self.budget.measure(phase,'reset task cases'):
            states=reset(ids,jp.zeros(n,jp.int32),alpha);jax.block_until_ready(states)
        def advance(states,z):
            end,reward,done,logs,final_obs=jax.vmap(e.policy_step)(states,z)
            if reset_guard:
                nxt=conditional_batch_reset(end,done,lambda s:reset_one(s.env_id,s.episode_index+1,s.alpha))
            else:
                nxt=jax.vmap(lambda s,d:jax.lax.cond(d,lambda s:reset_one(s.env_id,s.episode_index+1,s.alpha),lambda s:s,s))(end,done)
            a,c,f,clip=jax.vmap(e.observation)(nxt)
            active=logs['active_tick'].astype(jp.float32)
            ev=logs['actual_forward_speed']-logs['limited_command'][:,:,0];ed=logs['actual_delta']-logs['limited_command'][:,:,1]
            base=jp.stack([active,ev**2*active,ed**2*active,(logs['peak_roll']>.3)*active,
                logs['cap_fraction']*active,logs['raw_cost']*active,logs['effective_cost']*active,
                logs['residual_clipped']*active,logs['final_command_clipped']*active,
                logs['reference_reference_clipped']*active,logs['reference_rate_clipped']*active,
                logs['e_psi_unwrapped']**2*active,logs['offsets'][:,:,0]*active,logs['offsets'][:,:,1]*active],axis=-1)
            raw_cost=jp.stack(list(logs['raw_components'].values()),axis=-1)*active[:,:,None]
            effective=jp.stack(list(logs['effective_components'].values()),axis=-1)*active[:,:,None]
            base=jp.concatenate([base,raw_cost,effective],axis=-1)
            windows=jp.stack([jp.ones_like(active),(logs['chi']==0)&(logs['g']==0),logs['chi']>0,logs['g']>0],axis=-1)
            if self.spec.get('preference_v5'):
                recovery=logs['recovery_phase'];conflict=(logs['chi']>0)&(logs['g']==0)
                windows=jp.stack([jp.ones_like(active),~(recovery|conflict),conflict,recovery],axis=-1)
            summary=jp.einsum('ntf,ntw->nwf',base,windows)
            diag=(jax.tree.map(lambda x:x[jp.array([0,n//2])],logs) if log_mode=='evaluation_full' else {})
            if self.spec.get('smooth_v4'):
                diag['all_component_cap_counts']={k:jp.sum(v & logs['active_tick']) for k,v in logs['component_capped'].items()}
                diag['all_active_ticks']=jp.sum(logs['active_tick'])
            if self.spec.get('preference_v5'):
                from .direct_command_policy import map_latent
                diag['all_proposal_dv']=map_latent(z,self.spec)[:,0]
                diag['all_governed_dv']=logs['offsets'][:,:,0]
                diag['all_valid_ticks']=logs['active_tick']
            if log_mode=='training_summary':
                diag['reset_event']=dict(done=done,env_id=nxt.env_id,episode_index=nxt.episode_index,alpha=nxt.alpha,
                    family=nxt.family,initial_heading_error=nxt.initial_heading_error,reference_pose=nxt.physical.reference_pose,
                    actual_yaw=nxt.physical.yaw_unwrapped,slew=nxt.slew,rows=nxt.rows)
            return nxt,a,c,f,reward,done,summary,end.physical.failed,end.fault,jp.max(logs['peak_roll'],axis=1),diag,final_obs
        advance=self.compile(f'{n} direct actions four physical ticks and terminal reset',advance,states,jp.zeros((n,2)))
        observe=self.compile(f'{n} observations',jax.vmap(e.observation),states)
        return states,advance,observe
    def case_manifest(self,states,phase):
        ids=np.asarray(states.env_id);episodes=np.asarray(states.episode_index)
        new=[i for i,key in enumerate(zip(ids,episodes)) if (phase,int(key[0]),int(key[1])) not in self.seen_cases]
        if not new:return
        rows=np.asarray(states.rows);family=np.asarray(states.family);slew=np.asarray(states.slew);alpha=np.asarray(states.alpha)
        initial_heading_error=np.asarray(states.initial_heading_error)
        reference_pose=np.asarray(states.physical.reference_pose);actual_yaw=np.asarray(states.physical.yaw_unwrapped)
        path=self.out/phase/'case_manifest.jsonl';path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('a') as f:
            for i in new:
                rec=dict(env_id=int(ids[i]),episode_index=int(episodes[i]),alpha=float(alpha[i]),upper_alpha=float(alpha[i]),lower_alpha=1. if self.env.lower is not None else None,
                         family=int(family[i]),initial_heading_error_rad=float(initial_heading_error[i]),initial_reference_pose=reference_pose[i].tolist(),
                         initial_actual_yaw_rad=float(actual_yaw[i]),slew=slew[i].tolist(),rows=rows[i].tolist())
                f.write(json.dumps(rec)+'\n');self.seen_cases.add((phase,int(ids[i]),int(episodes[i])))
    def write_reset_events(self,events,phase):
        """Events captured at reset, never reconstructed from rollout-end state."""
        path=self.out/phase/'case_manifest.jsonl';path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('a') as f:
            for t,i in zip(*np.nonzero(events['done'])):
                def get(key):return events[key][t,i]
                key=(phase,int(get('env_id')),int(get('episode_index')))
                if key in self.seen_cases:continue
                rec=dict(env_id=key[1],episode_index=key[2],alpha=float(get('alpha')),upper_alpha=float(get('alpha')),
                    lower_alpha=1. if self.env.lower is not None else None,family=int(get('family')),
                    initial_heading_error_rad=float(get('initial_heading_error')),initial_reference_pose=get('reference_pose').tolist(),
                    initial_actual_yaw_rad=float(get('actual_yaw')),slew=get('slew').tolist(),rows=get('rows').tolist())
                f.write(json.dumps(rec)+'\n');self.seen_cases.add(key)

    def checkpoint(self,algo,update,phase,states):
        import torch,jax
        from .direct_command_policy import export_actor
        root=self.out/phase/'checkpoints';root.mkdir(parents=True,exist_ok=True)
        path=root/f'update_{update:04d}.pt';temp=path.with_suffix('.tmp')
        if path.exists():raise FileExistsError(path)
        torch.save(dict(schema='sttw_direct_command_v3_345_346_2',update=update,phase=phase,config=self.spec,
            policy=algo.policy.state_dict(),optimizer=algo.optimizer.state_dict(),torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all(),numpy_rng=np.random.get_state(),python_rng=random.getstate(),
            environment_seed=self.spec['commands']['random_seed'],env_ids=np.asarray(states.env_id),episode_indices=np.asarray(states.episode_index),
            task_ticks=np.asarray(states.tick),environment_continuation='future resume reconstructs prepared physics; not frame-continuous'),temp)
        os.replace(temp,path)
        with (root/f'actor_{update:04d}.pkl').open('wb') as f:pickle.dump(jax.device_get(export_actor(algo.policy)),f)
        write(self.out/phase/'last_completed.json',dict(update=update,checkpoint=str(path.resolve()),initialization='fresh'))
    def train(self,phase,updates):
        import jax,jax.numpy as jp,torch
        from tensordict import TensorDict
        from torch.utils.tensorboard import SummaryWriter
        from .direct_command_ppo import make_algorithm
        n,steps=(8,16) if phase=='smoke' else (self.spec['ppo']['num_envs'],self.spec['ppo']['rollout_policy_steps'])
        states,advance,observe=self.setup_batch(n)
        def td(a,c):return TensorDict({'policy':torch.utils.dlpack.from_dlpack(a),'critic':torch.utils.dlpack.from_dlpack(c)},batch_size=[n])
        with self.budget.measure(phase,'fresh shared Actor/Critic initialization'):
            seed=self.spec['ppo']['seed'];torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);np.random.seed(seed);random.seed(seed)
            a,c,_,_=observe(states);obs=td(a,c);algo=make_algorithm(obs,steps,self.spec,'cuda')
            original={k:v.detach().clone() for k,v in algo.policy.state_dict().items()}
            self.checkpoint(algo,0,phase,states);self.case_manifest(states,phase)
            writer=SummaryWriter(str(self.out/'tensorboard'/phase));writer.add_scalar('progress/updates',0,0);writer.flush()
        completed=0;previous_duration=0.;stop=None
        for update in range(1,updates+1):
            if previous_duration and self.budget.remaining(phase)<1.1*previous_duration:stop='predicted_stage_budget';break
            self._status(stage=phase,state='running',stage_completed_updates=completed,stage_declared_updates=updates)
            self.budget.reserve(n*steps*4,f'{phase} full rollout {update}')
            start=time.monotonic();all_stats=[];all_rewards=[];all_families=[];fails=[];ends=[];peaks=[];diagnostics=[];clips=[]
            try:
                with self.budget.measure(phase,f'update {update}',estimate=1.1*previous_duration):
                    with torch.no_grad():
                        for t in range(steps):
                            if not bool(torch.isfinite(obs['policy']).all()&torch.isfinite(obs['critic']).all()):raise RuntimeError('policy_fault observation')
                            family=np.asarray(states.family);z=algo.act(obs);mu=algo.policy.action_mean.detach().clone()
                            if not bool(torch.isfinite(z).all()):raise RuntimeError('policy_fault latent')
                            diag_obs=obs['policy'][[0,n//2]].cpu().numpy()
                            states,a,c,f,r,d,stat,fail,fault,peak,diag,final_obs=advance(states,jax.dlpack.from_dlpack(z.detach().contiguous()))
                            if bool(jp.any(f)) or bool(jp.any(fault)):raise RuntimeError('policy_fault: nonfinite live observation')
                            obs=td(a,c);algo.process_env_step(obs,torch.utils.dlpack.from_dlpack(r),torch.utils.dlpack.from_dlpack(d),{})
                            all_stats.append(np.asarray(stat));all_rewards.append(np.asarray(r));all_families.append(family)
                            fails.append(np.asarray(fail));ends.append(np.asarray(d));peaks.append(np.asarray(peak))
                            dg=jax.device_get(diag);dg['actor_observation']=diag_obs;dg['latent_mean']=mu[[0,n//2]].cpu().numpy()
                            # pre-reset final observation is explicitly preserved for audit, not used to bootstrap a terminal.
                            dg['final_actor_observation']=np.asarray(final_obs[0])[[0,n//2]]
                            diagnostics.append(dg)
                            if bool(jp.any(d)):self.case_manifest(states,phase)
                        algo.compute_returns(obs)
                    sample_seconds=time.monotonic()-start
                    returns=algo.storage.returns.detach();values=algo.storage.values.detach();advantages=algo.storage.advantages.detach()
                    var=float(returns.var());ev=1-float((returns-values).var())/var if var>1e-12 else 0.
                    value_stats=dict(return_mean=float(returns.mean()),value_mean=float(values.mean()),advantage_mean=float(advantages.mean()),advantage_std=float(advantages.std()),explained_variance=ev)
                    opt_start=time.monotonic();metrics=algo.update();opt_seconds=time.monotonic()-opt_start
                    if metrics['nonfinite_stop']:stop='nonfinite_optimizer_epoch_rolled_back'
                    elif metrics['hard_kl_stop']:stop='hard_kl_epoch_rolled_back'
                    self.checkpoint(algo,update,phase,states);completed=update
                    stats=np.stack(all_stats);rews=np.stack(all_rewards);families=np.stack(all_families);failed=np.stack(fails);ended=np.stack(ends)
                    record=dict(update=update,sampling_model_update=update-1,mean_step_reward=float(rews.mean()),optimizer=clean_numbers(metrics),value=value_stats,
                        sample_seconds=sample_seconds,optimize_seconds=opt_seconds,policy_transitions=completed*n*steps,control_ticks_reserved=completed*n*steps*4,
                        groups={},stop_reason=stop)
                    names=['count','speed_mse_sum','steer_mse_sum','roll_violations','cost_cap','raw_cost','effective_cost','motor_clip','final_clip','reference_clip','reference_rate_clip','heading_mse_sum','speed_offset','steer_offset']
                    costnames=sorted(diagnostics[0]['raw_components'])
                    names += ['raw_'+x for x in costnames]+['effective_'+x for x in costnames]
                    alpha_groups=([(int(self.spec['upper_alpha']),slice(None))] if 'upper_alpha' in self.spec else [(0,slice(0,n//2)),(1,slice(n//2,n))])
                    for alpha,sl in alpha_groups:
                        for fam in [-1,0,1,2]:
                            mask=np.ones(families[:,sl].shape,bool) if fam==-1 else families[:,sl]==fam
                            for wi,window in enumerate(['all','ordinary','conflict','recovery']):
                                sums=(stats[:,sl,wi,:]*mask[:,:,None]).sum(axis=(0,1));count=sums[0]
                                if count<=0:continue
                                group={k:float(v/count) for k,v in zip(names,sums)};group['count']=int(count)
                                group['speed_rmse']=float(np.sqrt(sums[1]/count));group['steer_rmse']=float(np.sqrt(sums[2]/count))
                                group['heading_rmse']=float(np.sqrt(sums[11]/count));group['reward_resolution_warning']=bool(window in ('ordinary','recovery') and group['cost_cap']>.05)
                                group['failed_episodes']=int(np.sum(failed[:,sl]&mask));group['episode_ends']=int(np.sum(ended[:,sl]&mask))
                                group['failure_rate']=group['failed_episodes']/max(1,group['episode_ends'])
                                group['peak_roll']=float(np.max(np.where(mask,np.stack(peaks)[:,sl],0)))
                                label=f'alpha{alpha}/family{fam}/{window}';record['groups'][label]=group
                                if fam==-1:
                                    for key in ['speed_rmse','steer_rmse','heading_rmse','cost_cap','motor_clip','final_clip','roll_violations','failure_rate']:
                                        writer.add_scalar(label+'/'+key,group[key],update)
                        writer.add_scalar(f'alpha{alpha}/mean_step_reward',float(rews[:,sl].mean()),update)
                    record['actor_weight_delta_l2']=float(torch.sqrt(sum(((v-original[k])**2).sum() for k,v in algo.policy.state_dict().items() if k.startswith('actor.'))))
                    record['critic_weight_delta_l2']=float(torch.sqrt(sum(((v-original[k])**2).sum() for k,v in algo.policy.state_dict().items() if k.startswith('critic.'))))
                    record['seconds']=time.monotonic()-start
                    with (self.out/phase/'metrics.jsonl').open('a') as f:f.write(json.dumps(clean_numbers(record),allow_nan=False)+'\n')
                    writer.add_scalar('train/mean_step_reward',record['mean_step_reward'],update)
                    for k,v in metrics.items():
                        if isinstance(v,(int,float,bool)) and np.isfinite(v):writer.add_scalar('ppo/'+k,float(v),update)
                    writer.add_scalar('weights/actor_delta_l2',record['actor_weight_delta_l2'],update);writer.add_scalar('weights/critic_delta_l2',record['critic_weight_delta_l2'],update)
                    writer.flush()
                    flat=flatten_logs(jax.tree.map(lambda *x:np.stack(x),*diagnostics));np.savez_compressed(self.out/phase/f'diagnostic_{update:04d}.npz',**flat)
                    self._status(stage=phase,completed_updates=completed if phase=='pilot' else 0,stage_completed_updates=completed,
                        policy_transitions=completed*n*steps if phase=='pilot' else 0,control_ticks=completed*n*steps*4 if phase=='pilot' else 0,
                        last_reward=record['mean_step_reward'],training_stop_reason=stop)
                    print(json.dumps(dict(phase=phase,update=completed,reward=record['mean_step_reward'],kl=metrics['mean_kl'],kl0=metrics['kl_alpha0'],kl1=metrics['kl_alpha1'],actor_grad=metrics['actor_grad_norm'],seconds=record['seconds'])),flush=True)
                    if stop:break
            except BudgetStop:
                algo.storage.clear();stop='incomplete_rollout_or_optimization_budget_stop'
                saved=torch.load(self.out/phase/'checkpoints'/f'update_{completed:04d}.pt',map_location='cuda',weights_only=False)
                algo.policy.load_state_dict(saved['policy']);algo.optimizer.load_state_dict(saved['optimizer']);break
            previous_duration=time.monotonic()-start
        writer.close();self._status(training_stop_reason=stop,stage_completed_updates=completed)
        notify('STTW V3 training stage ended',f'{phase}: {completed}/{updates}')
        if phase=='smoke':
            if completed!=2 or stop:raise RuntimeError('engineering short test incomplete/abnormal; no pilot')
            if record['actor_weight_delta_l2']<=0 or record['critic_weight_delta_l2']<=0:raise RuntimeError('engineering gradients did not update both networks')
            write(self.out/'smoke/passed.json',dict(completed=2,actor_delta=record['actor_weight_delta_l2'],critic_delta=record['critic_weight_delta_l2']))
        return algo,completed
    def review(self,algo,completed):
        import jax,jax.numpy as jp
        from .direct_command_policy import DirectCommandActor,export_actor
        from .direct_command_scenarios import review_rows
        e=self.env;params=export_actor(algo.policy);actor=DirectCommandActor()
        main,random_rows,slew=review_rows(self.spec)
        write(self.out/'review/frozen_schedules.json',dict(main=main.tolist(),random=random_rows.tolist(),slew=slew.tolist(),seed=88001))
        alphas=jp.array([0.,0.,1.]);bypass=jp.array([True,False,False])
        def resets(rows):return jax.vmap(lambda a:e.reset(self.sample,jp.int32(77001),jp.int32(0),a,rows,jp.asarray(slew,jp.float32)))(alphas)
        reset=self.compile('three matched review states',resets,jp.asarray(main,jp.float32))
        with self.budget.measure('review','paired review state creation'):states=reset(jp.asarray(main,jp.float32));jax.block_until_ready(states)
        def chunk(states):
            def step(states,_):
                obs,_,fault,_=jax.vmap(e.observation)(states)
                mu=actor.apply(params,obs);z=jp.where((bypass|states.physical.failed)[:,None],jp.zeros_like(mu),mu)
                states=states.replace(fault=states.fault|(fault&~states.physical.failed)|~jp.all(jp.isfinite(z),axis=-1))
                end,r,done,logs,final=jax.vmap(e.policy_step)(states,z,bypass)
                logs['latent_mean']=jp.repeat(mu[:,None,:],4,axis=1)
                return end,(logs,obs)
            return jax.lax.scan(step,states,None,length=50)
        execute=self.compile('one-second three-method deterministic review chunk',chunk,states)
        methods=self.spec['evaluation']['methods']
        for case,rows in [('main',main),('random',random_rows)]:
            self._status(stage='review',state='running',review_case=case,completed_updates=completed)
            root=self.out/'review'/case;root.mkdir(parents=True,exist_ok=True)
            np.savetxt(root/'target_rows.csv',rows[rows[:,0]<99],delimiter=',',header='time_s,speed_m_s,steer_rad',comments='')
            self.budget.reserve(9600,f'{case}: three paired16s episodes upper bound')
            chunks=[];observations=[];partial=True
            def persist():
                if not chunks:return
                full=jax.tree.map(lambda *x:np.concatenate(x,axis=0),*chunks)
                flat=flatten_logs(full)
                for i,method in enumerate(methods):
                    data={k:v[:,i].reshape((-1,)+v.shape[3:]) for k,v in flat.items()}
                    mask=data['active_tick'];data={k:v[mask] for k,v in data.items()}
                    data['partial']=np.asarray(partial);data['checkpoint_update']=np.asarray(completed)
                    temp=root/(method+'.tmp.npz');np.savez_compressed(temp,**data);os.replace(temp,root/(method+'.npz'))
                np.savez_compressed(root/'actual_observations.npz',observations=np.concatenate(observations,axis=0))
            try:
                # All three run concurrently: elapsed applies to every full episode.
                with self.budget.measure('review',case+' three-method physical review',cap=self.spec['budget']['full_episode_wall_seconds']):
                    states=reset(jp.asarray(rows,jp.float32))
                    for second in range(16):
                        states,(logs,obs)=execute(states);jax.block_until_ready(states)
                        chunks.append(jax.device_get(logs));observations.append(np.asarray(obs));persist()
                        if bool(jp.any(states.fault)):raise RuntimeError('policy_fault review; stop remaining list')
                        if bool(jp.all(states.physical.failed|(states.tick>=3200))):partial=False;break
                    partial=False;persist()
            finally:persist()
            from .direct_command_reporting import generate_report
            with self.budget.measure('review',case+' physical comparison plots'):
                generate_report(self.out,self.spec)
        self._status(stage='complete',state='complete',completed_updates=completed)
        notify('STTW V3 pipeline ended',f'checkpoint {completed}; comparison evidence saved')

def run(config,output,updates=None,compute_wall_budget=None,dry_run=False):
    c=Campaign(config,output)
    updates,compute_wall_budget=validate_run_limits(c.spec,updates,compute_wall_budget)
    c.budget.total_limit=min(c.budget.total_limit,compute_wall_budget)
    c._status(declared_updates=updates)
    if dry_run:c._status(stage='dry_run',state='dry_run');return
    try:
        c.initialize();c.preflight();c.train('smoke',2)
        algo,completed=c.train('pilot',updates)
        if c.status.get('training_stop_reason')=='nonfinite_optimizer_epoch_rolled_back':
            raise RuntimeError('nonfinite optimizer epoch rolled back; saved finite checkpoint, no review')
        if completed>0:c.review(algo,completed)
        else:c._status(state='budget_stopped',reason='no completed pilot checkpoint')
    except Exception as exc:
        c._status(state='budget_stopped' if isinstance(exc,BudgetStop) else 'error',reason=str(exc))
        notify('STTW V3 stopped',str(exc));raise
    finally:
        try:
            from .direct_command_reporting import generate_report
            with c.budget.measure('checks','final evidence report'):generate_report(c.out,c.spec)
        except Exception as exc:write(c.out/'report_error.json',dict(reason=str(exc)))
