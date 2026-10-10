"""Fresh bounded lower-controller training; reuses RSL and teleop physics."""
from pathlib import Path
import json,os,time,pickle,random,hashlib,subprocess,math
import numpy as np
from .direct_command_training import write,clean_numbers,notify

def make_algorithm(obs,steps,cfg,device='cuda'):
    import torch
    from rsl_rl.modules import ActorCritic
    from rsl_rl.storage import RolloutStorage
    from .direct_command_ppo import DirectCommandPPO
    assert obs['policy'].shape[-1]==cfg['actor_dim'] and obs['critic'].shape[-1]==cfg['critic_dim']
    policy=ActorCritic(obs,{'policy':['policy'],'critic':['critic']},2,actor_hidden_dims=cfg['hidden_sizes'],critic_hidden_dims=cfg['hidden_sizes'],activation='elu',init_noise_std=cfg['std'],noise_std_type='log',state_dependent_std=False,actor_obs_normalization=False,critic_obs_normalization=False).to(device)
    with torch.no_grad():
        for net in [policy.actor,policy.critic]:
            layers=[x for x in net.modules() if isinstance(x,torch.nn.Linear)]
            for layer in layers:
                torch.nn.init.orthogonal_(layer.weight,math.sqrt(2));torch.nn.init.zeros_(layer.bias)
            if net is policy.actor:torch.nn.init.zeros_(layers[-1].weight)
            else:torch.nn.init.orthogonal_(layers[-1].weight,1.)
    storage=RolloutStorage('rl',obs.batch_size[0],steps,obs,[2],device=device)
    return DirectCommandPPO(policy,storage,num_learning_epochs=cfg['epochs'],num_mini_batches=cfg['minibatches'],clip_param=cfg['clip_ratio'],gamma=cfg['gamma'],lam=cfg['gae_lambda'],value_loss_coef=1.,entropy_coef=cfg['entropy'],use_clipped_value_loss=False,schedule='fixed',desired_kl=cfg['soft_kl'],normalize_advantage_per_mini_batch=False,device=device,actor_lr=cfg['actor_lr'],critic_lr=cfg['critic_lr'],adam_betas=(.9,.999),adam_epsilon=1e-5,weight_decay=0.,std_min=cfg['std_min'],std_max=cfg['std_max'],hard_kl=cfg['hard_kl'],actor_grad_limit=1.,critic_grad_limit=1.,kl_group_index=None)

class Training:
    def __init__(self,config,output,authorize_local_training=False,resume_checkpoint=None,target_updates=None):
        self.cfg=json.loads(Path(config).read_text())
        if self.cfg.get('candidate_local_interface'):
            self.cfg['formal_training_authorized']=bool(authorize_local_training)
            if not authorize_local_training:raise PermissionError('local candidate training resources not authorized')
        self.resume_checkpoint=resume_checkpoint;self.resume_metadata={}
        if target_updates is not None:
            if target_updates>self.cfg.get('max_total_updates',self.cfg['updates']):raise ValueError('target exceeds declared total cap')
            self.cfg['updates']=target_updates
        self.out=Path(output);self.started=time.monotonic()
        if resume_checkpoint:resume_preflight(resume_checkpoint,self.out,self.cfg)
        if (self.out/'manifest.json').exists() and not resume_checkpoint:raise FileExistsError('run already exists; no overwrite/resume')
        self.status=dict(state='initializing',run_id=self.out.name,declared_updates=self.cfg['updates'],completed_updates=0,training_transitions=0,initialization='fresh_actor_critic_optimizer',training_seed=self.cfg['seed'])
        self.update_status();write(self.out/('resume_config.json' if resume_checkpoint else 'config.json'),self.cfg)
    def update_status(self,**kw):
        self.status.update(kw);self.status.update(updated_epoch=time.time(),elapsed_s=time.monotonic()-self.started);write(self.out/'status.json',self.status)
    def check_budget(self):
        if self.cfg.get('wall_budget_s') is not None and time.monotonic()-self.started>self.cfg['wall_budget_s']:raise TimeoutError('declared wall budget exhausted')
    def compile(self,label,fn,*args):
        import jax
        self.check_budget();self.update_status(stage='compile',detail=label,state='running');start=time.monotonic()
        result=jax.jit(fn).lower(*args).compile();self.check_budget()
        with (self.out/'compile.jsonl').open('a') as f:f.write(json.dumps(dict(label=label,seconds=time.monotonic()-start))+'\n')
        return result
    def initialize(self):
        import jax,jax.numpy as jp,torch
        from dataclasses import asdict
        from .lower_command import LowerCommandEnv
        from .runtime import configure_compilation_cache
        torch.set_num_threads(2);configure_compilation_cache(self.out/'jax_cache')
        self.env=LowerCommandEnv(self.cfg)
        if self.cfg.get('candidate_local_interface'):
            from .lower_command import SCALES
            np.testing.assert_allclose(np.asarray(SCALES),self.cfg['observation_scales'],rtol=1e-7,atol=0)
            assert self.cfg['actor_dim']==210 and self.cfg['critic_dim']==211 and self.cfg['history_frames']==10
            assert self.cfg['num_envs']*self.cfg['rollout_steps']%self.cfg['minibatches']==0
        with Path(self.cfg['prepared_bank']).open('rb') as f:self.bank=jax.tree.map(jp.asarray,pickle.load(f))
        source=json.loads(Path(self.cfg['prepared_manifest']).read_text())
        assert self.env.physics.bundle.identity==source['model'],'prepared bank physics identity mismatch'
        assert asdict(self.env.cc)==source['controller'] and asdict(self.env.ac)==source['actuator']
        if self.cfg.get('prepared_bank_expected_sha256') and hashlib.sha256(Path(self.cfg['prepared_bank']).read_bytes()).hexdigest()!=self.cfg['prepared_bank_expected_sha256']:raise ValueError('prepared bank SHA mismatch')
        manifest_path=self.out/(f'resume_manifests/to_{self.cfg["updates"]:04d}.json' if self.resume_checkpoint else 'manifest.json')
        write(manifest_path,dict(schema=self.cfg['schema'],config=self.cfg,source_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),source_patch=subprocess.check_output(['git','diff'],text=True),model=self.env.physics.bundle.identity,controller=asdict(self.env.cc),actuator=asdict(self.env.ac),prepared_bank_sha256=hashlib.sha256(Path(self.cfg['prepared_bank']).read_bytes()).hexdigest(),policy_source=None,optimizer_source=None,action_semantics='tanh Gaussian -> front/rear bounded residual added to ECBC',actor_dim=210,critic_dim=211,frequency_hz=200))
        if self.cfg.get('candidate_local_interface'):
            # Existing lower/ECBC interface validation is retained; do not repeat it.
            write(self.out/'interface_checks.json',dict(status='reused_existing_interface_no_new_parity_physics',candidate_checks='learning/tests/test_local_lower_candidate.py'))
            return
        # Complete physical/controller/actuator parity, not merely matching motor commands.
        sample=jax.tree.map(lambda x:x[3],self.bank);s=self.env.reset(sample,jp.int32(0),jp.int32(0))
        def parity(s):
            a,_,_,log=self.env.step(s,jp.zeros(2));b,base=self.env.physics._step(s.physical,s.physical.raw,True,exact_governed=True)
            return a.physical,b,log,base
        fn=self.compile('zero-residual ECBC ESO physics parity',parity,s);a,b,log,base=fn(s);jax.block_until_ready(a)
        for name in ['controller','actuator','data']:
            leaves1=jax.tree.leaves(getattr(a,name));leaves2=jax.tree.leaves(getattr(b,name))
            for x,y in zip(leaves1,leaves2):np.testing.assert_allclose(np.asarray(x),np.asarray(y),rtol=1e-5,atol=1e-6)
        write(self.out/'interface_checks.json',dict(zero_residual_physics_parity=True,eso_once=True,history_valid_frames=int(np.asarray(s.mask).sum()),command_schedule='16rows',dt=.005))
    def batch(self,n):
        import jax,jax.numpy as jp
        e=self.env
        def reset_one(eid,episode):
            key=jax.random.fold_in(jax.random.fold_in(jax.random.PRNGKey(self.cfg['seed']),eid),episode)
            idx=jax.random.randint(key,(),0,8);snap=jax.tree.map(lambda x:x[idx],self.bank)
            return e.reset(snap,eid,episode)
        reset=self.compile(f'{n} resets',jax.vmap(reset_one),jp.arange(n,dtype=jp.int32),jp.zeros(n,jp.int32))
        states=reset(jp.arange(n,dtype=jp.int32),jp.zeros(n,jp.int32));jax.block_until_ready(states)
        def advance(states,z):
            end,reward,done,logs=jax.vmap(e.step)(states,z)
            nxt=jax.vmap(lambda s,d:jax.lax.cond(d,lambda x:reset_one(x.env_id,x.episode+1),lambda x:x,s))(end,done)
            a,c=jax.vmap(e.observation)(nxt)
            ev=logs['actual_forward_speed']-logs['limited_command'][:,0];ed=logs['actual_delta']-logs['limited_command'][:,1]
            # Nonfinite errors are logged separately; training stops instead of treating them as zero.
            stats=jp.stack([jp.mean(ev**2),jp.mean(ed**2),jp.mean(jp.abs(ev)),jp.mean(jp.abs(ed)),jp.mean(jp.abs(logs['phi'])>.3),jp.mean(end.physical.failed),jp.mean(done),jp.mean(jp.abs(logs['normalized_residual'])>=.99),jp.max(logs['peak_roll']),jp.mean(logs['actual_forward_speed']),jp.mean(logs['limited_command'][:,0]),jp.mean(logs['limited_command'][:,1]),jp.mean(logs['final_command_clipped'])])
            parts={k:jp.mean(v) for k,v in logs['reward_parts'].items()};diag=jax.tree.map(lambda x:x[:2],logs)
            sampling={}
            if self.cfg['command_design'].get('version')=='four_local_families_v2':
                labels=states.task_labels
                values=jp.column_stack([logs['target'],logs['limited_command'],logs['raw_rates']])
                sampling['min']=jp.min(values,axis=0);sampling['max']=jp.max(values,axis=0)
                for family in range(4):
                    mask=labels[:,0]==family;count=jp.sum(mask)
                    sampling['family_'+str(family)]=jp.array([count,jp.sum(jp.where(mask,reward,0.)),jp.sum(jp.where(mask,ev**2,0.)),jp.sum(jp.where(mask,ed**2,0.)),jp.sum(mask&end.physical.failed),jp.sum(mask&done),jp.sum(mask&(logs['peak_roll']>.302))])
                sampling['completed_episodes']=jp.sum(done)
                sampling['completed_episode_totals']=jp.sum(logs['completed_episode_totals'],axis=0)
                for family in range(4):
                    mask=(labels[:,0]==family)&done
                    sampling['completed_family_'+str(family)]=jp.concatenate([jp.array([jp.sum(mask)]),jp.sum(jp.where(mask[:,None],logs['completed_episode_totals'],0.),axis=0)])
                sampling['labels']=jp.array([jp.sum(labels[:,1]),jp.sum(labels[:,2]),jp.sum(labels[:,3]>0),jp.sum(labels[:,4]>0)])
                for key,value in logs['reward_parts'].items():
                    if key in self.cfg['reward']['caps']:sampling['cap_'+key]=jp.mean(abs(value)>=self.cfg['reward']['scale']*self.cfg['dt']*self.cfg['reward']['caps'][key]-1e-7)
            return nxt,a,c,reward,done,stats,parts,diag,sampling
        advance=self.compile(f'{n} residual control physics and reset',advance,states,jp.zeros((n,2)))
        observe=self.compile(f'{n} observations',jax.vmap(e.observation),states)
        return states,advance,observe
    def checkpoint(self,algo,update,phase,states,name=None):
        import torch
        root=self.out/phase/'checkpoints';root.mkdir(parents=True,exist_ok=True);path=root/(name or f'update_{update:04d}.pt');tmp=path.with_suffix('.tmp')
        import jax
        full=(phase=='training' and (update%self.cfg.get('resume',{}).get('save_full_state_every',50)==0 or name=='resume_boundary.pt' or update==self.cfg['updates'])) and name!='training_reward_best.pt'
        data=dict(schema=self.cfg['schema'],update=update,config=self.cfg,policy=algo.policy.state_dict(),optimizer=algo.optimizer.state_dict(),accepted_policy_updates=algo.accepted_policy_updates,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),numpy_rng=np.random.get_state(),python_rng=random.getstate(),observation_scales='lower_command.SCALES',environment_continuation='full_rollout_boundary' if full else 'weights_only_not_exact_resume',resume_metadata=getattr(self,'resume_metadata',{}))
        if full:data['environment_pickle']=pickle.dumps(jax.device_get(states),protocol=pickle.HIGHEST_PROTOCOL)
        torch.save(data,tmp);os.replace(tmp,path)
        return str(path.resolve())
    def train(self,phase):
        import jax,jax.numpy as jp,torch
        from tensordict import TensorDict
        from torch.utils.tensorboard import SummaryWriter
        n,steps,updates=(8,16,2) if phase=='smoke' else (self.cfg['num_envs'],self.cfg['rollout_steps'],self.cfg['updates'])
        states,advance,observe=self.batch(n)
        torch.manual_seed(self.cfg['seed']);torch.cuda.manual_seed_all(self.cfg['seed']);np.random.seed(self.cfg['seed']);random.seed(self.cfg['seed'])
        def td(a,c):return TensorDict({'policy':torch.utils.dlpack.from_dlpack(a),'critic':torch.utils.dlpack.from_dlpack(c)},batch_size=[n])
        a,c=observe(states);obs=td(a,c);algo=make_algorithm(obs,steps,self.cfg)
        initial_update=0
        if phase=='training' and self.resume_checkpoint:
            states,initial_update,self.resume_metadata=restore_boundary(self.resume_checkpoint,algo,self.cfg)
            a,c=observe(states);obs=td(a,c)
            self.update_status(initialization='exact_resume_full_rollout_boundary',resume_checkpoint=str(self.resume_checkpoint),completed_updates=initial_update,training_transitions=initial_update*n*steps,accepted_policy_updates=algo.accepted_policy_updates)
            if self.resume_metadata.get('accepted_count_correction'):
                write(self.out/'accepted_count_correction.json',self.resume_metadata['accepted_count_correction'])
        else:self.checkpoint(algo,0,phase,states)
        writer=SummaryWriter(str(self.out/'tensorboard'/phase));writer.add_scalar('progress/updates',0,0);writer.flush()
        self.update_status(stage=phase,state='running',stage_updates=initial_update)
        best=self.resume_metadata.get('train_best_reward',-float('inf'));completed=initial_update;hard_rejects=self.resume_metadata.get('hard_rejects',0);boundary=True
        try:
            for update in range(initial_update+1,updates+1):
                self.check_budget();start=time.monotonic();stat=[];rews=[];parts=[];diags=[];sampling=[];boundary=False
                original={k:v.detach().clone() for k,v in algo.policy.state_dict().items() if k.startswith('actor.')}
                with torch.no_grad():
                    for t in range(steps):
                        self.check_budget();z=algo.act(obs)
                        if not bool(torch.isfinite(z).all()):raise RuntimeError('nonfinite latent')
                        states,a,c,r,d,st,pt,dg,sm=advance(states,jax.dlpack.from_dlpack(z.detach().contiguous()))
                        obs=td(a,c)
                        if not bool(torch.isfinite(obs['policy']).all()) or not bool(jp.all(jp.isfinite(r))) or not bool(jp.all(jp.isfinite(st))):raise RuntimeError('nonfinite observation/reward/physical diagnostics')
                        algo.process_env_step(obs,torch.utils.dlpack.from_dlpack(r),torch.utils.dlpack.from_dlpack(d),{})
                        stat.append(np.asarray(st));rews.append(float(jp.mean(r)));parts.append(jax.device_get(pt));diags.append(jax.device_get(dg));sampling.append(jax.device_get(sm))
                    algo.compute_returns(obs)
                sample_seconds=time.monotonic()-start;mean_reward=float(np.mean(rews))
                if mean_reward>best:
                    best=mean_reward;path=self.checkpoint(algo,update-1,phase,states,'training_reward_best.pt');write(self.out/phase/('train_best_candidate.json' if self.cfg.get('candidate_local_interface') else 'best_model.json'),dict(checkpoint=path,update=update-1,scored_by_rollout=update,mean_step_reward=best,selection='random training rollout, pre-update actor, not fixed-scene qualification'))
                opt=time.monotonic();metrics=algo.update();account_accepted_update(algo,metrics);opt_seconds=time.monotonic()-opt;completed=update;boundary=True
                if self.cfg.get('candidate_local_interface'):
                    hard_rejects,stop=handle_candidate_ppo_result(algo,metrics,hard_rejects,self.cfg)
                else:stop=None
                self.resume_metadata=dict(hard_rejects=hard_rejects,train_best_reward=best)
                arr=np.asarray(stat);avg=arr.mean(0);actor_delta=float(torch.sqrt(sum(((v-original[k])**2).sum() for k,v in algo.policy.state_dict().items() if k in original)))
                record=dict(update=update,transitions=update*n*steps,mean_step_reward=mean_reward,speed_rmse=float(np.sqrt(avg[0])),steer_rmse=float(np.sqrt(avg[1])),speed_mae=float(avg[2]),steer_mae=float(avg[3]),working_roll_fraction=float(avg[4]),failed_episodes=int(round(arr[:,5].sum()*n)),ended_episodes=int(round(arr[:,6].sum()*n)),residual_saturation_fraction=float(avg[7]),peak_roll=float(arr[:,8].max()),actual_speed_mean=float(avg[9]),command_speed_mean=float(avg[10]),command_steer_mean=float(avg[11]),logged_final_clip_fraction=float(avg[12]),sample_seconds=sample_seconds,optimize_seconds=opt_seconds,seconds=time.monotonic()-start,actor_weight_delta_l2=actor_delta,ppo=metrics,reward_parts={k:float(np.mean([x[k] for x in parts])) for k in parts[0]})
                if sampling and sampling[0]:
                    record['command_sampling']=aggregate_sampling(sampling,n,steps)
                record['accepted_policy_updates']=algo.accepted_policy_updates
                record['failure_rate']=record['failed_episodes']/record['ended_episodes'] if record['ended_episodes'] else None
                with (self.out/phase/'metrics.jsonl').open('a') as f:f.write(json.dumps(clean_numbers(record),allow_nan=False)+'\n')
                for k,v in record.items():
                    if isinstance(v,(int,float)) and np.isfinite(v):writer.add_scalar('train/'+k,v,update)
                for k,v in metrics.items():
                    if isinstance(v,(int,float)) and np.isfinite(v):writer.add_scalar('ppo/'+k,v,update)
                for k,v in record['reward_parts'].items():writer.add_scalar('reward_parts/'+k,v,update)
                writer.flush()
                if update==1 or update%self.cfg['save_every']==0 or update==updates or metrics['hard_kl_stop'] or metrics['nonfinite_stop']:
                    path=self.checkpoint(algo,update,phase,states);write(self.out/phase/'last_completed.json',dict(update=update,checkpoint=path))
                    from .direct_command_training import flatten_logs
                    full=flatten_logs(jax.tree.map(lambda *x:np.stack(x),*diags));np.savez_compressed(self.out/phase/f'diagnostic_{update:04d}.npz',**full)
                self.update_status(stage=phase,state='running',stage_updates=update,accepted_policy_updates=algo.accepted_policy_updates,completed_updates=update if phase=='training' else 0,training_transitions=update*n*steps if phase=='training' else 0,last_reward=mean_reward,last_speed_rmse=record['speed_rmse'],last_steer_rmse=record['steer_rmse'],last_actor_grad=metrics['actor_grad_norm'],last_update_seconds=record['seconds'])
                print(json.dumps(dict(phase=phase,update=update,reward=mean_reward,speed_rmse=record['speed_rmse'],steer_rmse=record['steer_rmse'],grad=metrics['actor_grad_norm'],kl=metrics['mean_kl'],seconds=record['seconds'])),flush=True)
                if getattr(self,'stop_requested',False):
                    self.checkpoint(algo,completed,phase,states,'resume_boundary.pt')
                    raise KeyboardInterrupt('requested stop saved at complete rollout boundary')
                if self.cfg.get('candidate_local_interface'):
                    if stop:raise RuntimeError(stop)
                    if phase=='training' and update in self.cfg['validation_updates']:
                        from .lower_command_validation import evaluate_fixed
                        path=self.checkpoint(algo,update,phase,states)
                        self.update_status(stage='fixed_validation',stage_updates=update)
                        summary=evaluate_fixed(self,algo,update,path)
                        for key in ('physical_quality_score','physical_failure_cases','working_limit_failure_cases','primary_failure_cases','joint_final_hold_failure_cases'):
                            writer.add_scalar('validation/'+key,summary[key],update)
                        writer.flush()
                elif metrics['hard_kl_stop'] or metrics['nonfinite_stop']:
                    raise RuntimeError('PPO numerical/KL stop, saved last finite checkpoint')
            if phase=='training' and self.cfg.get('candidate_local_interface'):
                notify('STTW lower training stage ended',f'{phase} {completed}/{updates}')
                from .lower_command_validation import load_best,evaluate_fixed
                # Last is already checkpointed. Final verification reloads immutable best.
                self.checkpoint(algo,completed,phase,states,'resume_boundary.pt')
                boundary=False
                best_record=load_best(self.out/'training',algo.policy)
                evaluate_fixed(self,algo,best_record['source_update'],best_record['checkpoint'],select=False)
                self.update_status(best_source_update=best_record['source_update'],best_qualified=best_record['qualified'],adoption_status='not_adopted')
            if phase=='smoke':
                assert actor_delta>0 and metrics['actor_grad_norm']>0 and metrics['critic_grad_norm']>0
                write(self.out/'smoke/passed.json',dict(updates=completed,finite_gradients=True,actor_delta=actor_delta))
        finally:
            if phase=='training' and boundary:
                self.checkpoint(algo,completed,phase,states,'resume_boundary.pt')
            writer.close()
        if not (phase=='training' and self.cfg.get('candidate_local_interface')):
            notify('STTW lower training stage ended',f'{phase} {completed}/{updates}')

def handle_candidate_ppo_result(algo,metrics,consecutive,cfg):
    # DirectCommandPPO already restored rejected epoch Actor/Critic/Adam/RNG.
    # Match V5.2: one hard rejection reduces Actor LR; fresh rollout follows.
    if metrics['nonfinite_stop']:return consecutive,'nonfinite_optimizer'
    consecutive=consecutive+1 if metrics['hard_kl_stop'] else 0
    if metrics['hard_kl_stop']:
        algo.optimizer.param_groups[0]['lr']=max(cfg['actor_lr_floor'],algo.optimizer.param_groups[0]['lr']/2)
        algo.hard_kl_stop=False
    return consecutive,'three_consecutive_hard_rejected_batches' if consecutive>=cfg['hard_reject_consecutive_stop'] else None


def run(config,output,smoke_only=False,authorize_local_training=False,formal_only=False,resume_checkpoint=None,target_updates=None):
    cfg=json.loads(Path(config).read_text())
    if cfg.get('candidate_local_interface') and not authorize_local_training:
        raise PermissionError('local candidate is preparation only; explicitly confirm training resources and pass --authorize-local-training')
    t=Training(config,output,authorize_local_training,resume_checkpoint,target_updates)
    import signal
    def request_stop(signum,frame):t.stop_requested=True
    signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGINT,request_stop)
    try:
        t.initialize()
        if not formal_only and not resume_checkpoint:t.train('smoke')
        if not smoke_only:t.train('training')
        t.update_status(state='completed',stage='complete');notify('STTW lower pipeline complete',str(t.out))
    except BaseException as exc:
        t.update_status(state='interrupted' if isinstance(exc,KeyboardInterrupt) else 'budget_stopped' if isinstance(exc,TimeoutError) else 'error',reason=str(exc));notify('STTW lower training stopped',str(exc));raise


def restore_boundary(path,algo,cfg):
    """Restore learner, full simulator/ESO/actuator/history and all RNG at boundary."""
    import torch,jax,jax.numpy as jp
    saved=torch.load(path,map_location=algo.policy.log_std.device if hasattr(algo.policy,'log_std') else 'cpu',weights_only=False)
    if saved['environment_continuation']!='full_rollout_boundary':raise ValueError('checkpoint is not an exact resume boundary')
    ignored={'updates','formal_transitions','status'}
    if {k:v for k,v in saved['config'].items() if k not in ignored}!={k:v for k,v in cfg.items() if k not in ignored}:raise ValueError('resume config identity mismatch')
    algo.policy.load_state_dict(saved['policy']);algo.optimizer.load_state_dict(saved['optimizer']);algo.accepted_policy_updates=saved['accepted_policy_updates']
    if algo.accepted_policy_updates==0 and saved['update']>0:
        metrics_path=Path(path).parent.parent/'metrics.jsonl'
        if metrics_path.exists():
            records=[json.loads(line) for line in metrics_path.read_text().splitlines() if json.loads(line)['update']<=saved['update']]
            if [x['update'] for x in records]!=list(range(1,saved['update']+1)):raise ValueError('cannot reconstruct accepted count from incomplete/duplicate metrics')
            count=sum(x['ppo']['accepted_epochs']>0 for x in records);algo.accepted_policy_updates=count
            saved['resume_metadata']['accepted_count_correction']=dict(original_saved_count=0,restored_count=count,source=str(metrics_path.resolve()),through_update=saved['update'],reason='lower PPO previously did not increment non-temporal accepted count; no weights/optimizer/physics change')
    torch.set_rng_state(saved['torch_rng'].cpu())
    if torch.cuda.is_available():torch.cuda.set_rng_state_all([x.cpu() for x in saved['cuda_rng']])
    np.random.set_state(saved['numpy_rng']);random.setstate(saved['python_rng'])
    states=jax.tree.map(jp.asarray,pickle.loads(saved['environment_pickle']))
    return states,saved['update'],saved['resume_metadata']


def aggregate_sampling(samples,n,steps):
    names=['ordinary','turn_return','post_return_small','reversal'];result={}
    result['target_issued_rates_min']=np.min([s['min'] for s in samples],axis=0).tolist()
    result['target_issued_rates_max']=np.max([s['max'] for s in samples],axis=0).tolist()
    result['labels_edge_dynamic_positive_turn_positive_small_fraction']=(np.sum([s['labels'] for s in samples],axis=0)/(n*steps)).tolist()
    for i,name in enumerate(names):
        a=np.sum([s['family_'+str(i)] for s in samples],axis=0).astype(float);c=a[0]
        result[name]=dict(samples=int(c),mean_reward=float(a[1]/c) if c else None,speed_rmse=float(np.sqrt(a[2]/c)) if c else None,steer_rmse=float(np.sqrt(a[3]/c)) if c else None,physical_failures=int(a[4]),ended_episodes=int(a[5]),working_fraction=float(a[6]/c) if c else None)
    count=float(sum(s['completed_episodes'] for s in samples));totals=np.sum([s['completed_episode_totals'] for s in samples],axis=0).astype(float)
    result['complete_episodes']=dict(count=int(count),mean_cost_per_tick=float(-totals[0]/totals[1]) if totals[1] else None,speed_rmse=float(np.sqrt(totals[2]/totals[1])) if totals[1] else None,steer_rmse=float(np.sqrt(totals[3]/totals[1])) if totals[1] else None,working_fraction=float(totals[4]/totals[1]) if totals[1] else None,mean_peak_roll=float(totals[5]/count) if count else None,scope='complete episodes across changing training policies')
    for i,name in enumerate(names):
        a=np.sum([s['completed_family_'+str(i)] for s in samples],axis=0).astype(float)
        result[name]['complete_episodes']=dict(count=int(a[0]),mean_cost_per_tick=float(-a[1]/a[2]) if a[2] else None,speed_rmse=float(np.sqrt(a[3]/a[2])) if a[2] else None,steer_rmse=float(np.sqrt(a[4]/a[2])) if a[2] else None)
    result['component_cap_fractions']={k[4:]:float(np.mean([s[k] for s in samples])) for k in samples[0] if k.startswith('cap_')}
    return result


def account_accepted_update(algo,metrics):
    # Generic PPO accounts only temporal upper policies; lower has no temporal spec.
    if not getattr(algo,'temporal_spec',None) and metrics['accepted_epochs']>0:
        algo.accepted_policy_updates+=1


def resume_preflight(path,out,cfg):
    """Reject stale/forked resumes before any run file is mutated."""
    import torch
    path=Path(path).resolve();out=Path(out).resolve()
    if not path.is_relative_to(out/'training'/'checkpoints'):raise ValueError('resume checkpoint must be owned by this run')
    d=torch.load(path,map_location='cpu',weights_only=False)
    if d['environment_continuation']!='full_rollout_boundary':raise ValueError('requires full rollout boundary')
    ignored={'updates','formal_transitions','status'}
    if {k:v for k,v in d['config'].items() if k not in ignored}!={k:v for k,v in cfg.items() if k not in ignored}:raise ValueError('resume config identity mismatch')
    if not d['update']<cfg['updates']<=cfg.get('max_total_updates',cfg['updates']):raise ValueError('target must exceed saved update within declared cap')
    metric_path=out/'training'/'metrics.jsonl'
    updates=[json.loads(x)['update'] for x in metric_path.read_text().splitlines()] if metric_path.exists() else []
    if updates and (max(updates)!=d['update'] or len(updates)!=len(set(updates))):raise ValueError('resume requires latest completed boundary without duplicate metrics')
    newer=[int(p.stem.split('_')[1]) for p in (out/'training'/'checkpoints').glob('update_*.pt')]
    if newer and max(newer)>d['update']:raise ValueError('resume requires latest checkpoint; later artifacts protected')
    return d['update']
