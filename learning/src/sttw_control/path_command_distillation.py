"""Frozen endpoint supervision for a single alpha-conditioned upper Actor.

Teacher data are complete pre-action Actor351 observations. No offline data are
used as PPO rollouts. The student passes through the original execution filter.
"""
from pathlib import Path
import copy,json,pickle,time
import numpy as np
import jax
import jax.numpy as j
from .preference_best import atomic_json
from .path_command_training import PathBatch,make_batched_advance as batched_advance
from .path_command_statistics import merge_statistics,reduce_statistics,reset_statistics
from .path_command_policy import make_path_actor,make_torch_path_network,export_torch_actor




class EndpointBatch(PathBatch):
    def reset(self,eid,episode,absolute_update=1):
        ep=super().reset(eid,episode,absolute_update)
        return ep.replace(state=ep.state.replace(alpha=((eid//2)%2).astype(j.float32)))


def load_teachers(folder):
    folder=Path(folder);records=[];params=[]
    for a in (0,1):
        record=json.loads((folder/f'alpha{a}/best_task_model.json').read_text())
        with Path(record['actor']).open('rb') as f:params.append(pickle.load(f))
        records.append({k:record[k] for k in ('alpha','source_update','actor','actor_sha256','checkpoint','lower_actor_sha256')})
    if records[0]['lower_actor_sha256']!=records[1]['lower_actor_sha256']:raise ValueError('teachers use different lower controllers')
    return records,jax.tree.map(lambda *v:j.stack(v),*params)


def mapped_torch(z,cfg):
    import torch
    c=cfg['residual_interface'];a=z.tanh()
    return torch.stack((a[:,0]*torch.where(a[:,0]>=0,c['speed_positive_scale_m_s'],c['speed_negative_scale_m_s']),a[:,1]*c['steer_scale_rad']),dim=-1)


def validation_env(env_ids):
    return (np.asarray(env_ids)//4)%4==0


def collect(env,snapshot,cfg,teacher_params,student_params,out,lanes,intervals,seed,status,writer,account):
    """One bounded scan50 collection; complete observations, phase and lane split."""
    out=Path(out);out.mkdir();batch=EndpointBatch(env,snapshot,cfg,seed,0.)
    ids=np.r_[np.arange(12),np.arange(102,126),np.arange(306,334)] if lanes==64 else np.r_[np.arange(8),np.arange(102,114),np.arange(306,318)]
    episodes=jax.jit(jax.vmap(batch.reset))(j.asarray(ids,j.int32),j.zeros(lanes,j.int32))
    env.set_log_template(jax.tree.map(lambda x:x[0],episodes.state),jax.tree.map(lambda x:x[0],episodes.route.path),episodes.route.speed[0])
    advance=batched_advance(batch);teacher=make_path_actor(cfg)
    scfg=copy.deepcopy(cfg);scfg['network']['architecture']='alpha_modulated_v1';student=make_path_actor(scfg)
    def chunk(episodes):
        def step(episodes,_):
            obs,_,fault=jax.vmap(batch.observation)(episodes)
            alpha=episodes.state.alpha.astype(j.int32)
            target=jax.vmap(lambda ob,a:teacher.apply(jax.tree.map(lambda v:v[a],teacher_params),ob))(obs,alpha)
            z=target if student_params is None else student.apply(student_params,obs)
            progress=episodes.state.projection.progress
            phase=j.where(progress>episodes.route.path.turn_end,2,j.where(j.abs(obs[:,340])>1e-4,1,0))
            nxt,_,reward,_,payload=advance(episodes,z,j.int32(1))
            return nxt,(obs,target,phase,episodes.env_id,episodes.episode,payload['active_ticks'],payload['engineering_fault_count']+j.sum(fault),j.mean(reward))
        return jax.lax.scan(step,episodes,None,length=50)
    status(stage='compiling_collection',completed_intervals=0,declared_intervals=intervals)
    before=time.monotonic();run=jax.jit(chunk).lower(episodes).compile();compile_seconds=time.monotonic()-before
    blocks=[]
    for start in range(0,intervals,50):
        episodes,data=run(episodes);data=jax.device_get(data)
        if np.any(data[6]):raise RuntimeError('teacher/student collection engineering fault')
        account(int(np.sum(data[5])));blocks.append(data[:5])
        writer.add_scalar('train/mean_step_reward',float(np.mean(data[7])),start+50);writer.flush()
        status(stage='collecting',completed_intervals=start+50,declared_intervals=intervals,compile_seconds=compile_seconds)
    obs,labels,phase,ids,episode=[np.concatenate([block[k] for block in blocks],axis=0) for k in range(5)]
    n=obs.shape[0]*obs.shape[1]
    data=dict(observation=obs.reshape(n,351),teacher_z=labels.reshape(n,2),phase=phase.reshape(n),env_id=ids.reshape(n),episode=episode.reshape(n),validation=validation_env(ids.reshape(n)))
    np.savez(out/'dataset.npz',**data)
    atomic_json(out/'receipt.json',dict(policy_steps=n,lower_ticks_max=n*4,observation_dim=351,supervision='mapped physical target before filter',split='fixed env IDs across teacher and student collection; no frame-level split',compile_seconds=compile_seconds,behavior='teacher' if student_params is None else 'student',endpoint_counts={str(a):int(np.sum(data['observation'][:,336]==a)) for a in (0,1)}))
    return data


def fit_student(data,teacher0_checkpoint,cfg,updates,seed,out,writer,status,actor=None):
    import torch
    torch.set_num_threads(2);torch.manual_seed(seed);rng=np.random.default_rng(seed)
    if actor is None:
        actor=make_torch_path_network(351,2,cfg).to('cuda')
        parent=torch.load(teacher0_checkpoint,map_location='cuda',weights_only=False)['policy']
        state=actor.state_dict()
        for key in state:
            if 'actor.'+key in parent:state[key]=parent['actor.'+key].clone()
        state['0.weight'][:,336]=0
        actor.load_state_dict(state)
    x=torch.as_tensor(data['observation'],device='cuda');z=torch.as_tensor(data['teacher_z'],device='cuda')
    with torch.no_grad():target=mapped_torch(z,cfg)/x.new_tensor([1.,.2])
    train=~data['validation'];phase=data['phase'];alpha=data['observation'][:,336]
    buckets=[np.flatnonzero(train&(alpha==a)&(phase==p)) for a in (0,1) for p in (0,1,2)]
    if any(not len(b) for b in buckets):raise RuntimeError('distillation needs both endpoints in straight/turn/exit')
    validation=[np.flatnonzero((~train)&(alpha==a)&(phase==p)) for a in (0,1) for p in (0,1,2)]
    if any(not len(b) for b in validation):raise RuntimeError('held-out endpoint/phase coverage missing')
    validation=[torch.as_tensor(b,device='cuda') for b in validation]
    optimizer=torch.optim.Adam(actor.parameters(),lr=1e-4)
    best=float('inf');best_state=None
    out=Path(out);out.mkdir(exist_ok=True)
    for update in range(1,updates+1):
        ids=np.concatenate([rng.choice(bucket,86,replace=True) for bucket in buckets]);index=torch.as_tensor(ids,device='cuda')
        pred=mapped_torch(actor(x[index]),cfg)/x.new_tensor([1.,.2]);loss=(pred-target[index]).square().mean()
        optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(actor.parameters(),1.);optimizer.step()
        if update==1 or update%100==0:
            with torch.no_grad():
                parts=[((mapped_torch(actor(x[ix]),cfg)/x.new_tensor([1.,.2])-target[ix])**2).mean() for ix in validation]
                v=float(torch.stack(parts).mean())
            if not np.isfinite(v):raise RuntimeError('nonfinite distillation loss')
            if v<best:best=v;best_state={k:t.detach().clone() for k,t in actor.state_dict().items()}
            writer.add_scalar('distill/train_mse',float(loss.detach()),update);writer.add_scalar('distill/validation_mse',v,update);writer.flush()
            status(stage='distilling',completed_bc_updates=update,declared_bc_updates=updates,validation_mse=v,validation_group_mse=[float(x) for x in parts])
    actor.load_state_dict(best_state)
    torch.save({'actor':actor.state_dict(),'validation_mse':best,'updates':updates},out/'student.pt')
    with (out/'student_actor.pkl').open('wb') as f:pickle.dump(jax.device_get(export_torch_actor(actor,cfg)),f)
    return actor


def inheritance_gate(panels,teacher_root,cfg):
    """Same fixed cases/initial state. Compare each student's own endpoint teacher."""
    from .path_command_selection import summarize_task
    from .path_command_selection import cases
    expected={row[0] for row in cases(cfg)}
    if set(panels)!={'alpha_0','alpha_1'} or any(set(v)!=expected for v in panels.values()):raise ValueError('complete endpoint panel required')
    def measure(d,path):
        dt=cfg['timing']['lower_dt_s'];h=cfg['training_future_phase_C']['episode_s'];t=np.asarray(d['time'],float)+dt
        valid=t<=h+1e-5;tail=valid&(t>h-1+1e-5)
        ev=np.asarray(d['actual_forward_speed'],float)-d['v_user'];ey=d['path_cross_track'];eh=d['path_heading_error'];progress=d['path_progress'];roll=d['peak_roll']
        curved=np.flatnonzero(abs(path['curvature'])>1e-8)
        turn=valid&((progress>=float(path['s'][curved[0]]))&(progress<=float(path['turn_end']))) if len(curved) else valid
        if not turn.any():turn=valid
        physical=bool(np.any(np.asarray(d['physical_failure'])[valid]));domain=bool(np.any(np.asarray(d['domain_exit'])[valid]))
        hold=bool(valid.sum()==round(h/dt) and tail.sum()==round(1/dt) and np.all(abs(ev[tail])<=.1) and np.all(abs(ey[tail])<=.1) and np.all(abs(eh[tail])<=.05) and not (physical or domain))
        goal=bool(not len(curved) or np.any(valid&(d['goal_section_signed_distance']>=0)&(progress>=d['goal_progress'])))
        def window(mask):return dict(path_rmse_m=float(np.sqrt(np.mean(ey[mask]**2))),speed_rmse_m_s=float(np.sqrt(np.mean(ev[mask]**2))),peak_roll_rad=float(max(roll[mask])))
        return dict(windows=dict(whole=window(valid),spatial_turn=window(turn)),physical_failure=physical,tracking_domain_failure=domain,goal_crossed=goal,final_hold=hold,source_observed_seconds=len(t)*dt,comparison_seconds=h)
    details=[]
    for alpha in (0,1):
        for case,d in panels[f'alpha_{alpha}'].items():
            teacher=Path(teacher_root)/f'alpha{alpha}/final_best'
            td=dict(np.load(teacher/(case+'.npz')));path=dict(np.load(teacher/(case+'_path.npz')))
            m=dict(alpha0=measure(td,path),alpha1=measure(d,path))
            # Native metrics report exact geometric turn windows; straight uses whole.
            t=m['alpha0'];s=m['alpha1'];window='spatial_turn' if 'straight' not in case else 'whole'
            primary='path_rmse_m' if alpha==0 else 'speed_rmse_m_s'
            sm=s['windows'][window][primary];tm=t['windows'][window][primary]
            passed=bool(not s['physical_failure'] and not s['tracking_domain_failure'] and s['goal_crossed'] and s['final_hold'] and s['windows']['whole']['peak_roll_rad']<=max(.302,t['windows']['whole']['peak_roll_rad']+.01) and sm<=1.25*tm+.02)
            details.append(dict(alpha=alpha,case=case,passed=passed,primary=primary,student=sm,teacher=tm,metrics=s))
    return dict(passed=all(x['passed'] for x in details),cases=details,threshold='no failure; goal+hold; peak <= max(.302,teacher+.01); own endpoint primary <=1.25*teacher+.02')
