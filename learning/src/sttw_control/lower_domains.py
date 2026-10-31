"""Model-bound initial states for finite-domain lower-controller training."""
import hashlib,json,pickle,time,os,shutil,subprocess
from pathlib import Path
from dataclasses import asdict
import numpy as np
import jax
import jax.numpy as j
from .teleop_env import TeleopEnv
from .actuator import initial_actuator
from .runtime import configure_compilation_cache


class DomainModels:
    """Batch only changed numeric fields; keep common geometry/sensors shared."""
    def __init__(self,models):
        self.shared,self.structure=jax.tree.flatten(models[0])
        leaves=[jax.tree.leaves(m) for m in models]
        self.variables=[]
        for i,reference in enumerate(self.shared):
            if any(not np.array_equal(np.asarray(reference),np.asarray(row[i])) for row in leaves[1:]):
                self.variables.append((i,j.stack([row[i] for row in leaves])))

    def select(self,index):
        leaves=list(self.shared)
        for i,values in self.variables:leaves[i]=values[index]
        return self.structure.unflatten(leaves)


def load_domain_models(manifest,roles):
    path=Path(manifest);data=json.loads(path.read_text())
    if hashlib.sha256(path.read_bytes()).hexdigest()!=json.loads((path.parent/'seal.json').read_text())['manifest_sha256']:
        raise ValueError('physical-domain manifest seal differs')
    records=[r for r in data['domains'] if r['data_role'] in roles]
    if not records or len({r['domain_id'] for r in records})!=len(records):raise ValueError('nonempty unique domain selection required')
    models=[];reference=None
    for row in records:
        if hashlib.sha256(Path(row['output']).read_bytes()).hexdigest()!=row['model']['source_xml_sha256']:
            raise ValueError('physical-domain XML changed')
        env=TeleopEnv(model_path=row['output'])
        if env.bundle.identity!=row['model']:raise ValueError('compiled domain identity differs')
        if reference is None:reference=env
        if asdict(env.cc)!=asdict(reference.cc) or asdict(env.ac)!=asdict(reference.ac):raise ValueError('domain changed controller or actuator contract')
        if jax.tree.structure(env.helpers.mjx_model)!=jax.tree.structure(reference.helpers.mjx_model):raise ValueError('domain topology differs')
        models.append(env.helpers.mjx_model)
    return reference,DomainModels(models),records


def prepare_domain_states(config,output):
    cfg=json.loads(Path(config).read_text());out=Path(output)
    count=cfg['states_per_domain'];ticks=cfg['preparation_ticks']
    if not isinstance(count,int) or count<2 or ticks!=700 or cfg['roles'] not in (['REFERENCE','TRAIN'],['TEST']):
        raise ValueError('separate TRAIN/reference or TEST preparation with 700 ticks required')
    speed=np.asarray(cfg['speed_m_s'],float);force=np.asarray(cfg['terminal_world_y_force_N'],float)
    if (speed.shape!=(2,) or force.shape!=(2,) or not np.isfinite(np.r_[speed,force]).all()
        or not 1.5<=speed[0]<speed[1]<=3.5 or not -12<=force[0]<force[1]<=12):
        raise ValueError('declared bounded speed and terminal force ranges required')
    out.mkdir(parents=True,exist_ok=False)
    snapshot=out/'source';snapshot.mkdir()
    for name in ['lower_domains.py','teleop_env.py','env.py','model.py','controller.py','actuator.py','closed_loop_kernel.py']:
        shutil.copyfile(Path(__file__).parent/name,snapshot/name)
    shutil.copyfile(config,out/'config.json')
    (out/'code_receipt.json').write_text(json.dumps(dict(git_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),core_source_snapshot=str(snapshot),batching='lax.map domains; vmap lanes with shared model'),indent=2)+'\n')
    configure_compilation_cache(Path(cfg['shared_jax_cache']))
    env,models,records=load_domain_models(cfg['domain_manifest'],cfg['roles'])
    n=len(records)*count
    if cfg['maximum_lower_ticks']!=n*ticks:raise ValueError('domain preparation budget differs')
    rng=np.random.default_rng(cfg['seed']);conditions=[];pulses=[]
    for record in records:
        velocities=speed[0]+(rng.permutation(count)+rng.random(count))/count*(speed[1]-speed[0])
        roll=rng.uniform(-.06,.06,count);f=rng.uniform(*force,count);f[:count//2]=0.
        duration=rng.uniform(.15,.3,count)
        conditions.extend(zip(velocities,roll))
        pulses.extend(np.column_stack([3.5-duration,np.full(count,3.5),np.zeros(count),f,np.zeros(count)]).tolist())
    domain_index=j.repeat(j.arange(len(records),dtype=j.int32),count)
    plan=dict(config=cfg,domains=records,conditions=np.asarray(conditions).tolist(),pulses=pulses,domain_index=np.asarray(domain_index).tolist(),controller=asdict(env.cc),actuator=asdict(env.ac),model=env.bundle.identity,physical_steps=0,optimizer_updates=0)
    (out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    status=dict(phase='initializing',pid=os.getpid(),actual_lower_ticks=0,maximum_lower_ticks=n*ticks,optimizer_updates=0)
    def save(): (out/'status.json').write_text(json.dumps(status,indent=2)+'\n')
    save();start=time.monotonic();base=env.initial()
    def initial(condition,di):
        model=models.select(di);v,angle=condition
        q=base.data.qpos;w,x,y,z=q[3:7];ca,sa=j.cos(-angle/2),j.sin(-angle/2)
        q=q.at[3:7].set(j.array([ca*w-sa*x,ca*x+sa*w,ca*y-sa*z,ca*z+sa*y]))
        vel=base.data.qvel.at[0].set(v).at[env.bundle.rear_dof].set(-v/.1).at[env.bundle.front_dof].set(-v/.1)
        data=base.data.replace(qpos=q,qvel=vel,ctrl=base.data.ctrl.at[1].set(-v/.1))
        data=env.helpers._mjx.forward(model,data);pose=env.helpers.pose(data);raw=j.array([v,0.])
        return base.replace(data=data,raw=raw,actuator=initial_actuator(env.ac,v/.1),reference_pose=pose,yaw_wrapped=pose[2],yaw_unwrapped=pose[2],governor=base.governor.replace(current_reference=raw,last_goal=raw))
    def initial_group(args):
        cs,di=args
        return jax.vmap(lambda c:initial(c,di))(cs)
    bank=jax.jit(lambda cs:jax.lax.map(initial_group,(cs,j.arange(len(records)))))(j.asarray(conditions).reshape(len(records),count,2))
    pulse=j.asarray(pulses).reshape(len(records),count,5)
    def one(state,di,p):
        model=models.select(di)
        return env._step(state,state.raw,True,force_pulse=p,physics_model=model)
    def group(args):
        states,di,ps=args
        # Domain selection is outside vmap: physics parameters are shared by lanes.
        return jax.vmap(lambda state,p:one(state,di,p))(states,ps)
    chunk=jax.jit(lambda b:jax.lax.scan(lambda b,_:jax.lax.map(group,(b,j.arange(len(records)),pulse)),b,None,length=50))
    traces=[]
    for block in range(ticks//50):
        status.update(phase='inflight',allocated_lower_ticks=(block+1)*50*n);save()
        bank,logs=chunk(bank);jax.block_until_ready(bank)
        traces.append({k:np.asarray(logs[k]).reshape((50,n)+logs[k].shape[3:]) for k in ['actual_xy','phi','phi_dot','actual_forward_speed','actual_delta','yaw_unwrapped','applied_force_impulse']})
        status.update(phase='running',actual_lower_ticks=(block+1)*50*n,elapsed_s=time.monotonic()-start);save()
    trace={k:np.concatenate([r[k] for r in traces]) for k in traces[0]}
    bank=jax.tree.map(lambda x:x.reshape((n,)+x.shape[2:]),bank)
    np.savez_compressed(out/'trace.npz',**trace)
    with (out/'prepared_bank.pkl').open('wb') as f:pickle.dump(jax.device_get(bank),f)
    yaw=(trace['yaw_unwrapped'][-1]-trace['yaw_unwrapped'][-2])/.005
    np.savez(out/'state_domains.npz',domain_index=np.asarray(domain_index),initial_yaw_rate=yaw)
    passed=(~np.asarray(bank.failed))&np.isfinite(trace['phi'][-1])&(abs(trace['phi'][-1])<=.35)&(abs(trace['phi_dot'][-1])<=1.5)&(abs(trace['actual_delta'][-1])<=.6)
    metrics=dict(admitted=passed.tolist(),failed=np.asarray(bank.failed).tolist(),speed=trace['actual_forward_speed'][-1].tolist(),roll=trace['phi'][-1].tolist(),roll_rate=trace['phi_dot'][-1].tolist(),steer=trace['actual_delta'][-1].tolist(),initial_yaw_rate=yaw.tolist(),admission='noncontact finite, abs roll<=.35rad, rollrate<=1.5rad/s, steer<=.6rad; not stable-bank qualification')
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    receipt=dict(plan,prepared_bank=str(out/'prepared_bank.pkl'),state_domains=str(out/'state_domains.npz'),status='admitted' if passed.all() else 'not_admitted')
    receipt['prepared_bank_sha256']=hashlib.sha256((out/'prepared_bank.pkl').read_bytes()).hexdigest()
    receipt['state_domains_sha256']=hashlib.sha256((out/'state_domains.npz').read_bytes()).hexdigest()
    (out/'initial_source.json').write_text(json.dumps(receipt,indent=2)+'\n')
    status.update(phase='completed' if passed.all() else 'not_admitted',admitted=int(passed.sum()),states=n,elapsed_s=time.monotonic()-start);save()
    print(json.dumps(status))


def evaluate_domain_pair(config, output):
    """Frozen local-command TEST, paired complete states and command seeds."""
    from .lower_command import LowerCommandEnv
    from .network import ResidualActor
    from .direct_command_training import flatten_logs
    cfg=json.loads(Path(config).read_text());out=Path(output)
    source=json.loads(Path(cfg['initial_source']).read_text())
    spec=json.loads(Path(cfg['controller_config']).read_text())
    if source['config']['roles']!=['TEST'] or source['status']!='admitted':
        raise ValueError('admitted TEST bank required; never optimizer sampling')
    for key in ('prepared_bank','state_domains'):
        if hashlib.sha256(Path(source[key]).read_bytes()).hexdigest()!=source[key+'_sha256']:
            raise ValueError('TEST state identity changed: '+key)
    spec['command_seed']=cfg['command_seed']
    spec['physical_model_xml']=source['domains'][0]['output']
    configure_compilation_cache(spec['shared_jax_cache'])
    _,models,records=load_domain_models(source['config']['domain_manifest'],['TEST'])
    if records!=source['domains']:raise ValueError('TEST models changed')
    with Path(source['prepared_bank']).open('rb') as f:bank=jax.tree.map(j.asarray,pickle.load(f))
    mapping=np.load(source['state_domains']);ids=j.asarray(mapping['domain_index']);yaw=j.asarray(mapping['initial_yaw_rate'])
    n=len(ids);ticks=round(spec['episode_seconds']/spec['dt'])
    if cfg['maximum_lower_ticks']!=2*n*ticks or ticks!=2000 or len(cfg['actors'])!=2:
        raise ValueError('declared two-policy 10-second budget required')
    params=[]
    for actor in cfg['actors']:
        path=Path(actor['actor'])
        if hashlib.sha256(path.read_bytes()).hexdigest()!=actor['actor_sha256']:raise ValueError('actor changed')
        with path.open('rb') as f:params.append(jax.tree.map(j.asarray,pickle.load(f)))
    out.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(config,out/'config.json');shutil.copyfile(__file__,out/'lower_domains.py')
    (out/'initial_source.json').write_text(json.dumps(source,indent=2)+'\n')
    env=LowerCommandEnv(spec);net=ResidualActor(tuple(spec['hidden_sizes']),activation=spec['activation'])
    initial=jax.jit(jax.vmap(lambda snap,i,di,y:env.reset(snap,i,j.int32(0),di,y)))(bank,j.arange(n),ids,yaw)
    example=jax.tree.map(lambda x:x[0],initial)
    shape=jax.eval_shape(lambda s:env.step(s,j.zeros(2),models.select(s.domain_index)),example)[3]
    zero=jax.tree.map(lambda x:j.zeros(x.shape,x.dtype),shape)
    def one(state,par):
        def active(s):
            obs,_=env.observation(s)
            ns,_,_,log=env.step(s,net.apply(par,obs,return_logits=True),models.select(s.domain_index))
            return ns,log
        return jax.lax.cond(state.physical.failed,lambda s:(s,zero),active,state)
    @jax.jit
    def chunk(states,par):
        return jax.lax.scan(lambda states,_:jax.vmap(one,in_axes=(0,None))(states,par),states,None,length=50)
    status=dict(phase='running',pid=os.getpid(),maximum_lower_ticks=cfg['maximum_lower_ticks'],allocated_lower_ticks=0,optimizer_updates=0)
    start=time.monotonic();summaries={}
    for actor,par in zip(cfg['actors'],params):
        states=initial;chunks=[];dest=out/actor['name'];dest.mkdir()
        for _ in range(ticks//50):
            status.update(actor=actor['name'],allocated_lower_ticks=status['allocated_lower_ticks']+50*n,elapsed_s=time.monotonic()-start)
            (out/'status.json').write_text(json.dumps(status,indent=2)+'\n')
            states,logs=chunk(states,par);jax.block_until_ready(states);chunks.append(jax.device_get(logs))
        logs=jax.tree.map(lambda *xs:np.concatenate(xs),*chunks);rows=[]
        for i in range(n):
            d=flatten_logs(jax.tree.map(lambda x:x[:,i],logs));used=int(states.tick[i]);d={k:v[:used] for k,v in d.items()}
            np.savez_compressed(dest/f'case_{i:02d}.npz',**d)
            ev=d['actual_forward_speed']-d['limited_command'][:,0];ed=d['actual_delta']-d['limited_command'][:,1]
            rows.append(dict(case=i,domain=int(ids[i]),steps=used,failed=bool(states.physical.failed[i]),speed_rmse=float(np.sqrt(np.mean(ev**2))),steer_rmse=float(np.sqrt(np.mean(ed**2))),reward=float(d['reward'].sum())))
        summaries[actor['name']]=rows
    status.update(phase='completed',elapsed_s=time.monotonic()-start,actual_lower_ticks=sum(r['steps'] for rows in summaries.values() for r in rows))
    (out/'results.json').write_text(json.dumps(summaries,indent=2)+'\n')
    (out/'status.json').write_text(json.dumps(status,indent=2)+'\n')
    print(json.dumps(status))
