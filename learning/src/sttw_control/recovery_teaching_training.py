"""Fixed-budget supervised coordination teaching, without PPO or physics."""
from pathlib import Path
import json,pickle,os,time
import numpy as np
from .preference_best import atomic_json,digest


def train(declaration, output):
 import jax,jax.numpy as j,optax
 from torch.utils.tensorboard import SummaryWriter
 from .path_command_policy import make_path_actor,validate_actor_parameters
 d=json.loads(Path(declaration).read_text());out=Path(output).resolve();out.mkdir(parents=True,exist_ok=True)
 if (out/'manifest.json').exists():raise FileExistsError('immutable teaching run exists')
 status=dict(phase='running',state='running',pid=os.getpid(),completed_updates=0,declared_updates=d['updates'],training_steps=0,physics_ticks=0)
 def update(**kw):
  status.update(kw,checked_unix=time.time());atomic_json(out/'status.json',status)
 update(stage='loading');writer=None
 try:
  if d['role']!='TRAIN' or d['updates']!=400:raise ValueError('unexpected teaching contract')
  dataset=Path(d['dataset']);assert digest(dataset/'manifest.json')==d['dataset_manifest_sha256']
  manifest=json.loads((dataset/'manifest.json').read_text());names=('intervention','recovery_tail','control','nominal');groups=[]
  for name in names:
   p=dataset/(name+'.npz')
   if digest(p)!=manifest['groups'][name]['sha256']:raise ValueError('dataset changed')
   with np.load(p) as z:obs=z['observations'];targets=z['targets']
   if obs.shape!=(len(targets),351) or targets.shape[1]!=2 or not np.isfinite(obs).all() or not np.isfinite(targets).all():raise ValueError('invalid teaching data')
   groups.append((obs,targets))
  parent=d['parent'];assert digest(parent['actor'])==parent['actor_sha256'];assert digest(parent['checkpoint'])==parent['checkpoint_sha256']
  with Path(parent['actor']).open('rb') as f:params=jax.tree.map(j.asarray,pickle.load(f))
  validate_actor_parameters(params,d['config']);actor=make_path_actor(d['config'])
  optimizer=optax.chain(optax.clip_by_global_norm(d['gradient_clip']),optax.adam(d['learning_rate']));state=optimizer.init(params)
  def loss(p,x,y):
   error=(actor.apply(p,x)-y)**2;channels=j.mean(error,axis=1);return j.mean(channels),channels
  @jax.jit
  def step(p,s,x,y):
   (value,channels),grad=jax.value_and_grad(loss,has_aux=True)(p,x,y)
   updates,s=optimizer.update(grad,s,p);return optax.apply_updates(p,updates),s,value,channels,optax.global_norm(grad)
  predict=jax.jit(lambda p,x:actor.apply(p,x))
  def audit(p):
   report={}
   for name,(x,y) in zip(names,groups):
    sums=np.zeros(2)
    for i in range(0,len(x),256):
     error=np.asarray(predict(p,j.asarray(x[i:i+256])))-y[i:i+256];sums+=(error**2).sum(0)
    report[name]=(sums/len(x)).tolist()
   return report
  atomic_json(out/'manifest.json',dict(declaration=d,semantics='actor-only supervised teaching; fresh optimizer; no critic/log_std update; not PPO-resumable',dataset=manifest))
  writer=SummaryWriter(str(out/'tensorboard'/'teaching'));initial=audit(params);atomic_json(out/'initial_fit.json',initial)
  rng=np.random.default_rng(d['seed']);metrics=out/'metrics.jsonl';batch=d['batch_per_group'];started=time.monotonic()
  for it in range(1,d['updates']+1):
   samples=[rng.integers(len(x),size=batch) for x,y in groups]
   x=j.asarray(np.stack([g[0][ix] for g,ix in zip(groups,samples)]));y=j.asarray(np.stack([g[1][ix] for g,ix in zip(groups,samples)]))
   params,state,value,channels,norm=step(params,state,x,y);value=float(value);channels=np.asarray(channels);norm=float(norm)
   if not np.isfinite(value) or not np.isfinite(channels).all() or not np.isfinite(norm):raise RuntimeError('nonfinite teaching optimization')
   row=dict(update=it,loss=value,gradient_norm=norm,group_channel_mse=channels.tolist(),elapsed_s=time.monotonic()-started)
   with metrics.open('a') as f:f.write(json.dumps(row)+'\n')
   writer.add_scalar('fit/loss',value,it);writer.add_scalar('fit/gradient_norm',norm,it)
   for k,name in enumerate(names):
    for c,label in enumerate(('speed','steer')):writer.add_scalar(f'fit/{name}_{label}_mse',float(channels[k,c]),it)
   if it==1 or it%20==0:writer.flush();update(stage='teaching',completed_updates=it,loss=value)
  final=audit(params);atomic_json(out/'final_fit.json',final)
  for leaf in jax.tree.leaves(params):
   if not np.isfinite(np.asarray(leaf)).all():raise RuntimeError('nonfinite final Actor')
  with (out/'actor.pkl').open('wb') as f:pickle.dump(jax.device_get(params),f)
  with (out/'training_state.pkl').open('wb') as f:pickle.dump(dict(actor=jax.device_get(params),optimizer=jax.device_get(state),rng_state=rng.bit_generator.state,completed_updates=d['updates'],kind='supervised_actor_only'),f)
  atomic_json(out/'candidate_record.json',dict(actor=str(out/'actor.pkl'),actor_sha256=digest(out/'actor.pkl'),checkpoint=str(out/'training_state.pkl'),checkpoint_sha256=digest(out/'training_state.pkl'),source_update=d['updates'],kind='supervised_actor_only_not_PPO_checkpoint'))
  writer.flush();update(phase='completed',state='completed',stage='candidate_ready',completed_updates=d['updates'],elapsed_s=time.monotonic()-started)
 except BaseException as exc:update(phase='error',state='error',error=f'{type(exc).__name__}: {exc}');raise
 finally:
  if writer is not None:writer.close()
