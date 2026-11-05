"""Export authentic H75 TRAIN states/routes and native forward oracles; no steps."""
import argparse,dataclasses,hashlib,json,os,pickle
from pathlib import Path
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
os.environ.setdefault('JAX_DEFAULT_MATMUL_PRECISION','highest')
import numpy as np
import jax
import jax.numpy as jnp
from sttw_control.model import load_model
from sttw_control.isaac_portable_state import export_physical_states,export_original_policy
from sttw_control.path_command_training import load_environment
from sttw_control.path_command_policy import PathCommandActor,architecture_identity
from sttw_control.geometric_path import at,project
p=argparse.ArgumentParser();p.add_argument('--bank',type=Path,required=True);p.add_argument('--bundle',type=Path,required=True);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--checkpoint',type=Path);a=p.parse_args()
manifest=json.loads((a.bank/'manifest.json').read_text());receipt=json.loads((a.bank/'receipt.json').read_text());data=(a.bank/'bank.pkl').read_bytes();digest=hashlib.sha256(data).hexdigest()
if digest!=receipt['bank_sha256']:raise ValueError('original H75 bank identity changed')
bank=pickle.loads(data);ids=np.flatnonzero(bank['valid']);states=jax.tree.map(lambda x:jnp.asarray(x[ids]),bank['states']);routes=jax.tree.map(lambda x:jnp.asarray(x[ids]),bank['routes']);a.output.mkdir(parents=True,exist_ok=False)
config=json.loads((a.bundle/'config.json').read_text());identity=json.loads((a.bundle/'identity.json').read_text());source=json.loads(a.source.read_text());env,_=load_environment(config,source)
obs,critic,fault=jax.jit(jax.vmap(lambda s,path,speed:env.observation(s,path,speed,10.)))(states,routes.path,routes.speed)
measurement,pose,speed=jax.jit(jax.vmap(lambda s:env.physics.observe(s.physical.data)))(states)
params=jax.tree.map(jnp.asarray,pickle.load(open(identity['source_actor']['source'],'rb')));actor=PathCommandActor(architecture=architecture_identity(config));means=jax.jit(jax.vmap(lambda x:actor.apply(params,x)))(obs)
physical=export_physical_states(states.physical,load_model(),range(len(ids)))
portable={**physical,'source_bank_index':ids,'source_route_id':np.asarray(bank['route_ids'])[ids],'source_onset_interval':np.asarray(bank['onset_intervals'])[ids],'native_observation':np.asarray(obs),'native_critic_observation':np.asarray(critic),'native_fault':np.asarray(fault),'native_mean':np.asarray(means),'native_measurement':np.asarray(measurement),'native_pose':np.asarray(pose),'native_forward_speed':np.asarray(speed)}
def flatten(value,prefix):
    if dataclasses.is_dataclass(value):
        for field in dataclasses.fields(value):
            if prefix=='closedloop.physical' and field.name=='data':continue
            flatten(getattr(value,field.name),prefix+'.'+field.name)
    else:portable[prefix]=np.asarray(value)
flatten(states,'closedloop')
np.savez_compressed(a.output/'states.npz',**portable)
paths=jax.device_get(routes.path);ss=np.asarray(paths.s)
if not np.all(ss==ss[0]):raise ValueError('shared source route grid required')
np.savez_compressed(a.output/'routes.npz',s=ss[0],**{key:np.asarray(getattr(paths,key)) for key in ['xy','heading','curvature','turn_end','goal']},**{key:np.asarray(getattr(routes,key)) for key in ['speed','family','radius','angle','lateral_offset','heading_offset','sample_group']})
progress=states.projection.progress;queries=progress+.123
points,headings,curvatures=jax.jit(jax.vmap(at))(routes.path,queries)
projection=jax.jit(jax.vmap(lambda path,xy,prev:project(path,xy,prev,jnp.float32(.025),config)))(routes.path,pose[:,:2],progress)
np.savez_compressed(a.output/'route_oracle.npz',query=np.asarray(queries),points=np.asarray(points),heading=np.asarray(headings),curvature=np.asarray(curvatures),pose_xy=np.asarray(pose[:,:2]),previous=np.asarray(progress),**{key:np.asarray(getattr(projection,key)) for key in ['progress','unclamped','point','cross_track']},projection_heading=np.asarray(projection.heading))
meta=dict(schema='sttw_h75_original_training_portable_v1',source_bank=str(a.bank.resolve()),source_bank_sha256=digest,source_parent=manifest['parent'],selected_h75_actor=identity['source_actor'],config=str((a.bundle/'config.json').resolve()),source_bank_manifest=manifest['declaration'],count=len(ids),physics_steps=0,optimizer_updates=0,source_alpha_counts={str(alpha):int(np.sum(np.asarray(states.alpha)==alpha)) for alpha in [0,.25,.5,.75,1]},clock_lower_tick_range=[int(np.min(states.tick)),int(np.max(states.tick))],speed_range_m_s=[float(np.min(routes.speed)),float(np.max(routes.speed))],family_counts={str(i):int(np.sum(np.asarray(routes.family)==i)) for i in [0,1,2]},role='portable original TRAIN data and forward oracle; not new on-policy physics samples',physical_limitations='MuJoCo solver/contact caches not transferred. Bank came from historical friction10 MuJoCo; current selected Lab uses friction0.8. Kinematic conversion and forward observation only; no closed-loop equivalence claim.',initialization='saved original prefix states; not cold reset, teacher collection remains u325, selected inference actor u77')
if a.checkpoint:
    export_original_policy(a.checkpoint,a.output/'original_policy.pt')
    meta['original_policy']='original_policy.pt'
(a.output/'identity.json').write_text(json.dumps(meta,indent=2));print(json.dumps({k:meta[k] for k in ['count','source_alpha_counts','clock_lower_tick_range','speed_range_m_s','family_counts','physics_steps','optimizer_updates']},indent=2))
