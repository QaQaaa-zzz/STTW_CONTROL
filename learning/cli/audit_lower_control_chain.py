from pathlib import Path
import os,sys,argparse
os.environ.setdefault("JAX_PLATFORMS","cpu")
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import json,pickle
import numpy as np
import jax,jax.numpy as jp
from sttw_control.teleop_env import TeleopEnv
from sttw_control.controller import controller_step
from sttw_control.lower_interface_diagnostics import ecbc_contributions,projection_details
from sttw_control.frozen_lower_controller import FrozenLowerState,path_features
from sttw_control.teleop_reference import integrate
from sttw_control.tracking_reward import TrackingConfig,initial_return,transition
p=argparse.ArgumentParser(description="Offline ECBC/ESO/return reconstruction; no physics")
p.add_argument('--parent-run',required=True);p.add_argument('--initial-state',required=True);p.add_argument('--output',required=True);args=p.parse_args()
ROOT=Path(args.parent_run);OUT=Path(args.output);OUT.mkdir(parents=True,exist_ok=True)
with open(args.initial_state,'rb') as f:state=jax.tree.map(jp.asarray,pickle.load(f))
env=TeleopEnv(backend='cpu');m0=env.helpers.measure(state.physical.data);cc=env.cc
registry=json.loads((Path(__file__).resolve().parents[1]/'configs/frozen_lower_registry.json').read_text())
source=json.loads(Path(registry['models']['STTW_R196_ALPHA1']['source_declaration']).read_text())['task'];tr=TrackingConfig(**source['tracking'])
checks={}
for method in ('B0','alpha0'):
 d=dict(np.load(ROOT/f'evaluation100/fast_turn/{method}.npz'));n=min(len(d['time']),2400)
 d={k:v[:n] for k,v in d.items() if np.ndim(v)>0 and len(v)>=n}
 # Each previous post-state is exactly the next control pre-state, not lag-shifted scoring.
 measurements=np.column_stack([d['wheel_speed_proxy'],d['actual_delta'],d['actual_delta_rate'],d['phi'],d['phi_dot'],d['governed'][:,1]])
 initial=np.array([m0[5]*.1,m0[2],m0[3],m0[0],m0[1],0],np.float32)
 pre=np.vstack([initial,measurements[:-1]]);pre[:,-1]=d['governed'][:,1]
 def step(cs,x):
  row,i=x;enable=(state.physical.physical_tick+i)*.005>3.
  new,out=controller_step(cs,row,enable,cc);terms=ecbc_contributions(row,new.gains,out,enable)
  return new,dict(ecbc_terms=terms,ecbc_output=out.steer_rate,gains=new.gains,eso_pre=cs.eso,disturbance=cs.disturbance,reference_roll=out.reference_roll,equilibrium_shift=out.equilibrium_shift)
 _,diag=jax.jit(lambda cs:jax.lax.scan(step,cs,(jp.asarray(pre),jp.arange(n))))(state.physical.controller)
 diag=jax.device_get(diag);error=np.max(np.abs(diag['ecbc_output']-d['u_nom'][:,0]))
 checks[method]={'max_ecbc_reconstruction_abs':float(error),'new_physics_steps':0,'alignment':'controller pre-state from previous stored post-state; scoring unchanged','return_post_source':'next stored row pre path; source extends beyond the12s audit' ,'full_lower_input_available':False,'actuator_force_available':False}
 # Reconstruct return from saved path features at t+dt, with no projection guess.
 # In geometric mode timed channels do not affect final/return decisions (passed explicitly as0).
 all_data=dict(np.load(ROOT/f'evaluation100/fast_turn/{method}.npz'));postpath=all_data['lower_path_features'][1:n+1]
 assert len(postpath)==n
 def retstep(ret,x):
  i,roll,rate,speed,g,features,act,failed=x
  nxt,_=transition(ret,roll=roll,roll_rate=rate,speed_error=speed-g[0],lateral_error=features[0],heading_error=features[1],action=act,alpha=1.,dt=.005,alive_rate=source['alive_reward_rate'],failure_penalty=source['failure_penalty'],failed=failed,enabled=i*.005>=tr.start_seconds,config=tr,longitudinal_error=0.,yaw_rate_error=0.)
  return nxt,ret._asdict()
 assert tr.geometric and tr.objective=='asymmetric_geometric_huber'
 _,rd=jax.jit(lambda:jax.lax.scan(retstep,initial_return(),(jp.arange(n),jp.asarray(d['phi']),jp.asarray(d['phi_dot']),jp.asarray(d['actual_forward_speed']),jp.asarray(d['governed']),jp.asarray(postpath),jp.asarray(d['lower_action']),jp.asarray(d['physical_failure']))))()
 diag.update({'return_'+k:v for k,v in jax.device_get(rd).items()});diag['pre_measurement']=pre
 initial_pose=env.helpers.pose(state.physical.data)
 def integrate_step(p,g):
  end=integrate(p,g,.005,cc.wheelbase,cc.caster);return end,end
 _,refs=jax.jit(lambda:jax.lax.scan(integrate_step,initial_pose,jp.asarray(d['governed'])))()
 curvature=jp.cos(cc.caster)*jp.tan(jp.asarray(d['governed'][:,1]))/cc.wheelbase
 points=jp.concatenate([jp.concatenate([initial_pose,curvature[:1]])[None],jp.concatenate([refs,curvature[:,None]],axis=1)])
 poses=np.column_stack([d['actual_xy'],d['yaw_wrapped']]);preposes=jp.asarray(np.vstack([np.asarray(initial_pose),poses[:-1]]))
 def geometry(_,x):
  i,pose=x;st=FrozenLowerState(None,None,points[i,:3],points,jp.int32(i+1),jp.bool_(False))
  features,ray=path_features(st,pose)
  return None,dict(**projection_details(st,pose),features=features)
 _,geom=jax.jit(lambda:jax.lax.scan(geometry,None,(jp.arange(n),preposes)))()
 geom=jax.device_get(geom);diag.update({'projection_'+k:v for k,v in geom.items()})
 checks[method]['max_path_feature_reconstruction_abs']=np.max(abs(geom['features']-d['lower_path_features']),axis=0).tolist()
 jumps=np.flatnonzero((abs(np.diff(geom['progress']))>.25)|(abs(np.diff(d['lower_path_features'][:,1]))>.3))+1
 checks[method]['projection_discontinuities']=[dict(time=float(d['time'][i]),progress_before=float(geom['progress'][i-1]),progress_after=float(geom['progress'][i]),index_before=int(geom['segment_index'][i-1]),index_after=int(geom['segment_index'][i]),endpoint_before=bool(geom['endpoint_ray'][i-1]),endpoint_after=bool(geom['endpoint_ray'][i]),heading_before=float(d['lower_path_features'][i-1,1]),heading_after=float(d['lower_path_features'][i,1])) for i in jumps]

 np.savez_compressed(OUT/f'{method}.npz',**diag)
 print(method,checks[method],flush=True)
(OUT/'checks.json').write_text(json.dumps(checks,indent=2))
