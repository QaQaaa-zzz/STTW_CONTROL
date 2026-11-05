"""Reconstruct the original frozen path reward before PPO/GAE, no physics."""
import argparse,json
from pathlib import Path
import numpy as np
import jax
import jax.numpy as j
from sttw_control.path_command_reward import costs,failure_reward
from sttw_control.geometric_path import Path as Route,project
from sttw_control.controller import ControllerConfig,_system
p=argparse.ArgumentParser();p.add_argument('--receipt',type=Path,required=True);p.add_argument('--bundle',type=Path,required=True);p.add_argument('--alpha',type=float,choices=(0,.25,.5,.75,1));p.add_argument('--output-prefix',type=Path);a=p.parse_args()
r=json.loads(a.receipt.read_text());cfg=json.loads((a.bundle/'config.json').read_text());d=np.load(a.receipt.with_suffix('.npz'));mesh=np.load(a.bundle/'route.npz');route=Route(**{k:j.asarray(mesh[k]) for k in ('s','xy','heading','curvature','turn_end','goal')})
c=d['control_trace'];m=d['middle_trace'];states=d['body_states'][:,0];n=len(c)
if n==0:raise ValueError('no physical control transitions')
end_index=np.minimum((np.arange(n)+1)*25,len(states))-1;post=states[end_index]
q=states[:,3:7];x,y,z,w=q.T;roll=-(np.arctan2(2*(w*x+y*z),1-2*(x*x+y*y))-np.pi/2)
x,y,z,w=post[:,3:7].T;yaw=np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z));speed=post[:,7]*np.cos(yaw)+post[:,8]*np.sin(yaw)
def inv(q,v):
 xyz=q[:,:3];w=q[:,3:];return v-2*w*np.cross(xyz,v)+2*np.cross(xyz,np.cross(xyz,v))
contract=json.load(open(Path(r['asset']).with_name('contract.json')));site=np.array(contract['imu_site_quat_wxyz']);site=np.tile(np.r_[site[1:],site[0]],(n,1));gyro=inv(site,inv(post[:,3:7],post[:,10:13]));rollrate=-gyro[:,0]
peak=np.array([max(abs(c[i,1]),np.max(np.abs(roll[i*25:end_index[i]+1]))) for i in range(n)])
projection=jax.jit(jax.vmap(lambda xy,s,dist:project(route,xy,s,dist,cfg)))(j.asarray(post[:,:2]),j.asarray(m[:,5]),j.asarray(np.linalg.norm(post[:,:2]-c[:,8:10],axis=-1)))
eh=np.arctan2(np.sin(np.asarray(projection.heading)-yaw),np.cos(np.asarray(projection.heading)-yaw));curvature=np.interp(np.asarray(projection.progress),mesh['s'],mesh['curvature']);cc=ControllerConfig(**r['controller_config']);ratio=float(_system(r['drive_speed'],j.zeros(3),cc)[3]);chi=np.clip((np.abs(np.arctan(cc.wheelbase*curvature/np.cos(cc.caster))/ratio)-.18)/.12,0,1)
offsets=m[:,3:5]-m[:,1:3];alpha=r['path_contract']['alpha'] if a.alpha is None else a.alpha
if a.alpha is not None and not r['path_contract']['zero_middle']:raise ValueError('alpha rescoring is only for the alpha-independent zero-middle baseline')
calc=jax.jit(jax.vmap(lambda ch,ev,ey,eh,pr,rr,v,o:costs(alpha=alpha,chi=ch,ev=ev,ey=ey,eh=eh,peak_roll=pr,roll_rate=rr,speed=v,offsets=o,offset_rate=j.zeros(2),config=cfg)))
reward=calc(j.asarray(chi),j.asarray(speed-r['drive_speed']),projection.cross_track,j.asarray(eh),j.asarray(peak),j.asarray(rollrate),j.asarray(speed),j.asarray(offsets))
keys=list(reward['raw']);raw=np.stack([np.asarray(reward['raw'][k]) for k in keys],axis=-1);effective=np.stack([np.asarray(reward['effective'][k]) for k in keys],axis=-1);tick_reward=np.asarray(reward['reward']).copy();rate_index=keys.index('offset_rate');cr=cfg['path_reward'];previous=np.zeros(2)
for start in range(0,n,4):
 end=min(start+4,n);rate=(offsets[end-1]-previous)/.02;value=cr['offset_rate_weight']*np.sum((rate/np.array(cr['offset_rate_scales']))**2);bounded=min(value,cr['component_caps']['offset_rate']);raw[start:end,rate_index]=value;effective[start:end,rate_index]=bounded;tick_reward[start:end]-=cr['scale']*.005*bounded;previous=offsets[end-1]
np.testing.assert_allclose(tick_reward,-cr['scale']*.005*effective.sum(-1),atol=2e-5,rtol=2e-5)
policy_reward=np.array([tick_reward[i:i+4].sum() for i in range(0,n,4)]);terminal_penalty=0.
if r['termination']!='budget_end':
 remaining=r['requested_physics_steps']/100-(len(policy_reward)-1);terminal_penalty=float(failure_reward(remaining,cfg));policy_reward[-1]=terminal_penalty
scored=tick_reward.copy()
if terminal_penalty:
 start=(len(policy_reward)-1)*4;scored[start:]=0;scored[-1]=terminal_penalty
obs=m[::4,7:358];latent=m[::4,358:360];remaining=1-np.arange(len(obs))*100/r['requested_physics_steps'];critic=np.column_stack((obs,remaining))
prefix=a.receipt if a.output_prefix is None else a.output_prefix
out=prefix.with_suffix('.rewards.npz');np.savez_compressed(out,observation=obs,critic=critic,latent=latent,reward=policy_reward,tick_reward=scored,normal_tick_reward=tick_reward,raw_components=raw,effective_components=effective,component_names=np.asarray(keys),xy=post[:,:2],speed=speed,cross=np.asarray(projection.cross_track),heading_error=eh,peak_roll=peak,offsets=offsets)
meta=dict(policy_transitions=len(obs),lower_transitions=n,reward_total=float(policy_reward.sum()),terminal_penalty=terminal_penalty,termination=r['termination'],alpha=alpha,role='training reward reconstructed before GAE/optimization; original v3 costs with explicit engineering termination',source_config_sha256=r['path_contract']['bundle']['files']['config.json'])
prefix.with_suffix('.rewards.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta))
