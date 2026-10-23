import json,pickle
from pathlib import Path
from collections import defaultdict
import numpy as np
import jax
import jax.numpy as j
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sttw_control.path_command_reward import costs,failure_reward,configured_cost_keys
from sttw_control.controller import ControllerConfig
from sttw_control.direct_command_policy import q_ratio
from sttw_control.geometric_path import build_path
root=Path('runs/short_recovery_post_v1_recovery1/evaluation');out=root/'diagnostics/reward_panels';out.mkdir(exist_ok=True)
manifest=json.loads((root/'manifest.json').read_text());campaign=Path('runs/short_recovery_train_v1_recovery1')
cfg=json.loads((campaign/'alpha0_config.json').read_text());plan=json.loads((campaign/'uniform_alpha0/plan.json').read_text());source=json.loads(Path(plan['parent_manifest']).read_text())['source'];cc=ControllerConfig(**source['controller']);keys=configured_cost_keys(cfg)
with (root/'parent_left90_R2_v2.45_a0_t0.8_snapshot.pkl').open('rb') as f:pose=np.asarray(pickle.load(f).physical.reference_pose)
paths={r:build_path(r,pose,cfg) for r in manifest['protocol']['routes']}
for r,p in paths.items():
 with np.load(root/(r+'_path.npz')) as x:np.testing.assert_array_equal(p.xy,x['xy'])
@jax.jit
def tick_cost(ev,ey,eh,pr,rr,v,o,chi,alpha):
 return jax.vmap(lambda a,b,c,d,e,f,g,h:costs(alpha=alpha,chi=h,ev=a,ey=b,eh=c,peak_roll=d,roll_rate=e,speed=f,offsets=g,offset_rate=j.zeros(2),config=cfg))(ev,ey,eh,pr,rr,v,o,chi)
rows=json.loads((root/'results.json').read_text())['cases'];groups=defaultdict(list)
for row in rows:groups[tuple(row[k] for k in ['route','speed','onset_s','force_N'])].append(row)
colors=dict(zip(manifest['protocol']['policies'],['#555555','#0072B2','#E69F00']));index=[];allsummary=[]
for number,(context,group) in enumerate(sorted(groups.items()),1):
 fig,axes=plt.subplots(2,4,figsize=(15,7),squeeze=False);archive={}
 for row in group:
  with np.load(root/(row['name']+'.npz')) as x:d={k:x[k] for k in x.files}
  n=len(d['time']);vuser=row['speed'];path=paths[row['route']]
  curv=np.asarray(j.interp(j.asarray(d['path_progress']),path.s,path.curvature))
  ff=np.arctan(cc.wheelbase*curv/np.cos(cc.caster));chi=np.clip((np.abs(ff/float(q_ratio(j.float32(vuser),cc)))-.18)/.12,0,1)
  offsets=d['governed']-d['nominal']
  c=jax.device_get(tick_cost(*[j.asarray(a) for a in [d['actual_forward_speed']-vuser,d['path_cross_track'],d['path_heading_error'],d['peak_roll'],d['phi_dot'],d['actual_forward_speed'],offsets,chi]],j.float32(row['alpha'])))
  last=np.minimum(np.arange((n+3)//4)*4+3,n-1);end=offsets[last];before=np.vstack([np.zeros((1,2),np.float32),end[:-1]])
  rate=(end-before)/.02;w=cfg['path_reward'];rawrate=w['offset_rate_weight']*np.sum((rate/np.array(w['offset_rate_scales'],np.float32))**2,axis=-1)
  effective=np.stack([c['effective'][k] for k in keys],axis=-1);effective[:,keys.index('offset_rate')]=np.repeat(np.minimum(rawrate,w['component_caps']['offset_rate']),4)[:n]
  signed=-w['scale']*.005*effective;tickreward=signed.sum(axis=-1)
  penalty=np.zeros(n,np.float32)
  if row['physical_failure'] or row['domain_failure']:
   start=((n-1)//4)*4;tickreward[start:]=0
   terminal=float(failure_reward(1000-(n-1)//4,cfg));tickreward[-1]=terminal
   # Terminal policy interval replaces all its running costs, as in policy_step.
   signed[start:]=0;penalty[-1]=terminal
  reward=np.add.reduceat(tickreward,np.arange(0,n,4));components=np.add.reduceat(signed,np.arange(0,n,4),axis=0)
  terminal=np.add.reduceat(penalty,np.arange(0,n,4));t=d['time'][last]+.005
  np.testing.assert_allclose(components.sum(-1)+terminal,reward,rtol=2e-6,atol=2e-5)
  a=row['alpha'];label=row['policy'];color=colors[label]
  for ax in axes[a]:ax.axvspan(row['onset_s'],row['onset_s']+.1,color='gray',alpha=.14)
  axes[a,0].plot(t,reward,label=label,c=color,lw=.8);axes[a,1].plot(t,np.cumsum(reward),label=label,c=color,lw=.8)
  axes[a,2].plot(d['time'],d['path_cross_track'],label=label,c=color,lw=.8)
  axes[a,3].plot(d['time'],d['actual_forward_speed']-vuser,label=label,c=color,lw=.8)
  for k,value in dict(time=t,reward=reward,cumulative_reward=np.cumsum(reward),signed_reward_components=components,terminal_reward=terminal).items():archive[f'{label}_a{a}_{k}']=value
  allsummary.append(dict(name=row['name'],return_reconstructed=float(reward.sum()),policy_intervals=len(reward),terminal_reward=float(terminal.sum())))
 for a in [0,1]:
  for col,title in enumerate(['Reward / upper step','Cumulative reward','Signed path error (m)','Speed error (m/s)']):
   axes[a,col].set(xlabel='Task time (s)',title=f'alpha={a}: {title}');axes[a,col].grid(alpha=.2)
  axes[a,0].legend(fontsize=6)
 route,speed,onset,force=context;fig.suptitle(f'{route}, v={speed:g}, pulse {force:+g} N at {onset:g}s | parent u50 / candidates u27')
 fig.text(.01,.008,'Rewards reconstructed from saved lower-tick traces + unchanged source/config; terminal replacement retained. Curves stop at true endpoints.',fontsize=7)
 fig.tight_layout(rect=(0,.025,1,.96));name=f'paired_{route}_v{speed:g}_t{onset:g}_f{force:+g}'
 fig.savefig(out/(name+'.png'),dpi=100);plt.close(fig)
 np.savez_compressed(out/(name+'.npz'),**archive,component_names=np.array(keys))
 index.append(name+'.png')
 if number%24==0:print('plots',number,'/',len(groups),flush=True)
(out/'index.json').write_text(json.dumps(dict(figures=index,rows=allsummary,scope='Source/config reconstruction, not originally logged rewards; signed components and terminal replacement retained; no new simulation'),indent=2)+'\n')
print('DONE',len(index),flush=True)
