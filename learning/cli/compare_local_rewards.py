"""Re-score saved lower trajectories with the original Local300 reward source."""
import argparse
import ast
import json
import math
from pathlib import Path
import jax
import jax.numpy as jp
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser()
p.add_argument('--config',type=Path,required=True)
p.add_argument('--reward-spec',type=Path,required=True)
p.add_argument('--runs',type=Path,nargs='+',required=True)
p.add_argument('--zero',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();cfg=json.loads(a.config.read_text());spec=json.loads(a.reward_spec.read_text())
source=Path(__file__).resolve().parents[1]/'src/sttw_control/lower_command.py'
# Same direct source reuse as the command-bank exporter; no replacement formula.
module=ast.parse(source.read_text())
functions=ast.Module(body=[node for node in module.body if isinstance(node,ast.FunctionDef) and node.name in {'huber','local_reward_terms'}],type_ignores=[])
exec(compile(functions,str(source),'exec'),globals())
score=jax.jit(jax.vmap(lambda ev,ed,roll,action,previous,failed,rate,left,peak:
    local_reward_terms(ev,ed,roll,action,previous,failed,spec,roll_rate=rate,remaining_ticks=left,peak_roll=peak)))
a.output.mkdir(parents=True,exist_ok=True)
datasets=[(folder.name,np.load(folder/'trajectory.npz')) for folder in a.runs]+[('Lab ECBC+ESO',np.load(a.zero/'trajectory.npz'))]
summary=[];saved={};styles=['-','--','-.',':'];colors=['#0072B2','#D55E00','#009E73','#CC79A7','#E69F00','#000000']
overview,axs=plt.subplots(2,len(cfg['cases']),figsize=(23,7),squeeze=False)
for i,case in enumerate(cfg['cases']):
 old=np.load(Path(cfg['mujoco_reference'])/(case+'.npz'))
 models=[('MuJoCo Local300',old['actual_forward_speed']-old['governed'][:,0],old['actual_delta']-old['governed'][:,1],old['phi'],old['action'],old['physical_failure'],old['phi_dot'],old['peak_roll'])]
 for label,data in datasets:
  n=int(data['valid'][:,i].sum())
  models.append((label,data['speed'][:n,i]-data['reference'][:n,i,0],data['measurement'][:n,i,2]-data['reference'][:n,i,1],data['measurement'][:n,i,0],data['action'][:n,i],data['failed'][:n,i],data['measurement'][:n,i,1],data['peak_roll'][:n,i]))
 parts_fig,parts_axes=plt.subplots(2,4,figsize=(17,8))
 for j,(label,ev,ed,roll,action,failed,rate,peak) in enumerate(models):
  n=len(ev);previous=np.concatenate((np.zeros_like(action[:1]),action[:-1]));left=round(spec['episode_seconds']/spec['dt'])-np.arange(n)
  terms={key:np.asarray(value) for key,value in score(ev,ed,roll,action,previous,failed,rate,left,peak).items()}
  steps=np.arange(1,n+1);color=colors[j%len(colors)];style=styles[j%len(styles)]
  for k,values in enumerate((terms['reward'],np.cumsum(terms['reward']))):
   axs[k,i].plot(steps,values,style,color=color,label=label);axs[k,i].scatter(steps[-1],values[-1],marker='x',color=color,s=15)
  component_names=[key for key in terms if key!='reward']
  for ax,key in zip(parts_axes.flat,component_names):
   ax.plot(steps,terms[key],style,color=color,label=label);ax.set(title=key,xlabel='Actual control step',ylabel='Signed reward');ax.grid(alpha=.2)
  row=dict(case=case,model=label,observed_steps=n,failed=bool(np.any(failed)),mean_step_reward=float(np.mean(terms['reward'])),total_reward=float(np.sum(terms['reward'])),component_totals={key:float(value.sum()) for key,value in terms.items() if key!='reward'})
  if j==0:row['source_log_max_absolute_difference']=float(np.max(np.abs(terms['reward']-old['reward'])))
  summary.append(row)
  for key,value in terms.items():saved[f'{case}/model{j}/{key}']=value
  saved[f'{case}/model{j}/speed_error']=ev;saved[f'{case}/model{j}/steer_error']=ed
 parts_axes.flat[0].legend(fontsize=7);parts_fig.suptitle(case+' | original Local300 reward components | no external disturbance');parts_fig.tight_layout();parts_fig.savefig(a.output/(case+'_reward_parts.png'),dpi=120);plt.close(parts_fig)
 for k,label in enumerate(('Per-step reward','Cumulative reward')):
  axs[k,i].set(title=case,xlabel='Actual control step',ylabel=label);axs[k,i].grid(alpha=.2)
axs[0,0].legend(fontsize=7)
overview.suptitle('Original Local300 reward, frozen standard commands | curves stop at actual endpoints\nRecomputed with original source; reward rank alone does not establish tracking or transfer qualification')
overview.tight_layout();overview.savefig(a.output/'reward_overview.png',dpi=130);plt.close(overview)
np.savez_compressed(a.output/'reward_parts.npz',**saved)
(a.output/'reward_comparison.json').write_text(json.dumps(dict(config=str(a.config.resolve()),reward_spec=str(a.reward_spec.resolve()),source=str(source),runs=[str(p.resolve()) for p in a.runs],zero=str(a.zero.resolve()),cases=summary),indent=2))
print(json.dumps(summary,indent=2))
