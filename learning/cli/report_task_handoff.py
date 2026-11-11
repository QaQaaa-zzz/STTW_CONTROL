"""Report grouped task-conditioned fit and all actual physical trajectories."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from sttw_control.handoff_model import HandoffContinuationModel
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',type=Path,required=True);p.add_argument('--collection',type=Path,required=True);a=p.parse_args()
r=json.loads((a.model/'results.json').read_text());fig,axes=plt.subplots(2,3,figsize=(15,7),layout='constrained');reward,ra=plt.subplots(2,3,figsize=(15,7),layout='constrained');cumulative,ca=plt.subplots(2,3,figsize=(15,7),layout='constrained');bank=np.load(a.collection/'roads.npz')['xy']
for col,name in enumerate(['straight','left','right']):
 run=a.collection/name;z=np.load(run/'evaluation/trajectories.npz');pr=json.loads((run/'protocol.json').read_text());cases=json.loads((run/'evaluation/summary.json').read_text())['cases'];root=z['root_xyzw'];valid=z['valid'];rew=z['reward']
 for row,role in enumerate(['TRAIN','DEV']):
  ax=axes[row,col];count=goals=0
  for case,info in zip(cases,pr['cases']):
   if info['role']!=role:continue
   lane=case['lane'];v=valid[:,lane];xy=root[v,lane,:2];color='tab:blue' if case['success'] else 'tab:red';ax.plot(xy[:,0],xy[:,1],color=color,alpha=.25,lw=.6);ax.scatter(*xy[-1],color=color,marker='o' if case['success'] else 'x',s=8);count+=1;goals+=case['success']
   ra[row,col].plot(z['time'][v],rew[v,lane],color=color,alpha=.15,lw=.5);ca[row,col].plot(z['time'][v],np.cumsum(rew[v,lane]),color=color,alpha=.15,lw=.5)
  ref=bank[pr['downstream_route_id']];ax.plot(ref[:,0],ref[:,1],'k--',lw=.8);ax.set(xlabel='world x(m)',ylabel='world y(m)',title=f'{role} {name}: {goals}/{count}',xlim=(0,19),ylim=(-4,4));ax.set_aspect('equal');ax.grid(alpha=.2)
  for aa,label in [(ra[row,col],'Per-step task reward'),(ca[row,col],'Cumulative task reward')]:aa.set(xlabel='time(s)',ylabel=label,title=f'{role} {name}');aa.grid(alpha=.2)
fig.suptitle('All540 executions; blue goal/red failure; dashed downstream reference; true endpoints')
for f,name in [(fig,'xy_all.png'),(reward,'reward_all.png'),(cumulative,'cumulative_reward_all.png')]:f.savefig(a.model/name,dpi=160);plt.close(f)
f,ax=plt.subplots(1,2,figsize=(11,4),layout='constrained')
for file in sorted(a.model.glob('*_curve.json')):
 curve=np.array(json.loads(file.read_text()));ax[0].plot(curve[:,0],curve[:,1],label=file.stem);ax[1].plot(curve[:,0],curve[:,2],label=file.stem)
 with SummaryWriter(str(a.model/'tensorboard'/file.stem)) as w:
  for e,l,v in curve:w.add_scalar('loss/train_bce',l,int(e));w.add_scalar('validation/brier',v,int(e))
ax[0].set(xlabel='Epoch',ylabel='TRAIN BCE');ax[1].set(xlabel='Epoch',ylabel='DEV Brier');ax[1].legend(fontsize=6);f.savefig(a.model/'training.png',dpi=160);plt.close(f)
z=np.load(a.model/'dataset.npz');x=z['x'];obs=np.zeros((len(x),80),np.float32);obs[:,50:75]=x[:,13:38]
model=HandoffContinuationModel(a.model,feature_set='task_context');pred=model.predict(x[:,:13],obs,x[:,38],x[:,39:])['conditional_goal_probability'];saved=np.load(a.model/'predictions.npz')['task_context'];error=float(abs(pred-saved).max());assert error<1e-6
(a.model/'inference_check.json').write_text(json.dumps(dict(max_probability_error=error,n=len(x),schema='39 measured descriptor +19 downstream geometry'),indent=2))
lines=['# Task-conditioned handoff model','','![All actual/reference XY](xy_all.png)','','![Per-step rewards](reward_all.png)','','![Cumulative rewards](cumulative_reward_all.png)','','![Training and selection](training.png)','','540 newly executed full prefixes: original12roots x15straight-jump candidates x3downstream tasks. Original8TRAIN/4DEV roots retained. Each task switches at actual handoff; none of the jumping or H75 policies is trained. Whole-task failures stay in the coverage table; the downstream fit is conditional on valid handoff.','','|Task|Goals/180|Valid handoffs|Prior-road handoffs|','|---|---:|---:|---:|']
for c in r['coverage']:lines.append(f"|{c['variant']}|{c['goals']}/180|{c['valid_handoffs']}|{c['prior_road_handoffs']}|")
lines+=['','|Role/task|N|Goals|State-only Brier|Task-context Brier|Constant|Task prior|','|---|---:|---:|---:|---:|---:|---:|']
for g in r['groups']:
 b=g['brier'];lines.append(f"|{g['role']}/{g['variant']}|{g['n']}|{g['goals']}|{b['state_only']:.5f}|{b['task_context']:.5f}|{b['constant']:.5f}|{b['task_prior']:.5f}|")
lines+=['','Inputs:39 measured root/latest frame/time plus19 geometric features:6world-x lookahead points,each with lateral offset,heading error,curvature,and shared road half-width. No scene-ID input. Width is fixed in these data,so width generalization is not evaluated. World-x-monotone downstream paths only.','', 'Both learned comparators use64/32SiLU,3seeds,200epochs,BCE/Adam and original grouped DEV checkpoint selection. Input widths and parameter counts differ. TRAIN-only normalization. Task prior is learned onTRAIN labels only. No untouched TEST or changed physical-map generalization.','', 'Task-conditioned DEV Brier is slightly worse overall,especially on right tasks; no prediction superiority claim. Functional task conditioning must still be evaluated in actual online selection. Do not infer planning improvement from the existence of nonidentical task inputs.','',f"Actual physics lane steps {sum(c['physics_lane_steps'] for c in r['coverage'])}; summed simulation wall {sum(c['elapsed_s'] for c in r['coverage']):.2f}s. Fitting/data-load wall {r['elapsed_s']:.2f}s;0PPO. Initial data reader repeatedly decompressed arrays; source now caches per-variant arrays without changing labels. Executed source snapshot retained in raw run. Reload parity maximum{error:.3g} on{len(x)}descriptors.",'',f'Raw collection: `{a.collection}`; weights/curves: `{a.model}`.']
(a.model/'RESULTS.md').write_text('\n'.join(lines)+'\n');print('Report complete; reload parity',error)
