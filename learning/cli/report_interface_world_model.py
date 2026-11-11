"""Evidence report for upstream distributions and composed conditional predictions."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();r=json.loads((a.run/'results.json').read_text());d=np.load(a.run/'dataset.npz');f=np.load(a.run/'predictions.npz')
for curve_path in a.run.glob('*_curve.json'):
 with SummaryWriter(str(a.run/'tensorboard'/curve_path.stem)) as writer:
  for epoch,loss,dev in json.loads(curve_path.read_text()):
   writer.add_scalar('fit/train_loss',loss,epoch);writer.add_scalar('selection/dev_score',dev,epoch)
mask=d['dev']&d['hit'];actual=d['target'][mask];pred=f['predicted_handoff_mean'][mask]
fig,axs=plt.subplots(2,3,figsize=(12,7),layout='constrained')
for ax,k,label in zip(axs.flat,[0,1,7,8,10,38],['Handoff x (m)','Handoff y (m)','vx (m/s)','vy (m/s)','world roll rate (rad/s)','handoff time (s)']):
 ax.scatter(actual[:,k],pred[:,k],s=12,alpha=.5);lo=min(actual[:,k].min(),pred[:,k].min());hi=max(actual[:,k].max(),pred[:,k].max());ax.plot([lo,hi],[lo,hi],'k--');ax.set(xlabel='Observed '+label,ylabel='Predicted mean',title=label);ax.grid(alpha=.2)
fig.suptitle('Grouped DEV conditional valid-handoff outputs; not future inputs');fig.savefig(a.run/'handoff_outputs.png',dpi=160);plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(12,4),layout='constrained')
for path in sorted(a.run.glob('upstream_*_curve.json')):
 c=np.array(json.loads(path.read_text()));axs[0].plot(c[:,0],c[:,1],label=path.stem);axs[1].plot(c[:,0],c[:,2],label=path.stem)
axs[0].set(xlabel='epoch',ylabel='TRAIN objective');axs[1].set(xlabel='epoch',ylabel='DEV objective');axs[0].legend(fontsize=7);fig.savefig(a.run/'training.png',dpi=160);plt.close(fig)
lines=['# Two-interface world-model fit','','Actual/reference trajectory source: [all-route handoff labels](../chapter3_interface_labels/RESULTS.md).','','![Conditional handoff outputs](handoff_outputs.png)','','![Training](training.png)','','|Data role|N|Goals|Interface Brier|Direct Brier|Route-prior Brier|Constant Brier|Handoff position RMSE(m)|Handoff time MAE(s)|','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for g in r['groups']:
 role=g['dataset']+' (mixed TRAIN/DEV;not transfer)' if g['dataset']=='source1440' and r['scope'].get('include_distance_training') else g['dataset']
 lines.append('|'+ '|'.join([role,str(g['n']),str(g['goal'])]+[f"{g['brier'][k]:.5f}" for k in ['interface','direct','route_prior','constant']]+[f"{g['handoff_position_rmse_m']:.4f}",f"{g['handoff_time_mae_s']:.4f}"])+'|')
lines+=['',f"CPU total {r['elapsed_s']:.2f}s including data preparation,fit and prediction. Fixed source jump/H75/cold reset/map. No PPO,new physics or RB-upstream training.",'', 'Inputs582: query root13,observation80,ground controller history352,time1,candidate8,known local map128. Upstream three-member128/64SiLU MLP outputs39D Gaussian mean/log-variance,valid-handoff probability and a ground-route goal probability. Invalid or missing handoffs remain Bernoulli negatives; conditional state loss uses valid handoffs. Original720+published1199 retain all candidate failures; unpublished241 are not attributed to requested candidates.','', 'The39D handoff output contains root13,latest measured25D frame and physical time1. The frozen downstream three-member64/32SiLU MLP predicts full-goal continuation under the declared cold reset. Jump score averages P(valid handoff) times the conditional goal prediction over32 sampled handoff states per upstream member. Ground bypass has no jump interface and uses a separate direct-goal branch. Intermediate event propagation/PCA prediction is not used. The2s continuation pilot remains separate auxiliary evidence.','', 'Three seeds and250epochs for upstream and direct comparator; TRAIN-only normalization; grouped DEV selects upstream likelihood objective and direct Brier checkpoints. Source720 has8TRAIN/4DEV roots; source1440 has6TRAIN/2DEV roots. Across the20 declared initial roots, no exact initial-condition duplicate crosses TRAIN/DEV. All earlier data have been inspected; source240 is historical transfer,not a new independent TEST.','', 'DEV Brier measures model selection performance. Route-prior comparison remains strong; no universal superiority is claimed. Diagonal Gaussian samples and ensemble disagreement are not certified feasible states or calibrated physical margins. Known-map/cold-reset context is frozen rather than learned over changing terrain or initialization. The compact downstream input does not serialize the PhysX solver or all controller memory.','', 'The early-query-only predecessor is retained in the sibling upstream directory: its transfer handoff position RMSE2.32m/time MAE3.01s motivated using the already partitioned distance-training data. This is declared development,not hidden TEST tuning. No new frozen online roots were consulted before this model was frozen.']
(a.run/'RESULTS.md').write_text('\n'.join(lines)+'\n')
