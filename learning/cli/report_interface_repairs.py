"""Compare proposed model repairs against frozen data; do not impute online gains."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from sttw_control.handoff_model import HandoffContinuationModel
from sttw_control.handoff_task_context import route_context
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',type=Path,required=True);a=p.parse_args();b=a.base;out=b/'chapter3_review_repairs';out.mkdir(exist_ok=True);torch.set_num_threads(2)
old=b/'chapter3_interface_all_routes/upstream_distance';new=b/'chapter3_compact_interface';d=np.load(old/'dataset.npz');oldp=np.load(old/'predictions.npz')['predicted_handoff_mean'];newp=np.load(new/'predictions.npz')['predicted_handoff_mean'];mask=d['dev']&d['hit'];target=d['target'];fields=[('x',0),('y',1),('vx',7),('vy',8),('wx',10),('time',38)];stats=[]
fig,axs=plt.subplots(2,3,figsize=(13,7),layout='constrained')
for ax,(name,j) in zip(axs.flat,fields):
 y=target[mask,j];row=dict(field=name,observed_std=float(y.std()))
 for label,pred,color in [('raw582',oldp,'tab:blue'),('compact55',newp,'tab:orange')]:
  x=pred[mask,j];ax.scatter(y,x,s=7,alpha=.6,label=label,c=color);row[label]=dict(rmse=float(np.sqrt(((x-y)**2).mean())),std=float(x.std()))
 lo,hi=min(y.min(),oldp[mask,j].min(),newp[mask,j].min()),max(y.max(),oldp[mask,j].max(),newp[mask,j].max());ax.plot([lo,hi],[lo,hi],'k--',lw=.7);unit={'x':'m','y':'m','vx':'m/s','vy':'m/s','wx':'rad/s','time':'s'}[name];ax.set(xlabel=f'Observed {name} ({unit})',ylabel=f'Predicted mean ({unit})');ax.legend(fontsize=7);stats.append(row)
fig.savefig(out/'handoff_predictions.png',dpi=160);plt.close(fig)
f,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
for label,path in [('raw582',old),('compact55',new)]:
 for seed in [71101,71102,71103]:
  curve=np.array(json.loads((path/f'upstream_{seed}_curve.json').read_text()))
  writer=SummaryWriter(str(out/'tensorboard'/f'{label}_{seed}'))
  for epoch,loss,dev in curve:writer.add_scalar('loss/train',loss,int(epoch));writer.add_scalar('loss/dev',dev,int(epoch))
  writer.close()
  for ax,col in zip(axes,[1,2]):ax.plot(curve[:,0],curve[:,col],label=f'{label}/{seed}')
for ax,label in zip(axes,['TRAIN objective','DEV objective']):ax.set(xlabel='Epoch',ylabel=label);ax.legend(fontsize=6)
f.savefig(out/'training.png',dpi=160);plt.close(f)
modeldir=b/'chapter3_frenet_condition_model';model=HandoffContinuationModel(modeldir,feature_set='task_context');decl=json.loads((modeldir/'declaration.json').read_text());bank=np.load(b/'chapter3_handoff_routes/issued.npz');historical=[]
for variant in ['straight','left','right']:
 run=b/'chapter3_handoff_routes'/variant;cfg=json.loads((run/'config.json').read_text());protocol=json.loads((run/'protocol.json').read_text());rid=protocol['downstream_route_id'];route={k:bank[k][rid] for k in ['xy','heading','curvature']};z=np.load(run/'evaluation/trajectories.npz');obs=z['observation'];events=z['event_root_xyzw'];times=z['event_time'];summary=json.loads((run/'evaluation/summary.json').read_text())['cases']
 for lane in [1,3,5,7]:
  root=events[lane,3][None];t=times[lane,3];context=route_context(root,route,.6,decl['mount_wxyz'],'frenet');prob=model.predict(root,obs[round(t/.02),lane:lane+1],[t],context)['conditional_goal_probability'][0]
  historical.append(dict(source=lane,route=variant,prediction=float(prob),success=summary[lane]['success'],relative_features=context[0].tolist()))
rdir=b/'chapter3_interface_all_routes/downstream_goal_distance';records=json.loads((rdir/'results.json').read_text())['cases'];z=np.load(rdir/'dataset.npz');pr=np.load(rdir/'predictions.npz')['state_history'];rb=[];fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
for ax,name in zip(axes,['source240','rb400_240']):
 m=np.array([r['dataset']==name for r in records]);x=z['x'][m];y=z['y'][m];ax.scatter(x[:,8],x[:,10],c=y,cmap='coolwarm',s=14,vmin=0,vmax=1);ax.set(xlabel='world vy (m/s)',ylabel='world wx (rad/s)',title=name+'; red goal/blue fail')
 rb.append(dict(dataset=name,n=int(m.sum()),goals=int(y.sum()),mean_abs_vy=float(abs(x[:,8]).mean()),mean_abs_wx=float(abs(x[:,10]).mean()),mean_fixed_R=float(pr[m].mean())))
fig.savefig(out/'rb_handoff_distribution.png',dpi=160);plt.close(fig)
result=dict(prediction=stats,historical=historical,rb=rb,scope='Original grouped DEV and previously inspected historical cases;model changes are bounded development;no prespecified improvement claimed')
(out/'model_comparison.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
