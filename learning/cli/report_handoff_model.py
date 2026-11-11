"""Report frozen downstream fit without relabeling historical evaluation as TEST."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();r=json.loads((a.run/'results.json').read_text());d=np.load(a.run/'predictions.npz')
for curve_path in sorted(a.run.glob('*_curve.json')):
 with SummaryWriter(str(a.run/'tensorboard'/curve_path.stem)) as writer:
  for epoch,loss,brier in json.loads(curve_path.read_text()):
   writer.add_scalar('fit/train_bce',loss,epoch);writer.add_scalar('selection/dev_brier',brier,epoch)
methods=['train_constant','nearest_neighbor','root_only','state_history'];names=['TRAIN constant','TRAIN nearest neighbor','Root + time MLP','Root + measured frame + time MLP']
fig,axs=plt.subplots(1,2,figsize=(12,4),layout='constrained')
groups=r['groups'][1:];xx=np.arange(len(groups))
for k,(m,label) in enumerate(zip(methods,names)):axs[0].bar(xx+(k-1.5)*.18,[g['brier'][m] for g in groups],.18,label=label)
axs[0].set_xticks(xx,[g['dataset'].replace('source','S').replace('_','\n') for g in groups]);axs[0].set_ylabel('Brier score (lower is better)');axs[0].legend(fontsize=7);axs[0].set_title('Grouped DEV and historical transfer')
for path in sorted(a.run.glob('state_history_*_curve.json')):
 curve=np.array(json.loads(path.read_text()));axs[1].plot(curve[:,0],curve[:,2],label=path.stem)
axs[1].set(xlabel='Epoch',ylabel='DEV Brier',title='Epoch selection uses DEV only');axs[1].legend(fontsize=7)
fig.savefig(a.run/'model_outputs.png',dpi=160);plt.close(fig)
rows=r['cases'];fig,axs=plt.subplots(2,2,figsize=(10,7),layout='constrained')
for ax,g in zip(axs.flat,groups):
 name=g['dataset'];mask=np.array([row['dataset']=='source720' and row['fit_role']=='DEV' if name=='source720_DEV' else row['dataset']==name for row in rows])
 for label,outcome,marker in [('continued2s',1,'o'),('failed2s',0,'x')]:
  ids=np.flatnonzero(mask&(d['y']==outcome));ax.scatter(ids,d['state_history'][ids],s=18,marker=marker,label=label)
 ax.set(title=f"{name}, n={g['n']}",xlabel='Stored sample index',ylabel='Predicted continuation probability',ylim=(-.03,1.03));ax.legend(fontsize=7)
fig.savefig(a.run/'probability_outputs.png',dpi=160);plt.close(fig)
lines=['# Downstream handoff continuation predictor','','Actual/reference XY for every included source: [frozen label evidence](../chapter3_continuation_labels/RESULTS.md).','','![Prediction comparison and training](model_outputs.png)','','![Per-case probabilities and observed outcomes](probability_outputs.png)','','|Data role|N|Continued2s|Constant|Nearest neighbor|Root+time|Measured frame+root+time|','|---|---:|---:|---:|---:|---:|---:|']
for g in r['groups']:lines.append('|'+ '|'.join([g['dataset'],str(g['n']),str(g['positive'])]+[f"{g['brier'][m]:.5f}" for m in methods])+'|')
lines+=['',f"CPU elapsed {r['elapsed_s']:.3f}s; three seeds per MLP,200 epochs each.82 samples from roots0–7 TRAIN;36 from roots8–11 DEV. DEV selects epochs, so its scores are selection-biased. Other collections are previously inspected historical transfer, not new independent TEST.",'', 'Input39: exact13D handoff root, latest25D measured frame already available before the switching step, physical time1. The frame contains measured gravity/gyro/acceleration, steering/pendulum positions and rates, wheel rates and prior-action context. The field name state_history denotes this compact measured-frame model, not a complete controller-history snapshot. H75 uses the fixed cold reset; no unknown controller state is claimed reconstructed. Root/frame are sampled at the same declared switch boundary; maximum root alignment error0. Input dimension and weights are frozen in saved checkpoints.','', 'Target: observed2s continuation, not full goal. BCE,64/32SiLU MLP,Adam lr.001/weight_decay.0001,TRAIN-only normalization; average three seeded predictions. Per-member disagreement is uncalibrated, not a physical margin. Nearest-neighbor TRAIN score0 is resubstitution, not generalization evidence. All routes here are straight, so a route-only table collapses to the TRAIN constant.','', 'The measured-frame model improves over the constant on DEV,source1440 and RB historical transfers, but is slightly worse on source240. Root-only is slightly better on DEV and worse on these transfer batches. This supports retaining the richer measured input as a candidate, not claiming universal benefit, recovery-domain certification or planning superiority. RB was not used to fit or select this source-trained model.','', 'No upstream handoff-distribution model or online integration is completed by this experiment. Next: fit the upstream model including missing/invalid handoffs,then test composition under the same fixed interface context before new layouts. Saved .pt files and raw arrays stay in the run directory; inference wrapper is learning/src/sttw_control/handoff_model.py.']
(a.run/'RESULTS.md').write_text('\n'.join(lines)+'\n')
