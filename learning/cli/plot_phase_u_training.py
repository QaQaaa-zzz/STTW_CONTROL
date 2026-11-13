"""Summarize recorded Phase U training scalars without additional simulation."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);a=p.parse_args()
events=EventAccumulator(str(a.run/'training'),size_guidance={'scalars':0});events.Reload()
data={tag:[[v.step,v.value] for v in events.Scalars(tag)] for tag in events.Tags()['scalars']}
# Native Go2 logs actor.std; this adapter uses that name for frozen observation normalization.
# Do not interpret the inherited constant tag as action exploration noise.
(a.run/'training_scalar_summary.json').write_text(json.dumps(data))
fig,axes=plt.subplots(2,3,figsize=(14,7),layout='constrained');axes=axes.flatten()
panels=[['Train/mean_step_reward_pre_update'],['Episode/reward'],['Episode/apex_fraction','Episode/physical_failure','Episode/failure','Episode/platform_hold_fraction'],['Loss/value_function'],['Loss/surrogate'],['Loss/learning_rate']]
playground='mean_step_reward' in data
if playground:
 panels=[['mean_step_reward'],['loss_value'],['loss_surrogate'],['kl_first_rollout_step'],['learning_rate'],['sample_s','optimize_s']]
for ax,tags in zip(axes,panels):
 for tag in tags:
  if tag in data:
   xy=np.asarray(data[tag]);ax.plot(xy[:,0]-(1 if tag=='mean_step_reward' else 0),xy[:,1],label=tag)
 ax.set_xlabel('Update');ax.grid(alpha=.25);ax.legend(fontsize=7)
best=json.loads((a.run/'best_step_model.json').read_text());best_update=best.get('update',best.get('completed_updates'));axes[0].axvline(best_update,color='gray',ls=':')
axes[0].set_title(f"Best sampled pre-update policy: {best_update}")
if playground:
 axes[0].set_xlabel('Scored policy update (before optimization)')
 axes[4].set_yscale('log');axes[5].set_ylabel('Wall time (s)')
 rows=[json.loads(line) for line in (a.run/'training.jsonl').read_text().splitlines()]
 counts=np.asarray([row['terminal_counts'] for row in rows]).sum(0)
 (a.run/'terminal_summary.json').write_text(json.dumps(dict(zip(['success','physical_failure','timeout','done'],counts.tolist())),indent=2))
 axes[1].set_title(f'Training terminal events: success {int(counts[0])}/{int(counts[3])}')
else:
 axes[2].set_ylabel('Fraction of logged completed episodes');axes[5].set_yscale('log')
fig.suptitle(a.run.name+'\nTraining diagnostics only; reward and survival do not establish task success; KL unavailable unless logged')
fig.savefig(a.run/'training_curves.png',dpi=150)
print(json.dumps(dict(run=str(a.run),best_update=best_update,recorded_tags=list(data)),indent=2))
