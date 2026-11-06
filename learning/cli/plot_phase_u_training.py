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
(a.run/'training_scalar_summary.json').write_text(json.dumps(data))
fig,axes=plt.subplots(2,3,figsize=(14,7),layout='constrained');axes=axes.flatten()
panels=[['Train/mean_step_reward_pre_update'],['Episode/reward'],['Episode/apex_fraction','Episode/physical_failure'],['Loss/value_function'],['Loss/surrogate'],['Loss/learning_rate']]
for ax,tags in zip(axes,panels):
 for tag in tags:
  if tag in data:
   xy=np.asarray(data[tag]);ax.plot(xy[:,0],xy[:,1],label=tag)
 ax.set_xlabel('Update');ax.grid(alpha=.25);ax.legend(fontsize=7)
best=json.loads((a.run/'best_step_model.json').read_text());axes[0].axvline(best['update'],color='gray',ls=':')
axes[0].set_title(f"Best sampled pre-update policy: {best['update']}")
axes[2].set_ylabel('Fraction of logged completed episodes');axes[5].set_yscale('log')
fig.suptitle(a.run.name+'\nTraining diagnostics only; apex is not full-jump success; KL unavailable unless logged')
fig.savefig(a.run/'training_curves.png',dpi=150)
print(json.dumps(dict(run=str(a.run),best_update=best['update'],recorded_tags=list(data)),indent=2))
