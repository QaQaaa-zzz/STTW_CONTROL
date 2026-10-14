"""Offline retained/discarded update accounting and scalar figure."""
from pathlib import Path
import json,csv,urllib.parse,urllib.request
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
WS=Path('/home/qy/STTW_CONTROL');OUT=Path(__file__).parent
roots={'warm':WS/'runs/worktrees/path-strong-preference/runs/path_strong_preference_200_reboot_resume_20261014/training','scratch':WS/'runs/worktrees/path-unified-scratch/runs/path_unified_scratch_200_20261014/resume_20261014_01'}
parents={'warm':WS/'runs/worktrees/path-strong-preference/runs/path_strong_preference_200_20261014/training','scratch':WS/'runs/worktrees/path-unified-scratch/runs/path_unified_scratch_200_20261014/training'}
rows=[];facts={}
for arm,root in roots.items():
    for stage,r in [('original',parents[arm]),('resume',root)]:
        for line in (r/'shared/metrics.jsonl').read_text().splitlines():
            m=json.loads(line);o=m['optimizer'];retained=stage=='resume' or (arm=='warm' and m['update']<=30)
            rows.append(dict(arm=arm,stage=stage,update_or_sampling_slot=m['update'],retained=retained,mean_step_reward=m['mean_step_reward'],value_loss=o['value'],surrogate_loss=o['surrogate'],mean_kl=o['mean_kl'],accepted_epochs=o['accepted_epochs'],physical_failures=m['physical_failures'],domain_failures=m['tracking_domain_failures'],sample_seconds=m['sample_seconds'],optimize_seconds=m['optimize_seconds']))
    s=json.loads((root/'status.json').read_text());ms=[x for x in rows if x['arm']==arm and x['stage']=='resume'];tb=json.loads((root/'tensorboard_server.json').read_text());q=urllib.parse.urlencode(dict(run='shared',tag='train/mean_step_reward'))
    with urllib.request.urlopen(tb['url']+'/data/environment',timeout=5) as f:env=json.load(f)
    assert env['data_location']==tb['logdir']
    with urllib.request.urlopen(tb['url']+'/data/plugin/scalars/scalars?'+q,timeout=5) as f:scalar=json.load(f)[-1]
    assert scalar[1]==200
    facts[arm]=dict(retained_learning_updates=sum(x['retained'] for x in rows if x['arm']==arm),resume_known_batches=len(ms),resume_physical_failures=sum(x['physical_failures'] for x in ms),resume_domain_failures=sum(x['domain_failures'] for x in ms),mean_last25_sample_reward=float(np.mean([x['mean_step_reward'] for x in ms[-25:]])),resume_sample_seconds=sum(x['sample_seconds'] for x in ms),resume_optimize_seconds=sum(x['optimize_seconds'] for x in ms),resume_dev_seconds=s['dev_seconds'],resume_elapsed_seconds=s['endpoint_elapsed_s'],tensorboard_url=tb['url'],http_latest_reward=scalar,known_lineage_policy_transitions=s['policy_transitions_total']+json.loads((parents[arm]/'status.json').read_text())['policy_transitions_total'],charged_lineage_policy_transitions_upper=s['policy_transitions_total']+json.loads((parents[arm]/'status.json').read_text())['policy_transitions_total']+65536,unrecoverable_completed_updates=8 if arm=='warm' else 5,lineage_dev_ticks=s['validation_control_ticks_total']+s['parent_consumed_validation_control_ticks'])
with (OUT/'training_metrics.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
(OUT/'training_accounting.json').write_text(json.dumps(facts,indent=2)+'\n')
fig,axes=plt.subplots(3,2,figsize=(12,10))
for arm,color in [('warm','tab:orange'),('scratch','tab:blue')]:
    ms=[x for x in rows if x['arm']==arm and x['retained']];xs=[m['update_or_sampling_slot'] for m in ms]
    for ax,key in zip(axes.flat,['mean_step_reward','value_loss','surrogate_loss','mean_kl','sample_seconds','optimize_seconds']):
        ax.plot(xs,[m[key] for m in ms],label=arm,color=color,lw=.8);ax.set(xlabel='Model update (warm) / charged slot (scratch)',ylabel=key);ax.grid(alpha=.2);ax.legend()
axes[0,0].set_title('Different alpha curricula: reward levels are not a controlled capability ranking')
fig.tight_layout();fig.savefig(OUT/'TRAINING.png',dpi=140);plt.close(fig)
print(json.dumps(facts,indent=2))
