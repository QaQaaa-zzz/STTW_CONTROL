"""Shortlist stored Phase U policies using their own subsequent sampled reward.

Upstream periodic model_i is saved AFTER update i: its policy is scored by
rollout i+1. Final model_N is the N-updated policy and may be unscored.
"""
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--top',type=int,default=3);a=p.parse_args()
status=json.loads((a.run/'status.json').read_text());scores={r['update']:r for r in (json.loads(line) for line in (a.run/'training_rewards.jsonl').read_text().splitlines() if line.strip())}
best=json.loads((a.run/'best_step_model.json').read_text());completed=status['completed_updates'];rows=[]
for path in sorted((a.run/'training').glob('model_*.pt')):
    suffix=path.stem.removeprefix('model_')
    if suffix=='step_best':update=best['update']
    elif suffix=='initial':update=status.get('start_update',0)
    elif suffix.startswith('update_'):update=int(suffix.removeprefix('update_'))
    elif suffix.isdigit():update=int(suffix) if status['stage']=='training_complete' and int(suffix)==completed else int(suffix)+1
    else:continue
    row=dict(checkpoint=str(path.resolve()),policy_update=update,score=scores.get(update),role='stored candidate; sampled training reward is not fixed-task performance')
    rows.append(row)
ranked=sorted((r for r in rows if r['score'] is not None),key=lambda r:-r['score']['mean_step_reward'])
chosen=[];updates=set()
for row in ranked:
    if row['policy_update'] not in updates:
        chosen.append(row);updates.add(row['policy_update'])
    if len(chosen)>=a.top:break
if status['stage']=='training_complete':
    final=next(r for r in rows if Path(r['checkpoint']).stem==f'model_{completed}')
    if final['policy_update'] not in updates:chosen.append(final)
result=dict(stage=status['stage'],completed_updates=completed,selection='up to top distinct stored policies by their own sampled reward, plus final after completion',scope='shortlist only; freeze then evaluate all on the same handoff cases',all_stored=rows,candidates=chosen)
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(stage=status['stage'],candidate_updates=[r['policy_update'] for r in chosen]),indent=2))
