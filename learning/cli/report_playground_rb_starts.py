"""Report the declared activation sweep without changing its success events."""
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('root',type=Path)
a=p.parse_args()
manifest=json.loads((a.root/'manifest.json').read_text())
rows=[];groups=[]
for entry in manifest['entries']:
    evaluation=Path(entry['evaluation'])
    status=json.loads((evaluation/'status.json').read_text())
    config=json.loads((evaluation/'resolved_config.json').read_text())
    activation=config['ground_handoff']['approach']['jump_start_x_m']
    anchors=json.loads((evaluation/'snapshot_anchors.json').read_text())
    for result in status['results']:
        with np.load(evaluation/result['case_dir']/'trajectory.npz') as data:
            trace=data['trace'];phase=trace[:,-1];valid=data['applied_target_valid']
            drift=[]
            for stage in [0,2]:
                q=data['qpos'][phase==stage,10:12]
                if len(q):drift.append(float(np.abs(q-q[0]).max()))
            row=dict(activation_id=entry['layout'],activation_x_m=activation,case=result['case_dir'],
                     root=result['variant']['root_variant'],route=result['route'],
                     termination=result['termination'],goal=result['goal'],
                     rb_entry_s=result['rb_entry_s'],handoff_s=result['handoff_s'],
                     duration_s=result['duration_s'],final_xy=result['final_xy'],
                     rb_complete=result['handoff_s'] is not None,
                     max_locked_joint_drift_rad=max(drift),
                     actual_target_samples=int(valid.sum()),
                     finite_applied_targets=bool(np.isfinite(data['applied_targets'][valid]).all()),
                     rb_history_width=data['rb_state'].shape[1],
                     terminal_target_invalid=bool(not valid[-1]),
                     anchor_before_entry=bool(result['rb_entry_s'] is not None and max(anchors['ticks'])*.005<result['rb_entry_s']),
                     source=str(evaluation/result['case_dir']/'trajectory.npz'))
            rows.append(row)
    selected=[r for r in rows if r['activation_id']==entry['layout']]
    groups.append(dict(activation_x_m=activation,count=len(selected),goals=sum(r['goal'] for r in selected),
                       rb_complete=sum(r['rb_complete'] for r in selected),
                       outcomes=dict(Counter(r['termination'] for r in selected)),
                       max_locked_joint_drift_rad=max(r['max_locked_joint_drift_rad'] for r in selected),
                       evaluation_s=status['elapsed_s']))
summary=dict(groups=groups,total=len(rows),goals=sum(r['goal'] for r in rows),
             rb_complete=sum(r['rb_complete'] for r in rows),
             runtime_records_valid=all(r['finite_applied_targets'] and r['terminal_target_invalid'] and r['rb_history_width']==156 and r['anchor_before_entry'] for r in rows),
             role='TRAIN motion-library expansion; 24 upstream/root/activation groups; no independent TEST',
             simulation_elapsed_s=sum(g['evaluation_s'] for g in groups))
(a.root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
(a.root/'episodes.json').write_text(json.dumps(rows,indent=2)+'\n')
lines=['# P2 RB activation sweep','','![All actual/reference XY](library/full_chain_xy.png)','',
       'Frozen RB225, same rough map and ditch buffers, H75 with locked hip/knee during STTW. Only activation position changes. Actual targets/FIFO are recorded on GPU and exported after rollout. Complete runtime anchor at3.5s precedes entry; it is not an entry-time reset.', '',
       '|Activation x m|Full chain goal|RB complete|Outcomes|Max lock drift rad|Evaluation s|',
       '|---:|---:|---:|---|---:|---:|']
for g in groups:
    lines.append(f"|{g['activation_x_m']:.2f}|{g['goals']}/{g['count']}|{g['rb_complete']}/{g['count']}|{g['outcomes']}|{g['max_locked_joint_drift_rad']:.3g}|{g['evaluation_s']:.2f}|")
lines+=['','|Activation x|Route|Root|Outcome|Entry s|Handoff s|End s|','|---:|---:|---:|---|---|---|---:|']
for r in rows:
    lines.append(f"|{r['activation_x_m']:.2f}|{r['route']}|{r['root']}|{r['termination']}|{r['rb_entry_s']}|{r['handoff_s']}|{r['duration_s']:.3f}|")
lines+=['','All observed starts and failures remain in the library. Success here does not establish an interpolated legal start domain, rolling planning, or joint-planner benefit.']
(a.root/'RESULTS.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(summary,indent=2))
