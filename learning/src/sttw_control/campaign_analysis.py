"""Aggregate matched physical outcomes; never compare cross-reward returns."""
import json
from pathlib import Path
import numpy as np
from .media import state_series


def summarize(plan_path):
    plan=json.loads(Path(plan_path).read_text());root=Path(plan['output']);repo=Path(plan['repository']);records=[]
    sources=[(a['name'],root/a['name']) for a in plan['arms']]
    sources += [(a['reference'],repo/'runs'/a['reference']) for a in plan['arms'] if a.get('reference')]
    seen=set()
    for name,run in sources:
        if name in seen:continue
        seen.add(name)
        for file in sorted((run/'evaluation').rglob('results.jsonl')):
            for row in map(json.loads,file.read_text().splitlines()):
                path=file.parent/row['scenario']/row['policy'];d=json.loads((path/'declaration.json').read_text());tr=dict(np.load(path/'trace.npz'));series=state_series(tr,d['config']);error=series['speed'][1:]-series['reference'][1:,1]
                track=row['summary'].get('path_tracking',row['summary'].get('circle_tracking',{}));pr=row['path_recovery']
                records.append(dict(run=name,scenario=row['scenario'],seed=row['summary']['seed'],alpha=d['config'].get('priority') and float(tr['priority_alpha'][0]),policy=row['policy'],failed=row['summary']['physical_failure'],speed_rmse=float(np.sqrt(np.mean(error**2))),path_rmse=track.get('right_error_rmse_m',track.get('radial_rmse_m')),recovered_after_excursion=pr.get('recovered_after_excursion'),recovery_seconds=pr.get('settled_joint_hold_completion_after_event_end_seconds')))
    dest=root/'analysis';dest.mkdir(exist_ok=True)
    (dest/'comparison.json').write_text(json.dumps(records,indent=2,allow_nan=False)+'\n')
    text=['# Matched physical comparison','', 'Development seeds only. Compare within the same task, scenario and alpha. No excursion is not a recovery success. Different hyperparameters prevent attributing differences to environment count alone.','', '| Run | Scenario | Alpha | Policy | N | Failures | Speed RMSE m/s | Path RMSE m |','|---|---|---|---|---:|---:|---:|---:|']
    groups={}
    for row in records:groups.setdefault((row['run'],row['scenario'],row['alpha'],row['policy']),[]).append(row)
    for key,rows in groups.items():
        name,scenario,alpha,policy=key
        text.append(f"| {name} | {scenario} | {alpha} | {policy} | {len(rows)} | {sum(r['failed'] for r in rows)} | {np.mean([r['speed_rmse'] for r in rows]):.6f} | {np.mean([r['path_rmse'] for r in rows]):.6f} |")
    (dest/'INDEX.md').write_text('\n'.join(text)+'\n')
    return records
