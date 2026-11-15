"""Prepare finite route branches from real, complete initial or handoff snapshots."""
import argparse,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--stage',choices=['approach','departure'],required=True);a=p.parse_args();manifest=json.loads((a.root/'manifest.json').read_text());entries=[];failed=[]
for m in manifest['entries']:
 folder=a.root/f"map{m['map_id']:02d}";base=json.loads(Path(m['config']).read_text());original=dict(np.load(base['ground_handoff']['reference_path']))
 for upstream in range(3):
  if a.stage=='approach':sources=[(0,folder/'base'/'rb_run3_initial_snapshot.pkl',0.)]
  else:
   source=folder/f'approach{upstream}'/'evaluation';anchors=source/'snapshot_anchors.json';ticks=json.loads(anchors.read_text())['ticks']
   results=json.loads((source/'status.json').read_text())['results']
   if not any(r['handoff_s'] is not None for r in results):ticks=[]
   if not ticks:
    failed.append(dict(map_id=m['map_id'],upstream=upstream,reason='no complete ground handoff snapshot',status=str(source/'status.json')));continue
   tick=ticks[0];sources=[(downstream,source/f'rb_run3_anchor_{tick:05d}.pkl',tick*.005) for downstream in range(3)]
  for downstream,snapshot,t in sources:
   candidate=3*upstream+downstream;out=folder/(f'approach{upstream}' if a.stage=='approach' else f'departure{upstream}_{downstream}');out.mkdir(exist_ok=False);bank={k:v.copy() for k,v in original.items()}
   for field in ['xy','heading','curvature']:bank[field][0]=original[field][candidate]
   np.savez_compressed(out/'routes.npz',**bank)
   cfg=json.loads(json.dumps(base));cfg.update(initial_snapshot_path=str(snapshot),snapshot_ground_handoff=a.stage=='approach',oracle_branch_reference=str(out/'routes.npz'))
   cfg['ground_handoff']['reference_path']=str(out/'routes.npz');cfg['role']=f'P2 DEV true-snapshot {a.stage} branch; candidate{candidate}; no oracle use in online decisions'
   (out/'config.json').write_text(json.dumps(cfg,indent=2)+'\n');entries.append(dict(map_id=m['map_id'],upstream=upstream,downstream=downstream,candidate=candidate,snapshot=str(snapshot),snapshot_time_s=t,config=str(out/'config.json'),output=str(out/'evaluation')))
(a.root/f'{a.stage}_branches.json').write_text(json.dumps(dict(entries=entries,failed_prefixes=failed),indent=2)+'\n');commands=['#!/usr/bin/env bash','set -e']
for e in entries:commands.append(f"/home/qy/mujoco_playground/.venv/bin/python learning/cli/evaluate_playground_handoff.py --config {e['config']} --output {e['output']} --skills rb_run3 --modes h75 --routes 0 --seconds 10 --physics-substeps 5")
(a.root/f'{a.stage}.sh').write_text('\n'.join(commands)+'\n');print(json.dumps(dict(branches=len(entries),failed_prefixes=len(failed))))
