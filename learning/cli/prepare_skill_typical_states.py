"""Freeze route/speed-conditioned typical interfaces using TRAIN trajectories only."""
import argparse,json
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--history-contract',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
rows=json.loads(a.history_contract.read_text())['rows'];groups={}
for row in rows:
    if row['split']!='TRAIN':continue
    source=Path(row['source']);z=np.load(source)
    record=next(r for r in json.loads((source.parent.parent/'status.json').read_text())['results'] if r['case_dir']==source.parent.name)
    key=(record['route']//3,round(float(z['trace'][0,18]),3))
    for phase in (1,2):
        t=int(np.flatnonzero(z['trace'][:,-1]==phase)[0])
        groups.setdefault((*key,phase),[]).append(dict(source=str(source),tick=t,qpos=z['qpos'][t],qvel=z['qvel'][t],previous_qpos=z['qpos'][max(0,t-4)],previous_qvel=z['qvel'][max(0,t-4)],history=np.r_[z['controller_state'][t],z['rb_state'][t]]))
arrays={};manifest=[]
for (up,speed,phase),items in sorted(groups.items()):
    features=np.array([np.r_[r['qpos'][:3],r['qvel']] for r in items])
    scale=features.std(0).clip(.1)
    index=int(np.square((features-np.median(features,axis=0))/scale).mean(1).argmin())
    selected=items[index];key=f'up{up}_v{speed:.3f}_phase{phase}'
    for name in ['qpos','qvel','previous_qpos','previous_qvel','history']:arrays[key+'_'+name]=selected[name]
    manifest.append(dict(key=key,upstream=up,speed=speed,phase=phase,candidates=len(items),source=selected['source'],tick=selected['tick']))
np.savez_compressed(a.output/'states.npz',**arrays)
(a.output/'manifest.json').write_text(json.dumps(dict(role='TRAIN only; independent typical state per upstream/speed/phase',selection='Nearest observed state to coordinatewise median of xyz and full qvel, normalized TRAIN group std floor0.1; keeps coherent history and quaternion',source_contract=str(a.history_contract),states=manifest),indent=2)+'\n')
print(f'{len(manifest)} typical interfaces from TRAIN')
