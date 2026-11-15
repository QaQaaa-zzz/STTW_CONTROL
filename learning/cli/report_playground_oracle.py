"""Assemble every real approach/departure branch, retaining failed prefixes."""
import argparse,csv,json
from pathlib import Path
import numpy as np
import mujoco
from sttw_control.playground_actual_clearance import obstacle_clearance
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=True,parents=True)
manifest=json.loads((a.root/'manifest.json').read_text());departures=json.loads((a.root/'departure_branches.json').read_text())['entries'];rows=[];clearance_cache={};fig,axes=plt.subplots(2,2,figsize=(15,8),layout='constrained')
for mi,map_entry in enumerate(manifest['entries']):
 mid=map_entry['map_id'];folder=a.root/f'map{mid:02d}';model=mujoco.MjModel.from_binary_path(str(folder/'base/terrain.mjb'))
 for up in range(3):
  source=folder/f'approach{up}'/'evaluation';parents=json.loads((source/'status.json').read_text())['results']
  for ri,parent in enumerate(parents):
   root=parent['variant']['root_variant'];ax=axes[mi,ri];prefix=np.load(source/parent['case_dir']/'trajectory.npz')
   for down in range(3):
    entry=next((e for e in departures if e['map_id']==mid and e['upstream']==up and e['downstream']==down),None)
    if entry is None:
     prefix_end=len(prefix['qpos'])
     actual=parent;xy=prefix['qpos'][:,:2];reference=prefix['reference_xy'];phase='approach_or_jump';suffix_path=None
    else:
     target=Path(entry['output']);actual=next(r for r in json.loads((target/'status.json').read_text())['results'] if r['variant']['root_variant']==root)
     suffix=np.load(target/actual['case_dir']/'trajectory.npz');reference=suffix['reference_xy'];before=prefix['trace'][:,0]<entry['snapshot_time_s']-1e-6;prefix_end=int(before.sum());xy=np.concatenate((prefix['qpos'][before,:2],suffix['qpos'][:,:2]));phase='complete' if actual['goal'] else 'departure';suffix_path=str(target/actual['case_dir']/'trajectory.npz')
    prefix_path=str(source/parent['case_dir']/'trajectory.npz');key=(prefix_path,prefix_end)
    if key not in clearance_cache:clearance_cache[key]=obstacle_clearance(model,prefix['qpos'][:prefix_end],prefix['trace'][:prefix_end,0])['minimum']
    witnesses=[clearance_cache[key]]
    if suffix_path:
     if suffix_path not in clearance_cache:clearance_cache[suffix_path]=obstacle_clearance(model,suffix['qpos'],suffix['trace'][:,0])['minimum']
     witnesses.append(clearance_cache[suffix_path])
    witness=min((w for w in witnesses if w is not None),key=lambda w:w['distance_m'],default=None)
    rows.append(dict(map_id=mid,root=root,upstream=up,downstream=down,approach_handoff=parent['handoff_s'] is not None,success=actual['goal'],failure_stage=phase,termination=actual['termination'],duration_s=actual['duration_s'],prefix_trajectory=str(source/parent['case_dir']/'trajectory.npz'),suffix_trajectory=suffix_path or '',min_obstacle_distance_m=witness['distance_m'] if witness else None,clearance_time_s=witness['time_s'] if witness else None,clearance_geometry=witness['geometry'] if witness else None,clearance_obstacle=witness['obstacle'] if witness else None,clearance_status='native geometry at recorded 5ms poses; obstacle-only, not continuous bound'))
    line,=ax.plot(xy[:,0],xy[:,1],alpha=.7,label=f'{up}→{down}: {"OK" if actual["goal"] else actual["termination"]}');ax.plot(reference[:,0],reference[:,1],':',color=line.get_color(),alpha=.3);ax.plot(*xy[-1],marker='o' if actual['goal'] else 'x',ms=3)
   ax.set(title=f'DEV map{mid:02d} root{root}',xlabel='World x (m), travel →',ylabel='World y (m)');ax.set_aspect('equal',adjustable='datalim')
 for ri in range(2):
  ax=axes[mi,ri];world=json.loads(Path(json.loads(Path(map_entry['config']).read_text())['world_map']).read_text());ax.axvspan(10.3,10.6,color='gray',alpha=.2)
  for x0,x1,y0,y1 in world['terrain']['forbidden_rectangles']:ax.add_patch(plt.Rectangle((x0,y0),x1-x0,y1-y0,color='red',alpha=.4))
  ax.legend(fontsize=6,ncol=2)
fig.savefig(a.output/'oracle_xy.png',dpi=160)
with (a.output/'oracle_matrix.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
mechanisms=[]
for mid in manifest['maps']:
 for root in manifest['roots']:
  admitted=[up for up in range(3) if any(r['approach_handoff'] for r in rows if (r['map_id'],r['root'],r['upstream'])==(mid,root,up))]
  viable={up:any(r['success'] for r in rows if (r['map_id'],r['root'],r['upstream'])==(mid,root,up)) for up in admitted}
  if any(viable.values()) and not all(viable.values()):mechanisms.append(dict(map_id=mid,root=root,continuation_exists=viable))
(a.output/'summary.json').write_text(json.dumps(dict(cells=len(rows),success=sum(r['success'] for r in rows),mechanism_candidates=mechanisms,scope='Known-map DEV oracle only; mechanism candidates require repeat confirmation and paired videos; sampled native obstacle distances included'),indent=2)+'\n');print(json.dumps(dict(cells=len(rows),mechanisms=mechanisms)))
