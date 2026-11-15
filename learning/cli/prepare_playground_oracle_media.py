"""Stitch recorded prefix and real-snapshot continuation for faithful replay media."""
import argparse,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--map',type=int,required=True);p.add_argument('--root-variant',type=int,required=True);p.add_argument('--pairs',nargs='+',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=False,parents=True)
entries=json.loads((a.root/'departure_branches.json').read_text())['entries'];rows=[];sources=[]
for pair in a.pairs:
 up,down=map(int,pair.split(':'));e=next(x for x in entries if (x['map_id'],x['upstream'],x['downstream'])==(a.map,up,down));parent=a.root/f'map{a.map:02d}/approach{up}/evaluation';child=Path(e['output'])
 before=next(r for r in json.loads((parent/'status.json').read_text())['results'] if r['variant']['root_variant']==a.root_variant);after=next(r for r in json.loads((child/'status.json').read_text())['results'] if r['variant']['root_variant']==a.root_variant)
 prefix_path=parent/before['case_dir']/'trajectory.npz';suffix_path=child/after['case_dir']/'trajectory.npz';d=np.load(prefix_path);s=np.load(suffix_path);mask=d['trace'][:,0]<e['snapshot_time_s']-1e-6
 anchor=int(np.argmin(abs(d['trace'][:,0]-e['snapshot_time_s'])));np.testing.assert_array_equal(d['qpos'][anchor],s['qpos'][0]);np.testing.assert_array_equal(d['qvel'][anchor],s['qvel'][0])
 merged={key:np.concatenate((d[key][mask],s[key])) for key in ['qpos','qvel','trace']};np.testing.assert_allclose(np.diff(merged['trace'][:,0]),.005,rtol=0,atol=1e-5)
 name=f'up{up}_down{down}_root{a.root_variant}';folder=a.output/name;folder.mkdir();np.savez_compressed(folder/'trajectory.npz',**merged,reference_xy=s['reference_xy'])
 rows.append(dict(after,case_dir=name));sources.append(dict(case=name,prefix=str(prefix_path),suffix=str(suffix_path),snapshot=e['snapshot'],snapshot_time_s=e['snapshot_time_s'],boundary_qpos_qvel_exact=True))
(a.output/'status.json').write_text(json.dumps(dict(stage='complete',results=rows),indent=2)+'\n');(a.output/'declaration.json').write_text(json.dumps(dict(terrain_model=str(a.root/f'map{a.map:02d}/base/terrain.mjb'),role='Recorded-state media assembly only; no new physics'),indent=2)+'\n');(a.output/'stitch_manifest.json').write_text(json.dumps(sources,indent=2)+'\n');print(json.dumps(sources,indent=2))
