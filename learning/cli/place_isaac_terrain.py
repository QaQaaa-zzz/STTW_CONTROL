"""Place an SDK tile after a flat approach, extending its exit without a cliff."""
import argparse
import hashlib
import json
from pathlib import Path
from isaacgym import terrain_utils
import numpy as np

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--library', type=Path, required=True)
p.add_argument('--terrain', choices=('slope', 'rough_slope'), required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
identity = json.loads((a.library / 'identity.json').read_text())
source = a.library / identity['terrains'][a.terrain]['file']
if hashlib.sha256(source.read_bytes()).hexdigest() != identity['terrains'][a.terrain]['sha256']:
    raise ValueError('source terrain identity mismatch')
a.output.mkdir(parents=True, exist_ok=False)
raw = np.load(source)['height_field']
spacing = identity['horizontal_scale_m']
axis = np.arange(-20, 141) * spacing
# World x=2 m starts the source tile. Clamp its far edges to avoid artificial cliffs.
ix = np.clip(np.rint((axis - 2) / spacing).astype(int), 0, raw.shape[0] - 1)
iy = np.clip(np.rint((axis + 2) / spacing).astype(int), 0, raw.shape[1] - 1)
height = raw[ix[:, None], iy[None, :]].copy()
# Ramp in only the first 0.5 m; the spawn and approach are exactly level.
blend = np.clip((axis - 2) / .5, 0, 1)
height = np.rint(height * blend[:, None]).astype(np.int16)
vertices, triangles = terrain_utils.convert_heightfield_to_trimesh(
    height, spacing, identity['vertical_scale_m'], identity['slope_threshold'])
vertices[:, :2] -= 2
np.savez_compressed(a.output / 'terrain.npz', vertices=vertices, triangles=triangles,
                    height_field=height)
meta = dict(source=str(source.resolve()), source_sha256=identity['terrains'][a.terrain]['sha256'],
            adapter='flat x<=2; 0.5m entry blend; source edge height extended at exit and lateral edges',
            bounds_m=[-2, 14], source_origin_m=[2, -2], horizontal_scale_m=spacing,
            vertical_scale_m=identity['vertical_scale_m'], vertices=len(vertices), triangles=len(triangles),
            sha256=hashlib.sha256((a.output / 'terrain.npz').read_bytes()).hexdigest())
(a.output / 'identity.json').write_text(json.dumps(meta, indent=2))
print(json.dumps(meta))
