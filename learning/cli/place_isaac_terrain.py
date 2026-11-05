"""Place an SDK tile after a flat approach, extending its exit without a cliff."""
import argparse
import hashlib
import json
from pathlib import Path
from isaacgym import terrain_utils
import numpy as np

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--library', type=Path)
p.add_argument('--scene-config', type=Path)
p.add_argument('--terrain', choices=('slope', 'rough_slope', 'bend_bump'), required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
if a.terrain == 'bend_bump':
    cfg = json.loads(a.scene_config.read_text())
    route_path = Path(cfg['route'])
    route = np.load(route_path)
    spacing = cfg['horizontal_scale_m']
    axis = np.arange(cfg['bounds_m'][0], cfg['bounds_m'][1] + spacing / 2, spacing)
    x, y = np.meshgrid(axis, axis, indexing='ij')
    points = np.stack((x.ravel(), y.ravel()), axis=-1)
    # Project onto the fixed route; the terrain never follows the vehicle.
    ids = np.flatnonzero(route['s'] <= float(route['turn_end']) + 3)
    xy = route['xy'][ids]
    start, edge = xy[:-1], np.diff(xy, axis=0)
    ss, lateral = [], []
    for chunk in np.array_split(points, 32):
        delta = chunk[:, None, :] - start
        fraction = np.clip((delta * edge).sum(-1) / (edge * edge).sum(-1), 0, 1)
        offset = delta - fraction[..., None] * edge
        nearest = (offset * offset).sum(-1).argmin(-1)
        row = np.arange(len(chunk))
        ss.append(route['s'][ids[:-1]][nearest] + fraction[row, nearest] * np.diff(route['s'][ids])[nearest])
        tangent = edge[nearest] / np.linalg.norm(edge[nearest], axis=-1)[:, None]
        lateral.append(tangent[:, 0] * offset[row, nearest, 1] - tangent[:, 1] * offset[row, nearest, 0])
    s, n = np.concatenate(ss), np.concatenate(lateral)
    u = (s - cfg['center_s_m'] - cfg['oblique_s_per_lateral_m'] * n) / cfg['length_m'] + .5
    width = np.clip(1 - np.abs(n) / cfg['half_width_m'], 0, 1)
    profile = np.where((u > 0) & (u < 1), np.sin(np.pi * u)**2, 0)
    height = np.rint(cfg['height_m'] * profile * np.sin(np.pi * width / 2)**2 / cfg['vertical_scale_m']).reshape(x.shape).astype(np.int16)
    vertices, triangles = terrain_utils.convert_heightfield_to_trimesh(height, spacing, cfg['vertical_scale_m'], cfg['slope_threshold'])
    vertices[:, :2] += cfg['bounds_m'][0]
    a.output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(a.output / 'terrain.npz', vertices=vertices, triangles=triangles, height_field=height)
    end = cfg['center_s_m'] + cfg['length_m'] / 2 + abs(cfg['oblique_s_per_lateral_m']) * cfg['half_width_m']
    meta = dict(cfg, route_sha256=hashlib.sha256(route_path.read_bytes()).hexdigest(),
                terrain_support_end_s_m=end, turn_end_s_m=float(route['turn_end']),
                switch_definition='first forward crossing of switch_s_m; preserve controller/history/filter state',
                contact_phase_definition='wheel contact points inside nonzero height support; geometric progress alone is not contact',
                vertices=len(vertices), triangles=len(triangles),
                sha256=hashlib.sha256((a.output / 'terrain.npz').read_bytes()).hexdigest())
    (a.output / 'identity.json').write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta))
    raise SystemExit
if a.library is None:
    p.error('--library is required for SDK slope placement')
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
