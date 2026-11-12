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
p.add_argument('--terrain', choices=('slope', 'rough_slope', 'bend_bump', 'bend_ramp', 'route_rough', 'route_waves'), required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
if a.terrain in ('bend_bump', 'bend_ramp', 'route_rough', 'route_waves'):
    cfg = json.loads(a.scene_config.read_text())
    route_path = Path(cfg['route'])
    route = np.load(route_path)
    spacing = cfg['horizontal_scale_m']
    axis = np.arange(cfg['bounds_m'][0], cfg['bounds_m'][1] + spacing / 2, spacing)
    x, y = np.meshgrid(axis, axis, indexing='ij')
    points = np.stack((x.ravel(), y.ravel()), axis=-1)
    if cfg.get('projection_cache'):
        projection=np.load(cfg['projection_cache']);s,n=projection['s'],projection['n']
    else:
        # Project onto the fixed route; the terrain never follows the vehicle.
        ids = np.flatnonzero(route['s'] <= float(route['turn_end']) + (6 if a.terrain.startswith('route_') else 3))
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
    if a.terrain.startswith('route_'):
        np.random.seed(cfg['seed'])
        tile = terrain_utils.SubTerrain(a.terrain, width=len(axis), length=len(axis),
                                       vertical_scale=cfg['vertical_scale_m'], horizontal_scale=spacing)
        if a.terrain == 'route_rough':
            terrain_utils.random_uniform_terrain(tile, min_height=-cfg['height_m'],
                max_height=cfg['height_m'], step=cfg['vertical_scale_m'],
                downsampled_scale=cfg['downsampled_scale_m'])
        else:
            terrain_utils.wave_terrain(tile, num_waves=cfg['num_waves'], amplitude=cfg['height_m'])
        # Terrain is fixed in world coordinates and spans the whole lateral domain.
        # Both ends fade smoothly; there is no narrow strip to bypass.
        entry = np.clip((s-cfg['start_s_m'])/cfg['blend_length_m'], 0, 1)
        leave = np.clip((cfg['end_s_m']-s)/cfg['blend_length_m'], 0, 1)
        envelope = .5*(1-np.cos(np.pi*entry)) * .5*(1-np.cos(np.pi*leave))
        surface = tile.height_field_raw.ravel()*cfg['vertical_scale_m']*envelope
    elif a.terrain == 'bend_ramp':
        u = np.clip((s - cfg['slope_start_s_m']) / cfg['length_m'], 0, 1)
        profile = .5 * (1 - np.cos(np.pi * u))
        surface = cfg['height_m'] * profile
    else:
        u = (s - cfg['center_s_m'] - cfg['oblique_s_per_lateral_m'] * n) / cfg['length_m'] + .5
        width = np.clip(1 - np.abs(n) / cfg['half_width_m'], 0, 1)
        profile = np.where((u > 0) & (u < 1), np.sin(np.pi * u)**2, 0)
        surface = cfg['height_m'] * profile * np.sin(np.pi * width / 2)**2
    height = np.rint(surface / cfg['vertical_scale_m']).reshape(x.shape).astype(np.int16)
    vertices, triangles = terrain_utils.convert_heightfield_to_trimesh(height, spacing, cfg['vertical_scale_m'], cfg['slope_threshold'])
    vertices[:, :2] += cfg['bounds_m'][0]
    a.output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(a.output / 'terrain.npz', vertices=vertices, triangles=triangles, height_field=height)
    meta = dict(cfg, route_sha256=hashlib.sha256(route_path.read_bytes()).hexdigest(),
                turn_end_s_m=float(route['turn_end']),
                switch_definition='first forward crossing of switch_s_m; preserve controller/history/filter state',
                contact_phase_definition='wheel contact points inside nonzero height support; geometric progress alone is not contact',
                vertices=len(vertices), triangles=len(triangles),
                sha256=hashlib.sha256((a.output / 'terrain.npz').read_bytes()).hexdigest())
    if a.terrain == 'bend_ramp':
        meta.update(profile='raised cosine ascent to persistent plateau; height independent of lateral offset',
                    slope_end_s_m=cfg['slope_start_s_m']+cfg['length_m'],
                    nominal_peak_grade=cfg['height_m']*np.pi/(2*cfg['length_m']))
    elif a.terrain.startswith('route_'):
        meta.update(terrain_support_end_s_m=cfg['end_s_m'], source_module=terrain_utils.__file__,
                    height_min_m=float(height.min()*cfg['vertical_scale_m']),
                    height_max_m=float(height.max()*cfg['vertical_scale_m']),
                    profile='SDK world-fixed surface, full lateral coverage, raised-cosine route entry and exit')
        if a.terrain == 'route_waves':
            meta['world_axis_wavelength_m'] = len(axis)*spacing/cfg['num_waves']
    else:
        meta['terrain_support_end_s_m'] = cfg['center_s_m'] + cfg['length_m']/2 + abs(cfg['oblique_s_per_lateral_m'])*cfg['half_width_m']
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
