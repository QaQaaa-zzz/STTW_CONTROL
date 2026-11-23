"""Freeze and execute the four-map-batch, seven-branch P0-B development experiment."""
import argparse
import csv
import hashlib
import json
from itertools import product
from pathlib import Path

import numpy as np


WIDE = Path('/media/qy/6A7E3DB07E3D75C3/STTW_runs/chapter3_playground_sttw_rb_sttw/'
            'platform_test0_20261120/approach_alignment/replication_wide')
FIELDS = ('xy', 'heading', 'curvature')
ARMS = ('left', 'straight', 'right', 'near_left', 'near_right', 'far_left', 'far_right')
MUJOCO_VERSION = '3.6.0'
GEOMETRY = ('baseline_right', 'signed_left', 'signed_right', 'baseline_mirror', 'half_left', 'half_right')
OLD = ((0, 0.), (1, 0.), (2, 0.), (1, .5), (1, 1.))


def read(path):
    return json.loads(path.read_text())


def save(path, values):
    path.write_text(json.dumps(values, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def digest(path):
    with path.open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def routes(old, trigger, front):
    """Three immutable originals. Express new geometry in physical x before arc reparameterization."""
    arc = old['s']
    geometries = [[old[k][i] for k in FIELDS] for i in range(3)]
    transformations = ((0, -1.), (1, .5), (2, .5))
    for source, scale in transformations:
        x, y = old['xy'][source].T
        # Impose exactly half the tangent angle at the trigger of each half-amplitude route.
        slope = np.tan(np.interp(trigger, x, old['heading'][source]))
        desired = -slope if scale < 0 else np.tan(.5 * np.arctan(slope))
        delta = x - trigger
        new_y = scale * y + (desired - scale * slope) * delta * np.exp(-.5 * (delta / .65)**2)
        # Match the donor reference again by the platform edge, while frozen RB owns control.
        # Otherwise a "new approach" would also silently change the post-lock default exit.
        u = ((x-trigger)/(front-trigger)).clip(0.,1.)
        blend = u**3*(10.-15.*u+6.*u**2)
        new_y = (1.-blend)*new_y + blend*y
        distance = arc[0] + np.r_[0., np.cumsum(np.hypot(np.diff(x), np.diff(new_y)))]
        xx = np.interp(arc, distance, x) + (arc - distance[-1]).clip(0)
        xy = np.column_stack((xx, np.interp(arc, distance, new_y)))
        hd = np.arctan2(np.gradient(xy[:, 1]), np.gradient(xx))
        geometries.append((xy, hd, np.gradient(hd, arc)))
    result = dict(s=arc, **dict(zip(FIELDS, map(np.stack, zip(*geometries)))))
    for k in FIELDS:
        np.testing.assert_array_equal(result[k][:3], old[k])
    return result


def make_bank(out, wide, ref, goal_x):
    from run_platform_task_oracle import departure_route
    n = len(ref['s'])
    with np.load(wide/'primary/departure_bank.npz') as old:
        bank = {k: np.empty((6 + 18*n, *ref[k].shape[1:]), np.float32) for k in FIELDS}
        for k in FIELDS:
            bank[k][:6] = ref[k]
            np.testing.assert_array_equal(old[k][:3], bank[k][:3])
            bank[k][6:6+9*n] = old[k][3:]
    for r, direction, join in product(range(3, 6), range(3), range(n)):
        one = dict(s=ref['s'], **{k:ref[k][r:r+1].copy() for k in FIELDS})
        departure_route(one, 0, join, (.36, 0., -.36)[direction], 1.6, 15., goal_x)
        idx = 6 + (r*3 + direction)*n + join
        for k in FIELDS:
            bank[k][idx] = one[k][0]
    np.savez(out/'departures.npz', s=ref['s'], **bank)


def freeze(out, wide):
    parent = wide/'chapter_mainline/post_review'
    pf = parent/'realtime_15ms/test_freeze_manifest.json'
    freeze = read(pf)
    source_cfg = wide/'primary/roots_0008/rb_run1.json'
    original = read(source_cfg)
    historic = read(parent/'wall_layouts/test_freeze_manifest.json')['maps']
    paths = [Path(freeze['bank']), wide/'initial_bank.npz']
    with np.load(paths[0]) as data, np.load(paths[1]) as earlier:
        states = np.column_stack((data['qpos'], data['qvel']))
        prior = {row.tobytes() for row in np.column_stack((earlier['qpos'], earlier['qvel']))}
        prior.update(row.tobytes() for row in states[:1171])
        fresh = {row.tobytes() for row in states[1171:1195]}
        if len(fresh) != 24 or prior & fresh:
            raise ValueError('P0-B roots overlap archived starts or each other')
    layouts = {}
    for j, (old_key, seed) in enumerate((('config_r', 62211431), ('config_double', 62211437))):
        base = Path(historic[old_key]['path'])
        world = read(base)
        key = ('right_offset', 'staggered')[j]
        world['map_id'] = f'p0b_{key}_{seed}'
        terrain = world['terrain']
        terrain['rough_source']['seed'] = seed
        # This is new terrain and obstacle geometry within the declared family.
        walls = np.array(terrain['forbidden_rectangles']) + np.array([.04, .04, -.012, -.012])
        terrain['forbidden_rectangles'] = walls.tolist()
        terrain['rough_source']['forbidden_rectangles'] = walls.tolist()
        loc = out/key
        loc.mkdir(exist_ok=True)
        save(loc/'world.json', world)
        layouts[key] = dict(path=str(loc/'world.json'), source=str(base), walls=walls.tolist(), seed=seed,
                            roots=list(range(1171+12*j, 1183+12*j)), sha256=digest(loc/'world.json'))
    trigger = terrain['front_x_m'] + original['planned_activation_offset_m']
    with np.load(wide/'approaches.npz') as old:
        ref = routes(old, trigger, terrain['front_x_m'])
    np.savez(out/'approaches.npz', **ref)
    with np.load(wide/'common_start.npz') as old:
        neutral = {k: np.repeat(old[k][:1], 6, axis=0) for k in FIELDS}
        np.savez(out/'neutral.npz', s=old['s'], **neutral)
    make_bank(out, wide, ref, world['goal']['x_min_m'])
    initializer = parent/'realtime_15ms/test_a/rb_run1_initializer.npz'
    actions = list(product(range(6), (0., .5, 1.)))
    batches = []
    for layout, info in layouts.items():
        for part in range(2):
            name = f'{layout}/part_{part}'
            folder = out/name
            folder.mkdir(exist_ok=True)
            opts, lanes, directions = [], [], []
            for root, force, (geometry, alpha) in product(info['roots'][6*part:6*(part+1)], (0, -5, 5), actions):
                for d, arm in enumerate(ARMS):
                    section = 'far' if arm.startswith('far') else 'near'
                    side = -1 if arm.endswith('_right') else 1
                    old_direction = d if d<3 else 1
                    lanes.append([geometry, 'h75', len(opts)])
                    directions.append(old_direction)
                    opts.append(dict(original['ground_handoff']['handoff_variants'][0], bank_row=root,
                        root_variant=root, data_split='DEV', planner='oracle', arm=arm, bin=section,
                        side=side, section_x=15. if section=='far' else 12.7, force_y_N=force,
                        planned_departure=old_direction, departure=ARMS[old_direction],
                        approach_alpha=alpha, approach_speed_m_s=2., geometry=geometry))
            cfg = dict(original, world_map=info['path'], evaluation_lanes=lanes, ccd_capacity=8*len(opts),
                       role='P0-B known-map paired physical development, seven concurrent branch replicas',
                       approach_initial_bank=freeze['bank'], collect_model_state=False,
                       ground_handoff=dict(original['ground_handoff'], reference_path=str(out/'neutral.npz'),
                                           handoff_variants=opts),
                       paired_departures=dict(approach_file=str(out/'approaches.npz'),
                            route_bank=str(out/'departures.npz'), direction_ids=directions),
                       platform_online=dict(ground_model=freeze['ground_model'], margin_m=freeze['margin']['rb_run1'],
                            latency_ticks=freeze['latency_ticks'], relative=dict(height=.16,
                            raster=str(out/layout/'terrain.npz'), initializer=str(initializer))),
                       p0_branches=dict(group_size=7, schedule_ticks=20, deadline_ms=100, roots=info['roots'],
                                        apex_forcing=True))
            save(folder/'config.json', cfg)
            batches.append(dict(batch=name, worlds=len(lanes), roots=info['roots'][6*part:6*(part+1)],
                                config_sha256=digest(folder/'config.json')))
    save(out/'protocol.json', dict(role='fallible development experiment, no training', layouts=layouts,
         source_config=str(source_cfg), parent_manifest=str(pf), ground_model=freeze['ground_model'],
         frozen_weights=freeze['weights'], bank=freeze['bank'], geometry=list(GEOMETRY), old_actions=OLD,
         actions=actions, trigger_m=trigger, arms=list(ARMS), batches=batches, updates=0,
         approach_intervention='Slope and offset at trigger; smooth return to donor geometry by native step front. Existing three curves copied verbatim.',
         preflight_revision=2, mujoco_version=MUJOCO_VERSION,
         expiration='On first tick after a published CEM forecast expires: retain the consumed/preview prefix and blend over 1.6 m to the common straight exit from handback, with no layout ID. Keep an unexpired incumbent upon rejection. Original three fixed probes execute their original full-horizon contracts.',
         controller_receipts='Separate route assignment, per-lane H75 actor input/actual issued wheel and steer targets, and pursuit refresh. Record observation-to-consumption motion; target values alone do not attribute causal changes.',
         unique_prefixes=1296, physical_replicas_per_prefix=7, trajectory_lanes=9072,
         active_suffix_max=9072, horizon_s=10, dt_s=.001, control_dt_s=.005, max_lane_substeps=90720000,
         known_map=True, physical_release='three old measured branches after charged 15 ms; CEM uses measured latency replay and 100 ms deadline',
         frozen_files={str(p):digest(p) for p in [pf, source_cfg, initializer, paths[0],
                 out/'approaches.npz', out/'departures.npz', out/'neutral.npz']}))


def run_batch(out, name, latency_file):
    import fcntl
    import subprocess
    import sys
    import time
    from types import SimpleNamespace
    import mujoco
    from evaluate_playground_handoff import transplant
    from predict_platform_departures import physical_raster
    plan = read(out/'protocol.json')
    if mujoco.__version__ != plan['mujoco_version']:
        raise ValueError(f'P0-B requires the frozen MuJoCo {MUJOCO_VERSION} runtime')
    batch = next(b for b in plan['batches'] if b['batch']==name)
    loc = out/name
    if digest(loc/'config.json')!=batch['config_sha256']:
        raise ValueError('Physical config differs from frozen preparation')
    cfg = read(loc/'config.json')
    contract = cfg['p0_branches']
    with latency_file.open() as stream:
        timed = sorted((r for r in csv.DictReader(stream) if int(r['candidates'])==contract.get('candidates',16)
                        and int(r['iterations'])==2), key=lambda r:int(r['repetition']))
    latencies = [float(r['obs_to_publish_ms']) for r in timed]
    if (len({r['repetition'] for r in timed})!=len(timed) or len(timed)<100
            or any(float(r['deadline_ms'])!=contract['deadline_ms'] for r in timed)
            or not np.isfinite(latencies).all() or min(latencies)<=0):
        raise ValueError('Expected 100+ unique measured single-robot latencies under the frozen budget')
    contract['latency_ms'] = latencies
    if (loc/'physical/status.json').is_file() and read(loc/'physical/status.json')['stage']=='complete':
        return
    layout = loc.parent
    with Path('/home/qy/STTW_CONTROL/research-hub/.gpu-job.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not (layout/'terrain.npz').is_file():
            xml = read(Path('learning/configs/playground_rb_skill_sources.json'))['model']['path']
            transplant(SimpleNamespace(xml_path=Path(xml)), read(layout/'world.json')['terrain'],
                       layout, 5, cfg['ground_physics'])
            physical_raster(layout/'terrain.mjb', layout/'terrain.npz')
        built = out/'terrain_build_receipt.json'
        if built.exists():
            record = read(built)['maps'][layout.name]
            if (record['compilation_version']!=MUJOCO_VERSION or record['world_sha256']!=digest(layout/'world.json')
                    or record['mjb_sha256']!=digest(layout/'terrain.mjb') or record['raster_sha256']!=digest(layout/'terrain.npz')):
                raise ValueError('Frozen terrain/raster receipt does not match the selected map')
        cfg['p0_branches']['timing_file'] = str(latency_file)
        cfg['p0_branches']['timing_sha256'] = digest(latency_file)
        save(loc/'run.json', cfg)
        start = time.monotonic()
        cmd = [sys.executable,'learning/cli/evaluate_playground_handoff.py','--config',str(loc/'run.json'),
               '--output',str(loc/'physical'),'--skills','rb_run1','--modes','h75','--routes','0','1',
               '--physics-substeps','5','--seconds','10']
        with (loc/'physics.log').open('a') as log:
            proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
        save(loc/'exit.json',dict(returncode=proc.returncode, seconds=time.monotonic()-start))
        proc.check_returncode()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('output', type=Path)
    p.add_argument('--wide', type=Path, default=WIDE)
    modes = p.add_mutually_exclusive_group(required=True)
    modes.add_argument('--prepare-only', action='store_true')
    modes.add_argument('--batch', choices=('right_offset/part_0','right_offset/part_1',
                                           'staggered/part_0','staggered/part_1','all'))
    p.add_argument('--timing', type=Path)
    args = p.parse_args()
    out = args.output.resolve()
    if args.prepare_only:
        out.mkdir(parents=True, exist_ok=True)
        if (out/'protocol.json').exists():
            raise FileExistsError(out/'protocol.json')
        freeze(out, args.wide.resolve())
    else:
        if not args.timing or not args.timing.is_file():
            p.error('--batch requires a measured single-robot --timing file')
        names = [b['batch'] for b in read(out/'protocol.json')['batches']] if args.batch=='all' else [args.batch]
        for name in names:
            run_batch(out, name, args.timing.resolve())


if __name__=='__main__':
    main()
