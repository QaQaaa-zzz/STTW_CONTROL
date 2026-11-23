"""Score the measured P0-B seven-arm branches, paired effects and finite entry ceilings."""
import argparse
import json
import os
from collections import defaultdict
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

DIRS = ('left', 'straight', 'right')
TASKS = (('near', 1), ('near', -1), ('far', 1), ('far', -1))
ENTRY = ['geometry', 'alpha']
GROUP = ['layout', 'root', 'section', 'side']
FORCES = [-5, 0, 5]
LEGACY = {(0, 0.), (1, 0.), (2, 0.), (1, .5), (1, 1.)}
CELLS = {'A': ('old', 'old_filter'), 'B': ('old', 'rolling'),
         'C': ('expanded', 'old_filter'), 'D': ('expanded', 'rolling')}


def read(path):
    return json.loads(path.read_text())


def read_events(folder):
    old, queries, pubs = {}, {}, {}
    changes, consumed = {}, {}
    with (folder / 'p0_events.jsonl').open() as stream:
        for line in stream:
            event = json.loads(line)
            for j, lane in enumerate(event['lanes']):
                row = {k: (v[j] if isinstance(v, list) else v) for k, v in event.items() if k != 'lanes'}
                row['lane'] = lane
                if row['kind'] == 'old_selection':
                    if lane in old or (not row['admitted'] and row['choice'] != 1):
                        raise ValueError(f'Invalid old selection for lane {lane}')
                    old[lane] = row
                elif row['kind'] == 'query':
                    key = (lane, row['tick'])
                    if key in queries:
                        raise ValueError(f'Duplicate query {key}')
                    queries[key] = row
                elif row['kind'] == 'publication':
                    key = (lane, row['sample_tick'])
                    if key in pubs:
                        raise ValueError(f'Duplicate publication {key}')
                    pubs[key] = row
                    if row['published']:
                        changes[lane,row['route_version']] = dict(kind='accepted_plan',sample_tick=row['sample_tick'],
                                    change_tick=row['tick'],valid_until=row['pending_valid_until'],
                                    route_id=row['route_id'],default_route_id=None,committed_s=None)
                elif row['kind'] == 'expired_to_default':
                    changes[lane,row['route_version']] = dict(kind='expired_to_default',sample_tick=None,
                                change_tick=row['tick'],valid_until=row['valid_until'],route_id=row['route_id'],
                                default_route_id=row['default_route_id'],committed_s=row['committed_s'])
                elif row['kind'] in ('pursuit_consumed', 'h75_actor_consumed'):
                    k = lane,row['version'],row['kind']
                    if k in consumed:
                        raise ValueError(f'Duplicate consumer receipt {k}')
                    consumed[k] = row
    if {(lane,ver) for lane,ver,_ in consumed} - changes.keys():
        raise ValueError('Physical clock receipt lacks original change of reference')
    route_changes = []
    for (lane,version), meta in changes.items():
        for kind,field in [('pursuit_consumed','follower_request'),('h75_actor_consumed','applied_steer_rear')]:
            receipt = consumed.get((lane,version,kind))
            if receipt and (receipt['publication_tick'] != meta['change_tick'] or receipt['tick'] < meta['change_tick']):
                raise ValueError(f'Invalid {kind} receipt clock for {(lane,version)}')
            meta[kind+'_tick'] = receipt['tick'] if receipt else None
            meta[kind+'_ms_from_change'] = (receipt['tick']-meta['change_tick'])*5 if receipt else None
            meta[kind+'_ms_from_query'] = ((receipt['tick']-meta['sample_tick'])*5
                        if receipt and meta['sample_tick'] is not None else None)
            meta[field] = receipt[field] if receipt else None
        route_changes.append(dict(lane=lane,version=version,**meta))
    if pubs.keys() - queries.keys():
        raise ValueError('Publication exists without a matching issued query')
    lane_counts = defaultdict(lambda: dict(queries=0, model_rejected=0, missed_deadline=0,
                            published=0, reached_due=0, stale_commit=0, ended_pending=0))
    events = []
    for key, query in queries.items():
        lane, tick = key
        pub = pubs.get(key)
        due, ms = query['due'], query['ms']
        charged = (due - tick)*5
        if pub is not None and (pub['tick'] != due or pub['model_admitted'] != query['admitted']):
            raise ValueError(f'Decision differs at publication: {key}')
        missed_deadline = charged > 100
        if pub is not None and pub['in_time'] == missed_deadline:
            raise ValueError(f'Publication deadline disagrees with charged latency: {key}')
        stats = lane_counts[lane]
        stats['queries'] += 1
        stats['model_rejected'] += not query['admitted']
        stats['missed_deadline'] += missed_deadline
        stats['reached_due'] += pub is not None
        stats['published'] += pub is not None and pub['published']
        stats['stale_commit'] += pub is not None and pub['in_time'] and pub['model_admitted'] and not pub['published']
        stats['ended_pending'] += pub is None
        events.append(dict(lane=lane, sample_tick=tick, due_tick=due, single_robot_ms=ms,
            scheduled_release_ms=charged, deadline_missed=missed_deadline, reached_due=pub is not None,
            published=bool(pub and pub['published']), model_admitted=query['admitted'],
            incumbent_admitted=query['incumbent'], sample_index=query['sample_index'],
            physical_travel_to_due_m=pub['travel_m'] if pub else None,
            offline_batched_compute_ms=query['offline_batch_wall_ms']))
    return old, lane_counts, events, route_changes


def prefix_flight_force(row, trace_file, model, terrain):
    """Match the archived controller force clock to the manuscript's crossing flight."""
    from report_platform_controls import flight, runs
    hand, entry = row['handoff_s'], row['rb_entry_s']
    force = row['variant']['force_y_N']
    onset, steps = row['air_event_tick'], row['applied_force_ticks']
    with np.load(trace_file) as z:
        time = z['trace'][:, 0]
        ticks = np.rint(time / .005).astype(int)
        if not np.array_equal(ticks, np.arange(len(ticks))) or not np.allclose(time, ticks * .005, atol=1e-6, rtol=0):
            raise ValueError(f'Unexpected P0-B control clock: {trace_file}')
        if onset < -1 or onset >= len(ticks) or not 0 <= steps <= 20 or (steps and (not force or onset < 0 or onset+steps > ticks[-1])):
            raise ValueError(f'Inconsistent force receipt: {trace_file}')
        # Preserve the frozen verification window and the original deadline comparison.
        stop = len(time) if hand is None else round(hand / .005) + 2
        loads = z['contact_substep'][:stop]
        n = loads.shape[1]
        dt_ms = 1000 * float(z['contact_physics_dt_s'])
        if loads.shape[2] != 4 or not np.isclose(n * dt_ms, 5):
            raise ValueError(f'Contact telemetry does not span its control interval: {trace_file}')
        supported = z['wheel_contact_flags'][:stop, 2] & (loads[:, -1, :2] > 1e-5).all(1)
        edge = terrain['front_x_m']
        q = z['qpos'][:stop]
        take, leave, landing = ((None, None, None) if entry is None else
                flight(model, q, loads, supported, time[:stop], edge, edge, terrain['far_bank_height_m'], entry))
        verified = hand is not None and landing is not None and landing <= hand + 1e-5
        # The additional diagnostic owns only the prefix, through the interval ending at handback.
        # A verified flight already ended and landed by handback. For an unverified crossing,
        # the frozen verifier's extra geometry row may belong to an independently moving suffix.
        if hand is not None and stop-1 < len(loads):
            end = stop-1
            loads = loads[:end]
            if entry is not None and not verified:
                take, leave, _ = flight(model, q[:end], loads, supported[:end], time[:end],
                                        edge, edge, terrain['far_bank_height_m'], entry)
    start = onset * 5 if onset >= 0 else None
    applied = steps * 5
    marker = min(100, (ticks[-1]-onset)*5) if onset >= 0 else 0
    begin_ms = end_ms = right_censored = overlap = marker_overlap = upper_overlap = None
    if take is not None:
        first, last = round(take/.005), round(leave/.005)
        free = ((loads[:, :, :3] <= 1e-5).all(2) & (loads[:, :, 3] == 0)).ravel()
        matches = [(b, e) for b, e in runs(free) if (b+n-1)//n == first and (e-1)//n == last]
        if len(matches) != 1:
            raise ValueError(f'Cannot resolve the selected crossing flight in contact telemetry: {trace_file}')
        b, e = matches[0]
        # Row k stores the substeps over [(k-1)*5, k*5) ms. Geometry is still sampled at 5 ms.
        begin_ms, end_ms = (b-n)*5/n, (e-n)*5/n
        right_censored = e == len(free)
        if start is not None:
            marker_overlap = max(0., min(start+marker, end_ms)-max(start, begin_ms))
            overlap = max(0., min(start+applied, end_ms)-max(start, begin_ms)) if applied else 0.
            upper_overlap = max(0., start+applied-max(start,begin_ms)) if right_censored else overlap
    status = ('no_trigger' if start is None else 'no_crossing_flight' if take is None else
              'zero_force_marker' if not force else 'no_applied_ticks' if not applied else
              'overlap_censored' if upper_overlap > overlap else
              'no_overlap' if overlap == 0 else 'full_applied_overlap' if overlap == applied else 'partial_applied_overlap')
    return dict(verified=verified, pulse_triggered=start is not None, zero_force_marker=force == 0,
        pulse_trigger_tick=onset, pulse_trigger_ms=start, pulse_requested_end_ms=start+100 if start is not None else None,
        pulse_marker_observed_ms=marker, pulse_applied_ticks=steps, pulse_applied_ms=applied,
        pulse_applied_start_ms=start if applied else None, pulse_applied_end_ms=start+applied if applied else None,
        has_crossing_flight=take is not None, crossing_first_geometry_tick=round(take/.005) if take is not None else None,
        crossing_last_geometry_tick=round(leave/.005) if leave is not None else None,
        flight_contact_grid_ms=5/n, flight_start_ms=begin_ms, flight_observed_end_ms=end_ms,
        flight_end_censored=right_censored, pulse_actual_overlap_ms=overlap if applied else 0.,
        pulse_overlap_upper_ms=upper_overlap if applied else 0.,
        pulse_marker_overlap_ms=marker_overlap, pulse_flight_status=status)


def read_batch(run_dir, name, plan):
    import mujoco
    from run_playground_task_reversal import crossings_by_section
    folder = run_dir / name / 'physical'
    layout = name.split('/')[0]
    cfg = read(run_dir / name / 'config.json')
    status = read(folder / 'status.json')
    if status['stage'] != 'complete':
        raise ValueError(f'Batch {name} is unfinished')
    rows = status['results']
    variants = cfg['ground_handoff']['handoff_variants']
    if not len(rows) == len(variants) == status['total'] == status['completed']:
        raise ValueError(f'Incomplete physical lane set in {name}')
    model = mujoco.MjModel.from_binary_path(str(folder/'terrain.mjb'))
    world = read(run_dir / layout / 'world.json')
    terrain = world['terrain']
    resolved = read(folder/'resolved_config.json')['ground_handoff']['handoff_variants']
    if len(resolved) != len(variants):
        raise ValueError(f'Resolved physical lane count changed in {name}')
    old_events, lane_counts, events, changes = read_events(folder)
    for q in [*events, *changes]:
        lane = q['lane']
        variant = rows[lane]['variant']
        q.update(batch=name, layout=layout, root=variant['bank_row'], force=variant['force_y_N'],
                 geometry=variant['geometry'], alpha=variant['approach_alpha'], arm=variant['arm'],
                 terminal_s=rows[lane]['duration_s'])
    with np.load(folder/'p0_branch_audit.npz') as archive:
        audit = dict(archive)
    lanes, outcomes, prefixes = [], [], []
    for base in range(0, len(rows), 7):
        group = rows[base:base+7]
        v = group[0]['variant']
        prefix = dict(layout=layout, root=v['bank_row'], force=v['force_y_N'], geometry=v['geometry'],
                      alpha=v['approach_alpha'], batch=name, group_start_lane=base,
                      template_jump_start_x_m=variants[base]['jump_start_x_m'], physical_jump_start_x_m=v['jump_start_x_m'])
        pair_ok = bool(audit['pair_ok'][base//7])
        physically_identical = pair_ok and len({r['handoff_s'] for r in group}) == 1
        paths = [folder/r['case_dir']/'trajectory.npz' for r in group]
        # Zero-force markers can be cut short by a different replica's termination.
        pulse_records_equal = len({(r['rb_entry_s'],r['air_event_tick'],r['applied_force_ticks'],
                         min(20,round(r['duration_s']/.005)-r['air_event_tick']) if r['air_event_tick']>=0 else 0)
                         for r in group}) == 1
        share_timing = physically_identical and pulse_records_equal
        timing = prefix_flight_force(group[0], paths[0], model, terrain)
        verified = timing['verified']
        prefixes.append(dict(**prefix, pair_ok=pair_ok, handbacks_equal=physically_identical,
                        first_bad_tick=int(audit['first_bad_tick'][base//7]), **timing,
                        pulse_records_equal=pulse_records_equal, timing_source_lane=base, timing_source=str(paths[0]),
                        handoff_s=group[0]['handoff_s'], rb_entry_s=group[0]['rb_entry_s']))
        physical = {}
        for j, (row, path) in enumerate(zip(group, paths, strict=True)):
            lane = base+j
            raw = variants[lane]
            # The physical runner resolves the world-relative activation offset in the template.
            # Compare the complete resolved variant both to the planned trigger and to the physical receipt.
            offset = raw.get('activation_offset_m',cfg.get('planned_activation_offset_m',world['forward_obstacle']['activation_offset_m']))
            trigger = terrain['front_x_m'] + offset
            target = dict(raw,jump_start_x_m=trigger)
            arm = row['variant']['arm']
            if trigger != plan['trigger_m'] or resolved[lane] != target or target != row['variant'] or arm != plan['arms'][j] or any(
                    row['variant'][k] != v[k] for k in ('bank_row','force_y_N','geometry','approach_alpha')):
                raise ValueError(f'Group or lane/variant identity changed: {name} lane {lane}')
            lane_timing = timing if not j or share_timing else prefix_flight_force(row, path, model, terrain)
            good_jump = verified if physically_identical else lane_timing['verified']
            with np.load(path) as z:
                trace = z['trace']
            x_sections = [12.7, 13.6, 15.0]
            hand = row['handoff_s'] if row['handoff_s'] is not None else np.inf
            gates = {sec:(t,y) for sec, _, t, y in crossings_by_section(trace, hand, x_sections)}
            met = lane_counts[lane]
            before = int(audit['before_first_accepted_ticks'][lane])
            after = int(audit['after_expiry_uncovered_ticks'][lane])
            if before+after != audit['uncovered_ticks'][lane]:
                raise ValueError(f'Measured default phases do not partition uncovered ticks at {name}/{lane}')
            met.update(initial_default_ticks=before, expired_default_ticks=after,
                       ever_initial_default=before>0, ever_expired_default=after>0,
                       published_then_expired_default=after>0 and met['published']>0,
                       default_until_termination=before>0 and met['published']==0)
            if met['queries'] != int(audit['queries'][lane]):
                raise ValueError(f'Query count mismatch for {name} lane {lane}')
            lanes.append(dict(lane_timing, **prefix, lane=lane, arm=arm, pair_ok=physically_identical, verified=good_jump,
                timing_source_lane=base if share_timing else lane,
                duration_s=row['duration_s'], termination=row['termination'], goal=row['goal'],
                wall=bool(row['forbidden_contact_steps']), ever_without_coverage=bool(audit['ever_without_coverage'][lane]),
                handoff_s=row['handoff_s'], rb_entry_s=row['rb_entry_s'], trajectory=str(path),
                near_cross_y=gates['near'][1], far_cross_y=gates['far'][1], **met))
            tasks = TASKS if j < 3 else [TASKS[j-3]]
            for section, side in tasks:
                cross_t, cross_y = gates[section]
                gate = cross_y is not None and .06 <= cross_y * side <= 3.9
                physical[j, section, side] = dict(success=bool(good_jump and row['goal'] and
                        not row['forbidden_contact_steps'] and gate), wall=bool(row['forbidden_contact_steps']),
                    gate=gate, cross_s=cross_t, cross_y=cross_y, verified=good_jump,
                    handoff_s=row['handoff_s'], duration_s=row['duration_s'], hold_s=row['goal_hold_s'],
                    landing_query_release_tick=row.get('departure_applied_tick'), termination=row['termination'],
                    trajectory=str(path), lane=lane, actual_arm=arm,
                    ever_without_coverage=bool(audit['ever_without_coverage'][lane]) if j>=3 else None,
                    **{k:(value if j>=3 else None) for k,value in met.items()})
        if not physically_identical:
            continue
        for j, (section, side) in enumerate(TASKS, start=3):
            lane = base+j
            old = old_events.get(lane)
            if old is None and verified and group[j]['handoff_s'] is not None and group[j]['duration_s'] > group[j]['handoff_s']+.01:
                raise ValueError(f'Missing frozen three-exit decision after verified handback: {name} lane {lane}')
            chosen = old['choice'] if old else 1
            for planner, k in [('old_filter',chosen),('rolling',j),('direct_straight',1),
                               ('direct_gate_exit',0 if side==1 else 2)]:
                outcomes.append(dict(**prefix, section=section, side=side, planner=planner,
                        gate_selection_observed=old is not None if planner=='old_filter' else None,
                        old_admitted=old['admitted'] if planner=='old_filter' and old else None,
                        incumbent_admitted=old['incumbent_admitted'] if planner=='old_filter' and old else None,
                        old_choice_tick=old['tick'] if planner=='old_filter' and old else None,
                        old_best_actual_exit_wins=int(any(physical[d,section,side]['success'] for d in range(3))),
                        **physical[k,section,side]))
    return pd.DataFrame(lanes),pd.DataFrame(prefixes),pd.DataFrame(outcomes),pd.DataFrame(events),pd.DataFrame(changes)


def select_entries(outcomes, plan):
    """Keep hindsight access to unknown pushes out of the fixed-policy and shared-entry rows."""
    efforts = []
    for cell,(library, post) in CELLS.items():
        tab = outcomes[outcomes.planner == post].copy()
        if library == 'old':
            in_old = [(g,a) in LEGACY for g,a in zip(tab.geometry, tab.alpha, strict=True)]
            tab = tab.loc[in_old]
        for pick, g in tab.groupby(GROUP):
            key = dict(zip(GROUP,pick))
            by_action = g.pivot(index=ENTRY, columns='force', values='success').sort_index()
            by_action = by_action.reindex(columns=FORCES)
            if len(by_action)!= (5 if library=='old' else 18) or by_action.isna().any().any():
                raise ValueError(f'Incomplete actions/force matrix {key}')
            shared_best = by_action.sum(axis=1).idxmax()
            fixed = (1,0.) if key['side']==1 else (0,0.)
            for force in FORCES:
                conditional = g[g.force == force].set_index(ENTRY)
                strategies = [('best_shared_entry',shared_best),('force_known_hindsight',by_action[force].idxmax())]
                if library == 'old':
                    strategies.append(('deployed_fixed_gate_entry',fixed))
                for strategy, (route,alpha) in strategies:
                    row = conditional.loc[(route,alpha)].to_dict()
                    efforts.append(dict(row, cell=cell, mode=strategy, geometry=route, alpha=alpha))
    data = pd.DataFrame(efforts)
    return data


def stratified_interval(values):
    """Paired root bootstrap separately in each map, keeping that map's original weight."""
    rng = np.random.default_rng(20261123)
    draws = None
    total = 0
    for _, sub in values.groupby('layout'):
        cluster = sub.groupby('root')['difference'].agg(['sum','count']).to_numpy()
        ids = rng.integers(len(cluster),size=(10000,len(cluster)))
        weighted = cluster[ids,0].sum(1)
        draws = weighted if draws is None else draws+weighted
        total += int(cluster[:,1].sum())
    return 100*np.quantile(draws / total,[.025,.975])


def contrasts(data):
    results = []
    for strategy, frame in data.groupby('mode'):
        for measure in ['success','wall']:
            wide = frame.pivot(index=GROUP+['force'],columns='cell',values=measure).astype(float)
            tests = {'B-A':('B','A'),'C-A':('C','A'),'D-B':('D','B'),'D-C':('D','C')}
            differences = {k:wide[x]-wide[y] for k,(x,y) in tests.items() if x in wide and y in wide}
            if set(CELLS)<=set(wide):
                differences['D-B-C+A'] = wide.D-wide.B-wide.C+wide.A
            for label, series in differences.items():
                cases = series.rename('difference').reset_index()
                groupings = [('all','all',cases)]
                groupings.extend((layout,'all',g) for layout,g in cases.groupby('layout'))
                groupings.extend(('all',str(side),g) for side,g in cases.groupby('side'))
                groupings.extend((layout,str(side),g) for (layout,side),g in cases.groupby(['layout','side']))
                for layout, side, g in groupings:
                    lo,hi = stratified_interval(g)
                    positive_roots = int((g.groupby(['layout','root']).difference.sum()>0).sum())
                    results.append(dict(mode=strategy, metric=measure, contrast=label, layout=layout,
                               side=side, n_contracts=len(g), n_roots=g.groupby(['layout','root']).ngroups,
                               increase_cases=int((g.difference>0).sum()), decrease_cases=int((g.difference<0).sum()),
                               net_positive_roots=positive_roots, delta_pp=100*g.difference.mean(), ci_lo_pp=lo, ci_hi_pp=hi))
    return pd.DataFrame(results)
def write_results(run_dir, destination):
    os.environ.setdefault('JAX_PLATFORMS','cpu')
    os.environ['CUDA_VISIBLE_DEVICES']=''
    plan = read(run_dir/'protocol.json')
    if set(map(tuple, plan['old_actions'])) != LEGACY or not LEGACY <= set(map(tuple, plan['actions'])) or len(plan['actions']) != 18:
        raise ValueError('Expanded library does not contain the frozen five old entries')
    destination.mkdir(parents=True, exist_ok=True)
    batches = [read_batch(run_dir,b['batch'],plan) for b in plan['batches']]
    lanes, prefixes, outcomes, events, changes = (pd.concat(group,ignore_index=True) for group in zip(*batches,strict=True))
    for frame, path in [(lanes,'physical_lanes.csv.gz'),(prefixes,'prefixes.csv'),(outcomes,'measured_exits.csv.gz'),
                        (events,'queries_and_publications.csv.gz'),(changes,'route_changes_and_consumers.csv.gz')]:
        if path=='measured_exits.csv.gz' and not len(frame):
            frame=pd.DataFrame(columns=['layout','root','force','geometry','alpha','section','side','planner','success'])
        if path=='queries_and_publications.csv.gz' and not len(frame):
            frame=pd.DataFrame(columns=['layout','batch','lane','due_tick','published'])
        if path=='route_changes_and_consumers.csv.gz' and not len(frame):
            frame=pd.DataFrame(columns=['layout','batch','lane','version','kind','change_tick','valid_until',
                      'h75_actor_consumed_tick','pursuit_consumed_tick'])
        frame.to_csv(destination/path,index=False)
    if len(lanes)!=plan['trajectory_lanes'] or len(prefixes)!=plan['unique_prefixes']:
        raise ValueError('Incomplete physical prefix or lane counts')
    prefix_keys = ['layout','root','force']+ENTRY
    if prefixes.duplicated(prefix_keys).any() or not prefixes.groupby(['layout','root']+ENTRY).force.nunique().eq(3).all():
        raise ValueError('Missing or duplicate physical prefix identity or force')
    prefixes.groupby(['layout','force','pulse_flight_status','has_crossing_flight','pulse_triggered',
                      'zero_force_marker','flight_contact_grid_ms'],dropna=False).size().rename('prefixes').to_csv(
                      destination/'prefix_pulse_flight_counts.csv')
    relock=prefixes.groupby(['layout']+ENTRY+['force']).agg(prefixes=('verified','size'),verified=('verified','sum'),
                         mismatched_groups=('handbacks_equal',lambda x:int((~x).sum())))
    relock.to_csv(destination/'verified_entry_ranges.csv')
    receipt = dict(role='paired development using archived physical executions',
                   observed_physical_lanes=len(lanes),prefixes=len(prefixes),
                   verified_prefixes=int(prefixes.verified.sum()),invalid_pairing=int((~prefixes.handbacks_equal).sum()),
                   observed_queries=len(events),unpublished_queries=int(lanes.ended_pending.sum()),
                   published_queries=int(lanes.published.sum()),new_18_entry_selector='no deployed 18-entry selector in P0-B',
                   pulse_flight_timing='Integer control ticks times 5 ms give trigger and actual applied endpoints. '
                       'The crossing test uses 5 ms poses rounded to nearest tick. Each recorded solver contact sample '
                       'is assigned its native bin in the preceding 5 ms interval. Report half-open overlap on the '
                       'recorded solver grid, bounded at handback or trajectory end. Zero N has only a marker window. '
                       'Missing crossing plus a real push gives an undefined overlap; no applied pulse gives zero. '
                       'A censored flight end is the last observed boundary. In that case actual_overlap_ms covers '
                       'observed bins and pulse_overlap_upper_ms allows flight to continue through the remaining push.')
    from report_chapter3_continuous_p0_b_xy import export
    if receipt['invalid_pairing']:
        export(run_dir,destination)
        (destination/'analysis.json').write_text(json.dumps(dict(receipt,stage='unmatched_prefixes'),indent=2)+'\n')
        raise ValueError('Physical prefixes disagree: archived individual traces preserved; four-cell contrasts are unidentified')
    data=select_entries(outcomes,plan)
    data.to_csv(destination/'entry_decisions_by_case.csv.gz',index=False)
    groups=['mode','cell','layout','root','side','section']
    data.groupby(groups).agg(n=('force','size'),done=('success','sum'),wall=('wall','sum'),
                            verified=('verified','sum')).to_csv(destination/'scores_by_root_gate.csv')
    totals=data.groupby(['mode','cell','layout','side','section']).agg(n=('force','size'),
                       done=('success','sum'),wall=('wall','sum'),verified=('verified','sum'))
    totals.to_csv(destination/'scores_by_map_gate.csv')
    counts=data.groupby(['mode','cell']).agg(n=('force','size'),done=('success','sum'),
                 wall=('wall','sum'),verified=('verified','sum'))
    counts.to_csv(destination/'four_cells.csv')
    diff=contrasts(data)
    diff.to_csv(destination/'paired_contrasts.csv',index=False)
    unmatched_modes = pd.DataFrame([dict(cell=c,entry_policy='no online expanded-library selector measured',
                    done=pd.NA,wall=pd.NA) for c in ('C','D')])
    unmatched_modes.to_csv(destination/'unmeasured_online_cells.csv',index=False)
    # Pair three physical controls from the exact same prefix, with no reference to future pushes:
    # 1) select using frozen filter (default straight), 2) always publish straight, 3) follow commanded gate.
    control_keys=GROUP+['force']+ENTRY
    cmp=outcomes[outcomes.planner.isin(['old_filter','direct_straight','direct_gate_exit'])].pivot(
                    index=control_keys,columns='planner',values=['success','wall'])
    cmp.columns=['_'.join(z) for z in cmp.columns]
    cmp=cmp.reset_index()
    old_entries = [(g,a) in LEGACY for g,a in zip(cmp.geometry,cmp.alpha,strict=True)]
    cmp['in_old_entry_library']=old_entries
    cmp['historically_fixed_gate_entry']=(((cmp.side==1)&(cmp.geometry==1))|((cmp.side==-1)&(cmp.geometry==0))) & (cmp.alpha==0)
    cmp.to_csv(destination/'same_state_filter_controls.csv.gz',index=False)
    for scope,rows in [('all_18',cmp),('old_5',cmp[cmp.in_old_entry_library]),
                       ('fixed_gate_entry',cmp[cmp.historically_fixed_gate_entry])]:
        rows.groupby(['layout','side']).agg(n=('root','size'),
            filter_done=('success_old_filter','sum'),always_straight_done=('success_direct_straight','sum'),
            gate_direct_done=('success_direct_gate_exit','sum'),filter_wall=('wall_old_filter','sum'),
            always_straight_wall=('wall_direct_straight','sum'),gate_direct_wall=('wall_direct_gate_exit','sum')
            ).to_csv(destination/f'shared_state_controls_{scope}.csv')
    rolling=lanes[~lanes.arm.isin(DIRS)].copy()
    rolling['domain_exit']=rolling.termination.eq('tracking_domain_exit')
    rolling.groupby(['layout','arm']).agg(n=('arm','size'),verified=('verified','sum'),
        queries=('queries','sum'),rejected=('model_rejected','sum'),late=('missed_deadline','sum'),
        stale=('stale_commit','sum'),terminated_before_release=('ended_pending','sum'),published=('published','sum'),
        ever_without_coverage=('ever_without_coverage','sum'),domain_exits=('domain_exit','sum')).to_csv(destination/'rolling_diagnostics.csv')
    coverage = dict(n=('lane','size'),verified=('verified','sum'),ever_initial_default=('ever_initial_default','sum'),
            stayed_on_default=('default_until_termination','sum'),ever_published=('published',lambda a:int((a>0).sum())),
            published_then_expired=('published_then_expired_default','sum'),initial_default_ticks=('initial_default_ticks','sum'),
            after_expiry_default_ticks=('expired_default_ticks','sum'),domain_exits=('domain_exit','sum'))
    for groups,label in [(['layout','root','arm'],'root_gate'),(['layout','arm'],'map_gate')]:
        rolling.groupby(groups).agg(**coverage).to_csv(destination/f'coverage_by_{label}.csv')
    rolling[['batch','group_start_lane','lane','layout','root','force','alpha','geometry','arm','termination',
             'published','initial_default_ticks','expired_default_ticks','ever_initial_default','default_until_termination',
             'published_then_expired_default']].to_csv(destination/'coverage_per_rolling_arm.csv.gz',index=False)
    compared = outcomes[outcomes.planner=='old_filter'].merge(
        outcomes[outcomes.planner=='rolling'],on=control_keys,suffixes=('_old','_cem'),validate='one_to_one')
    compared['coverage_phase'] = np.select([
        compared.published_then_expired_default_cem.astype(bool),compared.published_cem.gt(0),
        compared.default_until_termination_cem.astype(bool)],
        ['published_then_expired','published_without_expiry','initial_default_only'],default='no_active_rolling_tick')
    compared.groupby(['layout','root','section','side','coverage_phase']).agg(n=('side','size'),
          old_done=('success_old','sum'),rolling_done=('success_cem','sum'),
          old_wall=('wall_old','sum'),rolling_wall=('wall_cem','sum')).to_csv(destination/'same_state_outcomes_by_coverage.csv')
    if len(changes):
        changes.groupby(['layout','root','kind']).agg(n=('lane','size'),
            actor_receipts=('h75_actor_consumed_tick','count'),pursuit_receipts=('pursuit_consumed_tick','count'),
            actor_ms_from_change_p95=('h75_actor_consumed_ms_from_change',lambda v:v.quantile(.95)),
            pursuit_ms_from_change_p95=('pursuit_consumed_ms_from_change',lambda v:v.quantile(.95)),
            actor_ms_from_query_p95=('h75_actor_consumed_ms_from_query',lambda v:v.quantile(.95))
            ).to_csv(destination/'physical_consumption_by_root.csv')
    if len(events):
        events.groupby('layout').agg(n=('lane','size'),single_robot_obs_to_publish_p50_ms=('single_robot_ms','median'),
            request_p95_ms=('single_robot_ms',lambda v:v.quantile(.95)),request_p99_ms=('single_robot_ms',lambda v:v.quantile(.99)),
            publication_clock_p95_ms=('scheduled_release_ms',lambda v:v.quantile(.95)),
            due_reached=('reached_due','sum'), issued_late=('deadline_missed','sum'),
            published=('published','sum'),distance_to_due_p95_m=('physical_travel_to_due_m',lambda v:v.quantile(.95))
            ).to_csv(destination/'latency_by_map.csv')
    xy_receipt=export(run_dir, destination)
    (destination/'analysis.json').write_text(json.dumps(dict(receipt, stage='complete',
        xy_root_index='xy/index.html',xy_physical_lanes=xy_receipt['physical_lanes'],
        primary='best_shared_entry; hypothetical finite-library potential, one pre-choice across all 3 forces',
        measured_pair='fixed gate entry: A/B only. C/D have no expanded online approach selector',
        truth='old_filter follows measured old_selection event on the recovered state; rejected/no control defaults to straight; CEM missions use the physical executed rolling trajectories',
        force_known_hindsight='separate diagnostic, never available online',
        bootstrap='10000 draws stratified by map and paired by reset root across arms, forces and contracts; development maps held fixed',
        simultaneous_intervals=False),indent=2)+'\n')
    print(counts.to_string(),flush=True)


def verify_inputs(run_dir):
    """Check the frozen pre-physics contract without loading or fabricating any results."""
    plan=read(run_dir/'protocol.json')
    expected=set(product(range(6),(0.,.5,1.)))
    if set(map(tuple,plan['actions']))!=expected or set(map(tuple,plan['old_actions']))!=LEGACY:
        raise ValueError('Action contract mismatch')
    count=0
    for b in plan['batches']:
        cfg=read(run_dir/b['batch']/'config.json')
        variants=cfg['ground_handoff']['handoff_variants']
        if len(variants)!=b['worlds'] or len(variants)%7:
            raise ValueError('Incomplete physical batch')
        for k in range(0,len(variants),7):
            g=variants[k:k+7]
            if ([r['arm'] for r in g]!=plan['arms'] or
                 len({(r['bank_row'],r['geometry'],r['approach_alpha'],r['force_y_N']) for r in g})!=1):
                raise ValueError('Malformed paired replicas')
            if [(r['bin'],r['side']) for r in g[3:]]!=list(TASKS):
                raise ValueError('Mismatch between section/side and physical CEM arms')
        count+=len(variants)
    if count!=plan['trajectory_lanes'] or count//7!=plan['unique_prefixes']:
        raise ValueError('Incorrect physical configuration dimensions')
    print(f'Validated input contract only: {count//7} prefixes / {count} planned physical replicas; zero physics results read.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    parser.add_argument('--output',type=Path,help='Results folder (default: RUN/analysis).')
    parser.add_argument('--check-inputs',action='store_true',help='Validate frozen configs before the physical batches exist.')
    args=parser.parse_args()
    root=args.run.resolve()
    verify_inputs(root)
    if not args.check_inputs:
        write_results(root,args.output.resolve() if args.output else root/'analysis')


if __name__=='__main__':
    main()
