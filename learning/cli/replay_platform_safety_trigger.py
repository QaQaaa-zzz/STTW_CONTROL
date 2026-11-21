"""Score a frozen safety-only switch on paired, physically executed C and C-R continuations."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pandas as pd
import torch

from sttw_control.platform_online import PlatformDepartureQuery, velocity_state
from sttw_control.playground_history_model import load_history_model


def read(path):
    return json.loads(path.read_text())


@torch.inference_mode()
def replay(source, output):
    output.mkdir(exist_ok=True, parents=True)
    rows = pd.read_csv(source / 'online_cases.csv')
    keys = ['layout', 'skill', 'root', 'force', 'section', 'side']
    pairs = rows[rows.planner == 'C'].merge(rows[rows.planner == 'C-R'], on=keys,
                                            suffixes=('_c', '_cr'), validate='one_to_one')
    prefix = pd.read_csv(source / 'matched_cases.csv')
    if not prefix.same_prefix.all() or len(prefix) != len(pairs):
        raise ValueError('Missing paired physical prefix')
    sections = dict(zip(('near', 'mid', 'far'), read(source / 'plan.json')['section_x'], strict=True))
    changed = pairs[pairs.chosen_dep_c != pairs.chosen_dep_cr]
    details = []
    def tensor(x):
        return torch.tensor(np.asarray(x), dtype=torch.float32, device='cuda')

    for (layout, skill), group in changed.groupby(['layout', 'skill']):
        path = Path(group.trajectory_cr.iloc[0]).parent.parent
        cfg = read(path / 'resolved_config.json')
        hand = cfg['ground_handoff']
        online = cfg['platform_online']
        physics = mujoco.MjModel.from_binary_path(str(path / 'terrain.mjb'))
        engine = PlatformDepartureQuery(physics, load_history_model(Path(online['ground_model'])),
                                        hand['goal'], online['margin_m'], hand['road_half_width_m'], online['relative'])
        with np.load(cfg['paired_departures']['route_bank']) as bank:
            route = SimpleNamespace(**{k: tensor(bank[k]) for k in ('s','xy','heading','curvature')})
            xy = bank['xy'].astype(np.float32, copy=False)
        # Restrict matching to the actual approach/direction: straight references can coincide.
        stride = len(route.s)
        anchors = {}
        for approach, departure in set(zip(group.approach_cr, group.chosen_dep_cr)):
            offset = 3 + (3 * approach + departure) * stride
            anchors[approach, departure] = {sha256(row).digest(): j for j, row in enumerate(xy[offset:offset+stride])}
        group = group.assign(tick=np.rint(group.handoff_s_cr / .005).astype(int) + 1)
        for tick, block in group.groupby('tick'):
            states, refs = [], []
            for row in block.itertuples():
                with np.load(row.trajectory_cr) as z:
                    q, v = z['qpos'][[tick, tick-4]], z['qvel'][[tick, tick-4]]
                    anchor = anchors[row.approach_cr, row.chosen_dep_cr][sha256(z['published_reference_xy']).digest()]
                refs.append(3 + (3 * row.approach_cr + np.arange(3)) * stride + anchor)
                states.append((q[0],v[0],q[1],v[1]))
            q, v, pq, pv = [tensor(np.stack(a)) for a in zip(*states)]
            vel = velocity_state(q, v)
            request = hand['handoff_variants'][0]['continuation_speed_m_s']
            args = (q, vel[:,0], torch.full_like(vel[:,0], request), route, tensor(refs).long(),
                    tensor(block.section.map(sections)), tensor(block.side), tensor(block.chosen_dep_c).long(),
                    10. - tick * .005, vel, velocity_state(pq, pv))
            choice, vote, old_vote, reason = engine.choose(*args, return_reasons=True)
            result = pd.DataFrame({k: val.cpu().numpy() for k,val in reason.items()},index=block.index)
            result['replay_choice'] = choice.cpu().numpy()
            result['replay_vote'] = vote.cpu().numpy()
            result['replay_incumbent_vote'] = old_vote.cpu().numpy()
            result['margin_m'] = engine.margin
            details.append(result)
        print(layout, skill, len(group), 'changed contracts replayed', flush=True)
    replayed = pd.concat(details)
    pairs = pairs.join(replayed)
    mismatch = pairs.loc[replayed.index].eval('replay_choice != chosen_dep_cr or replay_vote != query_feasible_cr or replay_incumbent_vote != query_feasible_c')
    pairs.to_csv(output / 'replay_audit.csv', index=False)
    if mismatch.any():
        raise ValueError(f'{mismatch.sum()}/{len(mismatch)} replay decisions differ, see replay_audit.csv')
    keep = pairs.clear.eq(True) & pairs.in_domain.eq(True)
    for col in ['chosen_dep','success','collision','termination','trajectory','query_feasible']:
        pairs[col + '_safe'] = pairs[col + '_cr'].where(~keep, pairs[col + '_c'])
    pairs.to_csv(output / 'paired_cases.csv', index=False)
    print(pairs.groupby('skill')[['success_c','success_cr','success_safe','collision_c','collision_cr','collision_safe']].sum())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    replay(args.archive, args.output)
