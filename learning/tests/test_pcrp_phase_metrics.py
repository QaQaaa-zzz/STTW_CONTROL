"""Regression cases for E2 matched denominators, no events and boundary interpolation."""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'cli'))
from analyze_pcrp_e2 import ALPHAS, CENTERS, SHIFTS, SPEEDS, classify, entry_index, summarize


def test_matched_pair_counts_include_failed_cases_and_velocity_zero():
    rows = []
    for arm, centers, groups, values in (
            ('E2a', CENTERS, range(401,409), SHIFTS),
            ('E2b', CENTERS, range(401,405), SPEEDS),
            ('E2c', (None,), range(401,405), SPEEDS)):
        for center in centers:
            for group in groups:
                for value in values:
                    for alpha in ALPHAS:
                        rows.append(dict(experiment=arm, center_m=center, group=group, alpha=alpha,
                                         ds_cm=value if arm=='E2a' else 0, dv_m_s=value if arm!='E2a' else 0,
                                         full=(alpha < .5) if center==3.9 else (alpha > .5)))
    counts, smooth, summary = summarize(rows)
    overall = {r['metric']:r for r in counts if r['subset']=='overall'}
    assert [overall[k]['pairs'] for k in ('f_alpha','f_s','f_v','f_v_flat')] == [96,720,240,80]
    assert overall['f_alpha']['flips']==24 and all(overall[k]['flips']==0 for k in ('f_s','f_v','f_v_flat'))
    assert summary['decisions']['nominal_centers']=='alpha_structure'
    assert all(r['n']==56 for r in smooth)
    assert all(r['pairs']==0 for r in counts if r['subset']=='mixed_group_cell')
    assert classify(0,0,0,0,False)=='degenerate_alpha_benchmark'
    assert classify(.4,0,.8,.8,True)=='phase_dominated'


def test_oblique_front_wheel_crossing_and_missing_entry():
    route = dict(xy=np.array([[0.,0.],[1.,0.],[2.,0.]]),s=np.arange(3.),turn_end=2.)
    terrain = dict(center_s_m=1.,length_m=.8,oblique_s_per_lateral_m=.25,half_width_m=.8)
    pts = np.array([[.59,.4],[.62,.4],[.71,.4]])
    assert entry_index(pts,route,terrain)==2
    assert entry_index(pts[:2],route,terrain) is None
    assert entry_index(pts+[.8,2],route,terrain) is None


def test_dedicated_zero_sources_and_actual_same_slot_pairs():
    import pytest
    from analyze_pcrp_e2 import adjacent_pair_evidence

    rows = []
    for arm, centers, groups, offsets in (
            ('E2a', CENTERS, range(401,409), SHIFTS),
            ('E2b', CENTERS, range(401,405), SPEEDS),
            ('E2c', (None,), range(401,405), SPEEDS)):
        axis = 'ds' if arm=='E2a' else 'dv'
        for index,center in enumerate(centers):
            for group in groups:
                # Different coordinates and slot assignment between DS and DV are valid.
                slot = index*len(groups)+group-401 + (12 if arm=='E2c' else 0)
                origin = [float(slot),0. if axis=='ds' else 3.,0.]
                for shift in offsets:
                    for alpha in ALPHAS:
                        batch = f'{axis}/{shift:g}/{alpha:g}'
                        case = f'{arm}/{center}/{group}'
                        full = True if axis=='ds' else (shift==0 and alpha==.25)
                        rows.append(dict(experiment=arm,center_m=center,group=group,alpha=alpha,
                                         ds_cm=shift if axis=='ds' else 0,dv_m_s=shift if axis=='dv' else 0,
                                         full=full,axis=axis,actual_global_slot=slot,actual_origin_m=origin,
                                         actual_batch=batch,source=f'{batch}/{case}',
                                         zero_source=f'{axis}/0/{alpha:g}/{case}',
                                         alpha0_source=f'{axis}/{shift:g}/0/{case}'))
    edges = adjacent_pair_evidence(rows)
    assert len(edges)==2032 and sum(r['primary_alpha'] for r in edges)==96
    counts, _, _ = summarize(rows)
    overall = {r['metric']:r['flips'] for r in counts if r['subset']=='overall'}
    assert [overall[k] for k in ('f_alpha','f_alpha_groups401_404','f_s','f_v','f_v_flat')]==[0,24,0,24,8]
    changed = [dict(r) for r in rows]
    changed[0]['actual_global_slot'] += 1
    with pytest.raises(ValueError,match='same-slot'):
        adjacent_pair_evidence(changed)
    changed[0] = dict(rows[0],actual_origin_m=[.125,0.,0.])
    with pytest.raises(ValueError,match='world-origin'):
        adjacent_pair_evidence(changed)
    changed[0] = dict(rows[0],zero_source='other_packing/zero')
    with pytest.raises(ValueError,match='same-axis zero'):
        adjacent_pair_evidence(changed)


def test_actual_source_lane_is_recorded_from_split_and_aggregate(tmp_path):
    import json
    import pytest
    from analyze_pcrp_e2 import measured_slot

    folder=tmp_path/'some_batch'
    batch=folder/'batch'
    batch.mkdir(parents=True)
    origins=[[float(i),3.,0.] for i in range(16)]
    (batch/'contract.json').write_text(json.dumps(dict(settings=dict(num_envs=16,alphas=[.75]*16),origin=origins)))
    split=dict(origin=[origins[6]],batch_source=dict(output=str(batch),lane_start=6,lane_stop=7))
    row=dict(source=str(folder/'case'),lane=0,axis='dv',alpha=.75,batch=folder.name,planned_slot=6,planned_origin=origins[6])
    assert measured_slot(row,split)==dict(actual_batch=str(batch),actual_global_slot=6,actual_origin_m=origins[6])
    with pytest.raises(ValueError,match='physical slot'):
        measured_slot(dict(row,planned_slot=0),split)
    with pytest.raises(ValueError,match='world origin'):
        measured_slot(row,dict(split,origin=[origins[4]]))
