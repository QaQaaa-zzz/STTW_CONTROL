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
