import copy
import json
from pathlib import Path

import numpy as np
import pytest

from sttw_control.reference_feedback_transfer import transfer_cases,build_transfer_path


def inputs():
    protocol=json.loads((Path(__file__).parents[1]/'configs/feedback_geometry_transfer.json').read_text())
    contexts={f"{spec['name']}__bank{bank}":dict(geometry_name=spec['name'],bank_index=bank,
        speed_m_s=spec['speed_m_s'],world_direction=[0.,1.,0.])
        for spec in protocol['contexts'] for bank in protocol['bank_indices']}
    return protocol,contexts


def test_full_actor_feedback_factorial_has_matching_events_across_all_alphas_and_starts():
    p,c=inputs();old=copy.deepcopy(c);cases=transfer_cases(p,c)
    assert c==old and len(cases)==240 and len({r['case_id'] for r in cases})==240
    assert sum(not r['force_N'] for r in cases)==48
    event_onsets={}
    for row in cases:
        assert row['mode_index']==1 and row['causal_support'] and row['profile']==0
        assert row['actor_role']==('parent_causal' if row['role'].startswith('parent') else 'free_GAE')
        expected=.1 if row['role'] in ['parent_feedback_010','feedback_010'] else 0.
        assert row['feedback_gain_rad']==expected
        event_onsets.setdefault((row['geometry_name'],row['force_N']),set()).add(row['pulse_onset_task_tick'])
    assert all(len(v)==1 for v in event_onsets.values())
    assert transfer_cases(p,c)==cases


def test_transfer_cannot_drop_controls_or_change_gain_goal_or_alpha_panel():
    p,c=inputs()
    for key,value in [('roles',p['roles'][:-1]),('feedback_gain_rad',.05),('goal_radius_m',.6),('alphas',[0.,1.])]:
        wrong=copy.deepcopy(p);wrong[key]=value
        with pytest.raises(ValueError):transfer_cases(wrong,c)
    with pytest.raises(ValueError):transfer_cases(p,dict(list(c.items())[:-1]))


def test_generated_paths_have_declared_handedness_angle_radius_and_original_anchor():
    p,_=inputs()
    cfg={'path':dict(spacing_m=.02,curvature_transition_length_m=.5,
                    goal_section_offset_after_turn_m=4.)}
    pose=[7.8956,-.00025,-.00006]
    original=copy.deepcopy(cfg)
    for spec in p['contexts']:
        arrays,pc=build_transfer_path(spec,pose,cfg,p['straight_prefix_m'])
        sign=1 if spec['build_direction']=='left' else -1
        assert len(arrays['s'])==4001 and arrays['xy'].shape==(4001,2)
        np.testing.assert_allclose(arrays['xy'][0],pose[:2],atol=1e-6)
        np.testing.assert_allclose(arrays['heading'][-1]-pose[2],sign*np.deg2rad(spec['angle_deg']),atol=2e-5)
        np.testing.assert_allclose(sign*arrays['curvature'].max() if sign>0 else arrays['curvature'].min(),sign/spec['radius_m'],atol=1e-6)
        assert float(arrays['goal'])>float(arrays['turn_end'])
        assert pc['turn_radius_m']==spec['radius_m']
    assert cfg==original


def test_goal_summary_uses_real_parent_when_plain_parent_is_not_in_factorial():
    from sttw_control.finite_value_evaluation_report import goal_report
    roles=['parent_causal','free_GAE','parent_feedback_010','feedback_010']
    rows=[dict(role=role,context='left',alpha=0.,event_id='F16',force_N=16.,
               goal_arrived=role=='parent_feedback_010',physical_outcome='survived',scored_cost=1.) for role in roles]
    result=goal_report(rows,roles)['disturbed']
    assert result['paired']['parent_feedback_010_vs_parent_causal']['arrival_gains']==1
    assert not any('parent_plain' in name for name in result['paired'])
