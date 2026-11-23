"""E1c winner identity and case/trial denominators at decision boundaries."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'cli'))
from pcrp_oracle_robustness import LAYOUTS, SHIFTS, case_name, confirmed_layouts, frozen_winner, majority_evidence


def cases(n, gravel=0, one_cm=(1,1)):
    nominal, shifted = [], []
    for layout in LAYOUTS:
        for group in range(401,405):
            key = case_name(layout,group)
            ridge = layout.startswith('position')
            passed = layout=='flat_original' or layout=='gravel010' and group-401<gravel or ridge and n>0
            if ridge and passed:
                n-=1
            nominal.append(dict(case=key,layout=layout,group=group,C=int(passed)))
            if ridge:
                for d,ok in zip(SHIFTS,[0,*one_cm,0]):
                    shifted.append(dict(case=key,shift_cm=d,C=ok))
    return nominal,shifted


def test_exact_majority_and_knife_edge():
    ns,sh=cases(6,one_cm=(0,1))
    _,d=majority_evidence(ns,sh)
    assert d['verdict']=='case_by_case'
    assert d['eligible_one_of_two']==6
    sh[1]['C']=1
    rows,d=majority_evidence(ns,sh)
    assert d['verdict']=='conditional_majority_or_missing_shifts'
    assert d['readings']['pooled_trials']['passes']==7
    assert d['readings']['both_shifts_per_case']['passes']==1
    sh[5]['C']=sh[6]['C']=0
    rows,d=majority_evidence(ns,sh)
    assert d['eligible_zero_of_two']==1
    assert sum(x['blade_edge'] is True for x in rows)==1


def test_missing_nonridge_geometry_retains_uncertainty_and_all_20_cases():
    ns,sh=cases(3,3)
    rows,d=majority_evidence(ns,sh)
    assert len(rows)==24 and d['nonflat_nominal_pass']==6 and d['nonflat_nominal_total']==20
    assert d['readings']['pooled_trials']==dict(passes=6,observed=6,possible_additional=6,
        all_nominal_denominator=12,majority_certain=False,majority_possible=True)
    assert d['verdict']=='conditional_majority_or_missing_shifts' and d['unmeasured_eligible']==3
    assert next(r for r in rows if r['layout']=='gravel010')['blade_edge'] is None
    assert majority_evidence(*cases(7,1))[1]['verdict']=='route_B'
    assert majority_evidence(*cases(1))[1]['verdict']=='route_A_limited_search'
    ns[0]['C']=0
    assert majority_evidence(ns,sh)[1]['verdict']=='halt_flat_control'
    with pytest.raises(ValueError,match='Incomplete shift set'):
        majority_evidence(ns,sh[:-1])
    for spec in ({}, {'user_confirmation':'ridge only','nonridge':{'flat_original':None}}):
        with pytest.raises(ValueError,match='confirmed choice'):
            confirmed_layouts(spec)


def test_winner_uses_actual_source_lane_and_global_knots_index(tmp_path):
    contract=dict(settings=dict(case_labels=['CEM_1_0','CEM_1_1','Zero'],alphas=[.75,.75,0],
                                zero_middle=[False,False,True]),origin=[[0,0,0],[3,3,0],[6,6,0]])
    (tmp_path/'contract.json').write_text(json.dumps(contract))
    row=dict(raw=str(tmp_path),lane='2',method='Zero',generation=0,alpha_seed=.75)
    times=[0.,5.,10.]
    a,zero,ori,knots=frozen_winner(row,times)
    assert a==0 and zero and ori==[6,6,0] and np.all(knots==0)
    knots_file=tmp_path/'knots.npz'
    offsets=np.arange(42*3*2).reshape(42,3,2)/10000
    np.savez(knots_file,times_s=times,offsets=offsets)
    a,zero,ori,knots=frozen_winner(dict(row,method='CEM_1_1',lane='1',generation=1,
                                       knots_npz=str(knots_file),knots_lane='33'),times)
    assert a==.75 and not zero and ori==[3,3,0]
    np.testing.assert_array_equal(knots,offsets[33])
