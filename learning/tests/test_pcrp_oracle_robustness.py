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


def test_same_slot_replay_keeps_all_96_slots_and_routes_32_controls_separately(tmp_path):
    from collections import Counter
    from pcrp_oracle_search import PILOT
    from pcrp_replay_gate import sha
    from pcrp_oracle_robustness import paired_batches

    assets={}
    terrain_for={}
    for layout in LAYOUTS:
        single=layout.startswith('position_s')
        for cm in (0,*SHIFTS) if single else (0,):
            asset=tmp_path/f'{layout}_{cm}'
            asset.mkdir()
            meta=dict(sha256=asset.name,center_s_m=float(layout.split('_s')[1])+cm/100 if single else 0.,
                      height_m=.1,length_m=.8,half_width_m=.8,route_sha256='route',
                      horizontal_scale_m=.1,vertical_scale_m=.005,oblique_s_per_lateral_m=.25)
            (asset/'identity.json').write_text(json.dumps(meta))
            if cm:assets[layout,cm]=asset
            else:terrain_for[layout]=asset
    roles=Counter()
    for stage,n in [('E1a',3),('E1b',21)]:
        stage_cases,winners=[],[]
        knots=tmp_path/f'{stage}_warm.npz'
        np.savez(knots,times_s=[0,1.],offsets=np.zeros((n,2,2)))
        for layout in LAYOUTS:
            for g in range(401,405):
                if (stage=='E1a')!=(layout in PILOT and g==401):
                    continue
                slot=len(stage_cases)
                name=case_name(layout,g)
                path=tmp_path/stage/name
                path.mkdir(parents=True)
                terrain=terrain_for[layout]
                method='fixed0.75' if slot%2==0 else 'Zero'
                alpha=.75 if slot%2==0 else 0.
                c=dict(settings=dict(case_labels=[method],alphas=[alpha],zero_middle=[method=='Zero'],
                                     oracle_knots=str(knots),oracle_knots_sha256=sha(knots),seed=g,randomization={'group':g}),
                       batch_source=dict(lane_start=slot,lane_stop=slot+1),
                       origin=[[slot,2,0]],terrain=[dict(sha256=terrain.name)])
                (path/'contract.json').write_text(json.dumps(c))
                stage_cases.append(dict(case=name,slot=slot,layout=layout,group=g,terrain=str(terrain),
                                  terrain_sha256=terrain.name,randomization={'group':g}))
                winners.append(dict(case=name,raw=str(path),lane=0,physical_slot=slot,method=method,
                                    generation=0,knots_npz='',knots_lane=''))
        protocol=dict(stage=stage,times_s=[0,1.],cases=stage_cases)
        batches=paired_batches(protocol,winners[::-1],assets,tmp_path/'e1c')
        assert [b[0].name for b in batches]==[f'ds{d:+d}cm' for d in SHIFTS]
        for _,rows,manifests,bias in batches:
            assert [r['physical_slot'] for r in rows]==list(range(n)) and bias.shape==(n,2,2)
            assert not bias.any()
            for r,m,c in zip(rows,manifests,stage_cases,strict=True):
                roles[r['role']] += 1
                assert r['case']==c['case'] and r['origin']==[c['slot'],2,0]
                assert m['alphas']==[r['alpha']] and m['zero_middle']==[r['zero_middle']]
                if r['role']=='nominal_control':
                    assert r['shift_cm'] is None and m['terrain_path']==c['terrain']
                else:
                    assert r['shift_cm']==r['batch_shift_cm'] and m['terrain_path']!=c['terrain']
    assert roles=={'ridge_shift':64,'nominal_control':32}
    ns,sh=cases(3,3)
    with pytest.raises(ValueError,match='Unchanged terrain controls'):
        majority_evidence(ns,[*sh,dict(case=ns[0]['case'],role='nominal_control',C=0,shift_cm=None)])


def test_nominal_repeat_compares_measured_trace_tick_task_and_score(tmp_path):
    from pcrp_oracle_robustness import control_comparison
    orig,rerun=tmp_path/'source',tmp_path/'repeat'
    trace=np.zeros((1,4,1,11),np.float32)
    trace[:,:,0,6]=1
    trace[:,:,0,7]=25
    for path in (orig,rerun):
        path.mkdir()
        np.savez(path/'trajectory.npz',lower_trace=trace)
        (path/'contract.json').write_text(json.dumps(dict(physics_dt=.0002,lower_dt=.005)))
    nominal=dict(raw=str(orig),terminal='full_window',full='1',goal='0',hold='1',C='0',time_s='10',V='1.2')
    actual=dict(output=str(rerun),terminal='full_window',full=1,goal=0,hold=1,C=0,V=1.2)
    check=control_comparison(nominal,actual)
    assert check['matched'] and check['xy_bitwise'] and check['V_delta']==0
    trace[0,-1,0,0]=.25
    trace[0,-1,0,7]=12
    np.savez(rerun/'trajectory.npz',lower_trace=trace)
    check=control_comparison(nominal,dict(actual,terminal='roll_limit_failure',full=0,V=10.3))
    assert not check['matched'] and not check['times_equal'] and not check['terminal_equal']
    assert check['max_xy_distance_m']==.25 and np.isclose(check['V_delta'],9.1)
