import copy
import numpy as np
import pytest
from sttw_control import tolerance_late_evaluation as late
from sttw_control.tolerance_predictor_pilot import metrics


def inventory():
    result=[]
    for source in range(4):
        for alpha in (0.,.5,1.):
            for preparation in (.2,.6):
                for force in (-64.,64.):
                    result.append(dict(case_id=str(len(result)),source_group_id=f's{source}',source_index=source,split='late_state_development',bank_index=2 if source<2 else 5,candidate_alpha=alpha,preparation_s=preparation,force_N=force,duration_s=.1,onset_s=6.+preparation,primary_window_start_s=6.,primary_window_end_s=6.+preparation+.1+3.))
    return result


def test_exact_late48_conditions_and_no_oldbank_or_timing_mix():
    cases=inventory();late.validate_inventory(cases,copy.deepcopy(cases))
    cases[0]['bank_index']=0
    with pytest.raises(ValueError,match='banks2/5'):late.validate_inventory(cases,cases)
    cases=inventory();cases[0]['primary_window_start_s']=1.
    with pytest.raises(ValueError,match='task6'):late.validate_inventory(cases,cases)
    cases=inventory();cases[0]['candidate_alpha']=.5
    with pytest.raises(ValueError,match='four complete'):late.validate_inventory(cases,cases)


def test_inventory_record_substitution_rejected():
    cases=inventory();records=copy.deepcopy(cases);records[0]['force_N']=64.
    with pytest.raises(ValueError,match='condition changed'):late.validate_inventory(cases,records)


def test_frozen_authentication_precedes_any_new_label_access(tmp_path,monkeypatch):
    def denied(path):raise ValueError('frozen q checkpoint changed')
    monkeypatch.setattr(late.coordination,'load_export',denied)
    def forbidden(*args):raise AssertionError('source module loaded before authentication')
    monkeypatch.setattr(late.coordination,'file_module',forbidden)
    with pytest.raises(ValueError,match='checkpoint changed'):late.evaluate_late(tmp_path/'nonexistent_late',tmp_path/'q',tmp_path/'out')
    assert not (tmp_path/'out').exists()


def test_user_worst_hypothesis_threshold_report_is_distinct_from_row_labels():
    meta=[dict(source_group_id='s',alpha=1.,bank_index=2,preparation_s=p,force_N=f) for p in (.2,.6) for f in (-64.,64.)]
    y=np.array([1.,1.,1.,0.]);p=np.array([.9,.81,.79,.1])
    report=late.rejection_report(y,p,meta)
    assert report['0.8']==dict(user_rows=4,rejected_rows=2,survived_but_rejected_rows=1,fall_but_not_rejected_rows=0)
    assert report['0.85']['rejected_rows']==3
    assert report['four_hypothesis_user_gate']['s']==dict(resolved_user_rows=4,qmin=.1,reject_at_08=True,reject_at_085=True,actual_no_fall=3,actual_all_four_no_fall=False)
    incomplete=late.rejection_report(y[:3],p[:3],meta[:3])
    assert incomplete['four_hypothesis_user_gate']['s']['four_hypothesis_assessment'].startswith('incomplete')


def test_late_overall_and_grouped_ranking_keep_denominators():
    meta=[dict(source_group_id='s',alpha=a,bank_index=2,preparation_s=.2,force_N=64.) for a in (0.,.5,1.)]
    report=late.grouped_metrics(np.array([1.,1.,0.]),np.array([.9,.8,.1]),meta,metrics)
    assert report['overall']['discordant_pair_ranking']['pairs']==2
    assert report['overall']['discordant_pair_ranking']['mean']==1
    assert report['per_bank']['2']['n']==3 and report['per_source']['s']['n']==3


def test_explicit_late_plan_mode():
    protocol=dict(expected_cases=48,source_task_s=6.,horizon_s=10.,new_horizon_s=4.,maximum_prefix_lower_ticks=0,maximum_lower_ticks=38400,maximum_total_lower_ticks=38400,optimizer_updates=0,bank_indices=[2,5],candidate_alphas=[0.,.5,1.],preparation_s=[.2,.6],forces_N=[-64.,64.],duration_s=.1,primary_after_pulse_end_s=3.)
    late.validate_plan(dict(mode='h75_late_state_tolerance48',protocol=protocol))
    protocol['source_task_s']=1.
    with pytest.raises(ValueError,match='protocol mismatch'):late.validate_plan(dict(mode='h75_late_state_tolerance48',protocol=protocol))
    with pytest.raises(ValueError,match='explicit task6'):late.validate_plan(dict(mode='h75_samestate_bank144'))


def test_source_seal_task6_identity_and_features(tmp_path):
    import json
    root=tmp_path;directory=root/'sources';directory.mkdir();(root/'plan.json').write_text('{}')
    rows=[dict(source_index=i,source_task_tick=1200,source_physical_tick=1900,split='late_state_development') for i in range(4)]
    (directory/'records.json').write_text(json.dumps(rows))
    np.savez(directory/'source_features.npz',native_observation_user1=np.zeros((4,351),np.float32),source_world_yaw=np.zeros(4,np.float32))
    for name in ('complete_carry.pkl','executed_paths.pkl','host_state.npz'):(directory/name).write_bytes(b'identity only; no deserialization')
    files={str(p):late.coordination.sha(p) for p in directory.iterdir()}
    (directory/'seal.json').write_text(json.dumps(dict(plan_sha256=late.coordination.sha(root/'plan.json'),files=files)))
    meta,features,yaw=late.authenticate_late_sources(root)
    assert len(meta)==4 and features.shape==(4,351) and yaw.shape==(4,)
    (directory/'complete_carry.pkl').write_bytes(b'tampered')
    with pytest.raises(ValueError,match='artifact changed'):late.authenticate_late_sources(root)
