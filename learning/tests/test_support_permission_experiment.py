import json
import numpy as np
import pytest
from sttw_control.support_permission_experiment import build_inventory,execute
from sttw_control.reference_bias_runtime import batch_size
from sttw_control.preference_best import atomic_json,digest


def baseline_fixture():
    rows=[]
    for i in range(4):
        for a in [0.,.5,1.]:
            for phase,forces in [('nominal',[0]),('early',[-16,16,-32,32]),('late',[-16,16,-32,32])]:
                for force in forces:
                    event='nominal' if not force else f'{phase}_F{force:g}'
                    rows.append(dict(case_id=f'c{i}/{a}/{event}/full',context=f'c{i}',route=f'c{i}',
                        geometry_name='left',prefix_name='v1.7',alpha=a,force_N=force,
                        pulse_onset_task_tick=600 if phase=='early' else 1000,pulse_duration_s=.1,
                        world_direction=[0,1,0],requested_speed_m_s=1.1,role='full',actor_role='free_GAE',
                        mode_index=1,profile=0,causal_support=True,feedback_gain_rad=.1,goal_preview=True,
                        source_role='feedback_010',source_keep_channels=[True,True],event_id=event,replicate=0))
    return rows


def test_all108_cases_change_only_the_permission_factor_and_role():
    source=baseline_fixture();cases=build_inventory(source)
    assert len(cases)==len({c['case_id'] for c in cases})==108
    for c,b in zip(cases,source):
        assert c['support_independent_permission'] is True
        assert {k:v for k,v in c.items() if k not in ['support_independent_permission','case_id','role']}=={
            k:v for k,v in b.items() if k not in ['case_id','role']}
    with pytest.raises(ValueError):build_inventory(source[:-1])
    with pytest.raises(ValueError):build_inventory(source[:-1]+source[:1])
    with pytest.raises(ValueError):batch_size(cases,np.zeros((108,6)))
    assert batch_size(cases,np.zeros((108,6)),allowed=(108,))==108


def test_execute_keeps_failed_partial_cost_and_never_implicitly_retries(tmp_path,monkeypatch):
    import sttw_control.support_permission_experiment as experiment
    import sttw_control.reference_bias_runtime as runtime
    atomic_json(tmp_path/'plan.json',{})
    atomic_json(tmp_path/'status.json',dict(phase='prepared',stage='execution_ready'))
    (tmp_path/'shape').mkdir()
    atomic_json(tmp_path/'shape/receipt.json',dict(plan_sha256=digest(tmp_path/'plan.json'),batch=108,physics_steps=0))
    cases=build_inventory(baseline_fixture());calls=[]
    monkeypatch.setattr(experiment,'inputs',lambda root:(tmp_path,{},cases))
    class Failing:
        def __init__(self,root):pass
        def rollout(self,rows,vectors,directory,*,allowed_real_batch_sizes):
            calls.append(1);assert len(rows)==108 and allowed_real_batch_sizes==(108,)
            directory.mkdir(parents=True)
            atomic_json(directory/'status.json',dict(active_lower_ticks=137,allocated_lower_slots=21600))
            raise RuntimeError('explicit test-only failure')
    monkeypatch.setattr(runtime,'BiasRuntime',Failing)
    with pytest.raises(RuntimeError,match='test-only'):execute(tmp_path)
    status=json.loads((tmp_path/'status.json').read_text())
    assert status['phase']=='error' and status['active_lower_ticks']==137
    assert status['allocated_lower_slots']==21600 and len(calls)==1
    assert not (tmp_path/'candidate_results.json').exists()
