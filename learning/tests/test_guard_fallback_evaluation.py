import json,hashlib,importlib.util
from pathlib import Path
import numpy as np
import pytest
from sttw_control.tolerance_coordination import choose_alpha
from sttw_control.native_goal_run import tolerance_inventory,apply_route_context_variant


def helper():
    p=Path(__file__).parents[1]/'cli/audit_native_goal_initial.py';spec=importlib.util.spec_from_file_location('_guard_stage_test',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def config(stage):return json.loads((Path(__file__).parents[1]/'configs'/f'h75_guard_fallback_{stage}.json').read_text())


def test_actual_userhalf_counterexample_prioritizes_user_despite_lower_proxy0():
    import jax.numpy as j
    args=(j.array([[1.,0.],[1.2,.1],[2.,0.]]),1.,j.array([1.,0.]),1.,j.array([.7,.6,.4]),True,True,.5)
    old=choose_alpha(*args,user_alpha=.5,task_aware_fallback=True)
    new=choose_alpha(*args,user_alpha=.5,task_aware_fallback=True,guard_fallback=True)
    assert old['proxy_cost'][0]<old['proxy_cost'][1]
    assert old['alpha']==0. and new['alpha']==.5
    assert new['degraded'] and new['latched'] and new['guard_fallback_selected'] and not new['task_aware_selected']


def test_user1_task_infeasible_uses_feasible_minproxy_half_not_highestq0():
    import jax.numpy as j
    d=choose_alpha(j.array([[1.5,0.],[1.2,0.],[2.,0.]]),1.,j.array([1.2,0.]),1.2,j.array([.7,.6,.4]),True,True,.5,user_alpha=1.,task_aware_fallback=True,guard_fallback=True)
    np.testing.assert_array_equal(d['task_mask'],[True,True,False])
    assert d['alpha']==.5 and d['degraded'] and d['latched']


def test_permission_false_qqualified_and_defaultfalse_remain_historical():
    import jax,jax.numpy as j
    targets=j.array([[1.,0.],[1.2,0.],[2.,0.]])
    fn=jax.jit(lambda enabled,q,permission:choose_alpha(targets,1.,j.array([1.2,0.]),1.2,q,permission,True,.5,user_alpha=.5,task_aware_fallback=True,guard_fallback=enabled))
    blocked=fn(True,j.array([.7,.6,.4]),False)
    assert blocked['alpha']==.5 and blocked['latched'] and not blocked['guard_fallback_selected']
    for enabled in (False,True):
        d=fn(enabled,j.array([.2,.9,.4]),True)
        assert d['alpha']==.5 and not d['latched'] and not d['degraded']
    none=choose_alpha(j.array([[2.,0.],[2.1,0.],[2.2,0.]]),1.,j.array([2.,0.]),2.,j.array([.7,.6,.4]),True,True,1.,task_aware_fallback=True,guard_fallback=True)
    assert none['alpha']==.5 and none['degraded'] and none['latched'] and not none['guard_fallback_selected']


def test_two_stages_inventory_and_contexts_keep_declared_budget():
    for name,count,budget,logical in [('extension72',72,144000,312),('regression60',60,120000,260)]:
        p=config(name);rows=tolerance_inventory(p)
        assert len(rows)==len({r['case_id'] for r in rows})==count and p['maximum_lower_ticks']==budget
        assert p['comparison_total_records']==logical and p['maximum_campaign_lower_ticks']==264000
        assert {r['method'] for r in rows}=={'guard_degraded'}
        assert helper().confirmation_initial_layout(dict(mode=p['mode'],protocol=p))==(count,3)
        plan=dict(mode=p['mode'],protocol=p,contexts=['original']);apply_route_context_variant(plan)
        if name=='extension72':assert plan['contexts']==p['route_contexts']
        else:assert plan['contexts']==['original']


def stage_fixture(root,goal_rescue=False,contact_rescue=False,loss=False):
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    (root/'evaluation').mkdir();(root/'analysis').mkdir();raw=root/'raw.npz';raw.write_bytes(b'fixed unit fixture, no physics')
    rows=[];index=[]
    conditions=[(b,r,e) for b in (1,7) for r in ('L60','R60','L120','R120') for e in ('zero','minus','plus')]
    for ci,(b,r,e) in enumerate(conditions):
        for user in (0.,.5,1.):
            def row(method):
                return dict(case_id=f'{b}/{r}/{e}/{user}/{method}',bank_index=b,route=r,event=e,effective_user_alpha=user,method=method,raw_file=str(raw),raw_sha256=sha(raw),horizons={'10':dict(goal_arrived=True,physical_outcome='survived')})
            old=row('task_aware');new=row('guard_degraded')
            if ci==0 and user==0.:
                if goal_rescue:old['horizons']['10']['goal_arrived']=False
                if contact_rescue:old['horizons']['10']['physical_outcome']='contact'
            if loss and ci==1 and user==0.:new['horizons']['10']['goal_arrived']=False
            rows.extend([old,new]);index.append(new)
            rows.extend([row('guard'),row('fixed')])
        rows.append(dict(row('speedcap'),case_id=f'{b}/{r}/{e}/1/cap'))
    # 24 conditions ×(9 historical methods+1cap+3new)=312
    assert len(rows)==312 and len(index)==72
    (root/'analysis/results.json').write_text(json.dumps(rows));(root/'evaluation/case_index.json').write_text(json.dumps(index));(root/'inventory.json').write_text(json.dumps(index))
    plan=dict(mode='guard_fallback_evaluation',protocol={'stage':'extension'},cases=72)
    (root/'plan.json').write_text(json.dumps(plan));(root/'seal.json').write_text(json.dumps(dict(plan_sha256=sha(root/'plan.json'))));(root/'evaluation/status.json').write_text(json.dumps(dict(phase='completed',completed_cases=72,active_lower_ticks=140000)));(root/'analysis/receipt.json').write_text(json.dumps(dict(all_cases=312,plan_sha256=sha(root/'plan.json'))))


def test_stage2_gate_stops_without_rescue(tmp_path):
    stage_fixture(tmp_path);gate=helper().guard_fallback_stage1_gate(tmp_path)
    assert not gate['allowed'] and gate['paired_cases']==72


@pytest.mark.parametrize('goal,contact',[(True,False),(False,True)])
def test_stage2_any_rescue_allows_fixed_regression_even_with_loss(tmp_path,goal,contact):
    stage_fixture(tmp_path,goal_rescue=goal,contact_rescue=contact,loss=True);gate=helper().guard_fallback_stage1_gate(tmp_path)
    assert gate['allowed'] and gate['total_goal_gains']+gate['total_contact_rescues']==1
    assert gate['total_goal_losses']==1
