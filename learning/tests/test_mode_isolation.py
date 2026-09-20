"""Two-level checks for actor-only mode specialization; no vehicle success claims."""
import ast
import copy
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import numpy as np
import pytest

ROOT=Path(__file__).parents[2]
SRC=ROOT/'learning/src/sttw_control'

def functions(path,names,ns):
    tree=ast.parse(path.read_text())
    nodes=[x for x in tree.body if isinstance(x,(ast.FunctionDef,ast.ClassDef)) and x.name in names]
    assert len(nodes)==len(names),names
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),ns)
    return ns


def source_contract():
    return json.loads((ROOT/'learning/configs/soft_budget_ecbc1.json').read_text())


def config_type():
    return functions(SRC/'training.py',{'TrainingConfig'},{'dataclass':dataclass,'math':math})['TrainingConfig']


def transfer_function():
    ns={'json':json,'_MODE_PRIORITY_KEYS':('randomize_alpha','fixed_alpha','training_alphas','validation_alphas')}
    return functions(SRC/'network.py',{'check_mode_transfer_contract'},ns)['check_mode_transfer_contract']


def plan_function():
    return functions(ROOT/'learning/cli/experiment_campaign.py',{'mode_training_documents'},{'Path':Path})['mode_training_documents']


@pytest.mark.parametrize('alpha',[0.,.5,1.])
def test_alpha_only_transfer_explicit_and_other_state_untouched(alpha):
    s=source_contract();t=copy.deepcopy(s)
    t['priority'].update(fixed_alpha=alpha,randomize_alpha=False,training_alphas=[alpha])
    assert transfer_function()(s,t)==alpha
    assert s['priority']['randomize_alpha'] is True


@pytest.mark.parametrize('field,value',[
    ('speed_reference',2.4),('roll_failure',.5),('preparation_seconds',3.),('failure_penalty',100.),
    ('alive_reward_rate',2.),('eso_start',0.),('horizon_seconds',12.),('disturbance_force_frame','world_y')])
def test_transfer_rejects_hidden_top_level_changes(field,value):
    s=source_contract();t=copy.deepcopy(s);t['priority'].update(fixed_alpha=1.,randomize_alpha=False,training_alphas=[1.])
    t[field]=value
    with pytest.raises(ValueError):transfer_function()(s,t)


@pytest.mark.parametrize('field,key,value',[
    ('tracking','roll_weight',200.),('tracking','soft_cost_bound',60.),
    ('actuator','steer_residual_scale',3.),('actuator','base_output_scale',.8),
    ('observation','history_steps',1),('timed_reference','yaw_rate_max',3.),
    ('timed_reference','fast_yaw_slew',3.),('priority','risk_gate',True)])
def test_transfer_rejects_reward_network_physics_permission_changes(field,key,value):
    s=source_contract();t=copy.deepcopy(s);t['priority'].update(fixed_alpha=1.,randomize_alpha=False,training_alphas=[1.])
    t[field][key]=value
    with pytest.raises(ValueError):transfer_function()(s,t)


@pytest.mark.parametrize('alpha',[.1,.25,.9,-1,2,None])
def test_discrete_modes_not_silently_rounded(alpha):
    s=source_contract();t=copy.deepcopy(s);t['priority'].update(fixed_alpha=alpha,randomize_alpha=False,training_alphas=[alpha])
    with pytest.raises(ValueError):transfer_function()(s,t)


def test_plan_budget_and_independent_objects_and_original_reward():
    s=source_contract();p=json.loads((ROOT/'learning/configs/ppo_soft_budget.json').read_text())
    original=copy.deepcopy((s,p));arms,budget=plan_function()(s,p,'/tmp/source','/tmp/source/checkpoints/update_0244')
    assert budget==31457280<32768000
    assert s==original[0] and p==original[1]
    assert [a['alpha'] for a in arms]==[0.,.5,1.]
    for arm in arms:
        c=config_type()(**arm['training'])
        assert c.updates==80 and c.learning_rate==.0003 and c.resume_checkpoint is None
        assert arm['task']['tracking']==s['tracking']
        transfer_function()(s,arm['task'])
    arms[0]['task']['tracking']['roll_weight']=999
    assert arms[1]['task']['tracking']['roll_weight']==100 and s['tracking']['roll_weight']==100


@pytest.mark.parametrize('updates',[0,-1,81,1000,1.2,True])
def test_budget_overrun_rejected(updates):
    with pytest.raises(ValueError):plan_function()(source_contract(),{},'/tmp/source','/tmp/source/checkpoints/x',updates=updates)


def test_engineering_scope_is_576_transitions_not_full_training():
    p=json.loads((ROOT/'learning/configs/ppo_soft_budget.json').read_text())
    arms,n=plan_function()(source_contract(),p,'/tmp/source','/tmp/source/checkpoints/x',engineering=True)
    assert n==576
    for a in arms:
        c=config_type()(**a['training']);assert not c.phase_spread_initialization


@pytest.mark.parametrize('kw',[
    {'actor_init_checkpoint':'x'}, {'actor_init_training':'x'},
    {'actor_init_checkpoint':'x','actor_init_training':'x','resume_checkpoint':'x'},
])
def test_actor_transfer_is_not_optimizer_resume(kw):
    with pytest.raises(ValueError):config_type()(trainer='rsl',**kw)


def test_actor_copy_exact_and_does_not_touch_critic_std_or_source():
    torch=pytest.importorskip('torch');torch.set_num_threads(1)
    f=functions(SRC/'rsl_training.py',{'initialize_dense_actor'},{'np':np,'torch':torch})['initialize_dense_actor']
    class Policy(torch.nn.Module):
        def __init__(self):
            super().__init__();self.actor=torch.nn.Sequential(torch.nn.Linear(5,8),torch.nn.ELU(),torch.nn.Linear(8,2))
            self.critic=torch.nn.Linear(5,1);self.log_std=torch.nn.Parameter(torch.ones(2)*math.log(.15))
    torch.manual_seed(4);source=Policy()
    dense=[m for m in source.actor.modules() if isinstance(m,torch.nn.Linear)]
    params={'params':{f'Dense_{i}':{'kernel':m.weight.detach().numpy().T.copy(),'bias':m.bias.detach().numpy().copy()} for i,m in enumerate(dense)}}
    modes=[Policy() for _ in range(3)];obs=torch.randn(16,5)
    for m in modes:
        before=copy.deepcopy(m.critic.state_dict());std=m.log_std.detach().clone()
        f(m,params)
        torch.testing.assert_close(m.actor(obs),source.actor(obs),rtol=0,atol=0)
        for k,v in before.items():torch.testing.assert_close(m.critic.state_dict()[k],v,rtol=0,atol=0)
        torch.testing.assert_close(m.log_std,std,rtol=0,atol=0)
    # Three truly separate networks and Adam states: a mode0 update cannot alter others.
    before=[copy.deepcopy(m.state_dict()) for m in modes]
    optim=[torch.optim.Adam(m.parameters(),lr=.0003) for m in modes]
    loss=modes[0].actor(obs).square().mean();loss.backward();optim[0].step()
    assert optim[0].state and not optim[1].state and not optim[2].state
    assert any(not torch.equal(before[0][k],v) for k,v in modes[0].state_dict().items())
    for j in (1,2):
        for k,v in before[j].items():torch.testing.assert_close(modes[j].state_dict()[k],v,rtol=0,atol=0)


def test_atomic_actor_copy_rejects_bad_late_layer_before_any_write():
    torch=pytest.importorskip('torch')
    f=functions(SRC/'rsl_training.py',{'initialize_dense_actor'},{'np':np,'torch':torch})['initialize_dense_actor']
    from types import SimpleNamespace
    p=SimpleNamespace(actor=torch.nn.Sequential(torch.nn.Linear(5,8),torch.nn.ELU(),torch.nn.Linear(8,2)))
    before=copy.deepcopy(p.actor.state_dict())
    data={'params':{'Dense_0':{'kernel':np.zeros((5,8)),'bias':np.zeros(8)},'Dense_1':{'kernel':np.ones((8,3)),'bias':np.zeros(3)}}}
    with pytest.raises(ValueError):f(p,data)
    for k,v in before.items():torch.testing.assert_close(p.actor.state_dict()[k],v,rtol=0,atol=0)


def test_probe_frontier_retains_negative_evidence_and_no_forced_diversity():
    f=functions(ROOT/'learning/cli/experiment_campaign.py',{'pareto_indices'},{})['pareto_indices']
    rows=[dict(speed_rmse=v,path_rmse=p,final_hold=True,failed=False,deadline_missed=False,working_guard_passed=g) for v,p,g in [(1,2,True),(2,1,True),(3,3,True),(.1,.1,False)]]
    assert f(rows)==[3] and f(rows,require_working=True)==[0,1]
    assert len(rows)==4
    rows[3]['failed']=True;assert f(rows)==[0,1]


def test_probe_metrics_do_not_confuse_final_hold_with_process_safety():
    f=functions(ROOT/'learning/cli/experiment_campaign.py',{'probe_statistics'},{})['probe_statistics']
    trace={'time':np.arange(1801)*.005,'true_forward_speed':np.full(1801,2.5),
           'reference_command':np.tile([2.5,0.],(1801,1)), 'path_features':np.zeros((1801,3)),
           'measurement':np.zeros((1801,7)),'terminated':np.zeros(1801,bool),'return_state':np.zeros((1801,8))}
    trace['measurement'][701,0]=.47;trace['true_forward_speed'][720]=2.71
    data={'trace':trace,'config':{'controller':{'dt':.005},'tracking':{'roll_working_limit':.3,'overspeed_band':.05},'horizon_seconds':9.},'summary':{'terminal_tracking_hold':True}}
    d=f(data,start_seconds=3.,alpha=1.,bias=[0,0])
    assert d['final_hold'] and not d['failed'] and not d['working_guard_passed']
    assert d['roll_peak']==.47 and np.isclose(d['overspeed_peak'],.21)
    assert d['actual_transitions']==1800


def test_no_core_physics_reward_or_actuator_source_was_changed():
    manifest=ROOT/'learning/tests/fixtures/mode_frozen_core.json'
    for name,digest in json.loads(manifest.read_text()).items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name


def test_probe_early_failure_is_retained_without_fabricated_metrics():
    ns=functions(ROOT/'learning/cli/experiment_campaign.py',{'probe_statistics','pareto_indices'},{})
    tr={'time':np.arange(3)*.005,'terminated':np.array([False,False,True]),'return_state':np.zeros((3,8))}
    d=ns['probe_statistics']({'trace':tr,'config':{'controller':{'dt':.005}}},start_seconds=3.,alpha=0.,bias=[0,0])
    assert d['failed'] and d['speed_rmse'] is None and not d['comparison_window_observed']
    assert ns['pareto_indices']([d])==[]
    json.dumps(d,allow_nan=False)


def test_all_stages_are_separate_processes_with_cpu_cuda_contract(tmp_path,monkeypatch):
    from types import SimpleNamespace
    import os,sys
    calls=[]
    def invoke(cmd,check,env):calls.append((cmd,dict(env)))
    source=tmp_path/'source';cp=source/'checkpoints/update_0244';cp.mkdir(parents=True)
    ns=functions(ROOT/'learning/cli/experiment_campaign.py',{'run_mode_recovery'},
        {'Path':Path,'json':json,'os':os,'sys':sys,'subprocess':SimpleNamespace(run=invoke),
         '__file__':str(ROOT/'learning/cli/experiment_campaign.py')})
    monkeypatch.setenv('JAX_PLATFORMS','cpu')
    args=SimpleNamespace(output=tmp_path/'out',source_training=source,source_checkpoint=cp,
        stage='all',engineering=False,updates_per_mode=80)
    result=ns['run_mode_recovery'](args)
    assert len(calls)==3 and result['stages_executed']==['probe','specialize','review']
    assert calls[0][1]['JAX_PLATFORMS']=='cpu'
    assert 'JAX_PLATFORMS' not in calls[1][1]
    assert calls[2][1]['JAX_PLATFORMS']=='cpu'
    assert all('--source-checkpoint' in x[0] for x in calls)


def test_process_guard_is_distinct_from_final_hold_and_uses_previous_reference():
    f=functions(SRC/'tracking_diagnostics.py',{'process_guard_metrics'},{'np':np})['process_guard_metrics']
    tr={'time':np.arange(4)*.005,'true_forward_speed':np.array([2.,2.,2.3,2.3]),
        'reference_command':np.array([[2.,0.],[2.3,0.],[2.3,0.],[2.3,0.]]),
        'measurement':np.zeros((4,7)),'terminated':np.zeros(4,bool)}
    c={'controller':{'dt':.005},'tracking':{'roll_working_limit':.3,'overspeed_band':.05},'horizon_seconds':.015}
    result=f(tr,c);assert result['process_overspeed_peak_m_s']==0 and result['process_working_guard_passed']
    tr['measurement'][2,0]=.47
    result=f(tr,c);assert result['process_roll_excess_seconds']==.005 and not result['process_working_guard_passed']
