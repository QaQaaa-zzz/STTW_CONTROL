"""Dependency-light safety/interface contracts. These are NOT simulated outcomes."""
import ast,copy,hashlib,json,math
from dataclasses import dataclass,replace,asdict
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

ROOT=Path(__file__).parents[1];SRC=ROOT/'src/sttw_control'
def extract(name,names,ns):
    tree=ast.parse((SRC/name).read_text())
    nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
    assert len(nodes)==len(names)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),name,'exec'),ns)
    return ns

def test_actor_transfer_allows_only_alpha_sampling_and_never_mutates_source():
    f=extract('network.py',{'alpha_sampling_compatible'},{})['alpha_sampling_compatible']
    source={'priority':{'fixed_alpha':.5,'training_alphas':[0,.5,1],'randomize_alpha':True,'risk_gate':False},'reward':8,'base':1}
    saved=copy.deepcopy(source);target=copy.deepcopy(source)
    target['priority'].update(fixed_alpha=0.,training_alphas=[0.],randomize_alpha=False,validation_alphas=[0,.5,1])
    assert f(source,target) and source==saved
    for key in ('reward','base'):
        bad=copy.deepcopy(target);bad[key]=0;assert not f(source,bad)
    bad=copy.deepcopy(target);bad['priority']['risk_gate']=True;assert not f(source,bad)
    assert not f({},target)

def test_resolve_actor_fails_closed_instead_of_selecting_last(tmp_path):
    f=extract('network.py',{'resolve_actor_checkpoint'},{'Path':Path,'json':json})['resolve_actor_checkpoint']
    with pytest.raises(ValueError):f(None)
    last=tmp_path/'checkpoints/update_0999';last.mkdir(parents=True)
    (last/'actor.msgpack').write_bytes(b'x');(last/'identity.json').write_text('{}')
    with pytest.raises(ValueError):f(tmp_path)
    (tmp_path/'best_model.json').write_text(json.dumps({'checkpoint':'/old/run/checkpoints/update_0999'}))
    assert f(tmp_path)==last.resolve()

@dataclass(frozen=True)
class P:
    fixed_alpha:float=.5
    randomize_alpha:bool=True
    training_alphas:tuple=(0.,.5,1.)
    validation_alphas:tuple=(0.,.5,1.)
@dataclass(frozen=True)
class A:base_output_scale:float=1.
@dataclass(frozen=True)
class R:objective:str='soft_budget_v1'
@dataclass(frozen=True)
class Task:
    priority:P=P()
    actuator:A=A()
    tracking:R=R()
    preparation_base_output_scale:float=1.

def test_expert_plan_exact_discrete_modes_same_reward_and_bounded_total():
    cls=extract('training.py',{'TrainingConfig'},{'dataclass':dataclass,'math':math})['TrainingConfig']
    conf=cls(**json.loads((ROOT/'configs/ppo_independent_modes.json').read_text()),initialize_actor='/explicit/actor')
    plan=extract('training.py',{'plan_experts'},{})['plan_experts']
    task=Task();rows=plan(task,conf)
    assert [r[0] for r in rows]==[0.,.5,1.]
    for alpha,c,t in rows:
        assert c.priority.fixed_alpha==alpha and not c.priority.randomize_alpha
        assert c.priority.training_alphas==(alpha,)
        assert c.tracking is task.tracking and c.actuator is task.actuator
        assert t is conf and t.seed==66 and t.updates==80
    assert 3*conf.num_envs*conf.rollout_steps*conf.updates==31457280
    with pytest.raises(ValueError):plan(task,replace(conf,initialize_actor=None))
    with pytest.raises(ValueError):plan(replace(task,actuator=A(.5)),conf)
    with pytest.raises(ValueError):cls(trainer='jax',initialize_actor='x')
    with pytest.raises(ValueError):cls(trainer='rsl',initialize_actor='x',resume_checkpoint='y')

@pytest.mark.parametrize('bias',[-1.,-.5,0.,.5,1.])
def test_candidate_actions_bounded_finite_and_exactly_end_at_window(bias):
    f=extract('evaluation.py',{'candidate_offset'},{'np':np})['candidate_offset']
    for k in range(-1,102):
        action,flag=f([.9,-.9],[bias,-bias],k,100)
        assert np.max(np.abs(action))<=1
        if k<0 or k>=100:np.testing.assert_array_equal(action,[.9,-.9]);assert not flag
    np.testing.assert_array_equal(f([.1,-.2],[0,0],50,100)[0],[.1,-.2])
    with pytest.raises(ValueError):f([np.nan,0],[0,0],0,100)
    with pytest.raises(ValueError):f([0,0],[0,0],0,0)

def test_candidate_report_keeps_failure_and_does_not_force_distinct_winners():
    f=extract('evaluation.py',{'candidate_search_report'},{})['candidate_search_report']
    def row(name,v,y,task=True,work=True,finite=True):
        return dict(candidate=name,speed_rmse=v,path_rmse=y,task_qualified=task,work_envelope_qualified=work,
                    finite=finite,steps=1200,scores={'0.0':-v-y,'0.5':-v-y,'1.0':-v-y})
    result=f([row('good',.02,.03),row('worse',.1,.2),row('failed',0,0,False,False)])
    assert result['count']==3 and result['actual_search_transitions']==3600
    assert result['work_pareto_candidates']==['good'] and set(result['work_reward_winners'].values())=={'good'}
    result=f([row('over_roll',.01,.01,work=False)])
    assert all(x is None for x in result['work_reward_winners'].values()) and 'does not prove' in result['decision']

def test_actor_only_copy_and_separate_adam_have_zero_cross_expert_mutation():
    torch=pytest.importorskip('torch')
    init=extract('rsl_training.py',{'initialize_actor_weights'},{'torch':torch,'np':np})['initialize_actor_weights']
    class Policy(torch.nn.Module):
        def __init__(self):
            super().__init__();self.actor=torch.nn.Sequential(torch.nn.Linear(4,8),torch.nn.ELU(),torch.nn.Linear(8,2))
            self.critic=torch.nn.Linear(4,1);self.log_std=torch.nn.Parameter(torch.full((2,),math.log(.15)))
    rng=np.random.default_rng(12)
    source={'params':{'Dense_0':{'kernel':rng.normal(size=(4,8)).astype(np.float32),'bias':rng.normal(size=8).astype(np.float32)},
                      'Dense_1':{'kernel':rng.normal(size=(8,2)).astype(np.float32),'bias':rng.normal(size=2).astype(np.float32)}}}
    experts=[Policy() for _ in range(3)]
    saved_critics=[copy.deepcopy(p.critic.state_dict()) for p in experts]
    for p in experts:init(p,source)
    x=torch.randn(6,4)
    for p,s in zip(experts,saved_critics):
        for k,v in s.items():torch.testing.assert_close(p.critic.state_dict()[k],v,rtol=0,atol=0)
        torch.testing.assert_close(p.actor(x),experts[0].actor(x),rtol=0,atol=0)
    before=[copy.deepcopy(p.state_dict()) for p in experts]
    optim=[torch.optim.Adam(p.parameters(),lr=.0003) for p in experts]
    optim[0].zero_grad();experts[0].actor(x).square().mean().backward();optim[0].step()
    assert not torch.equal(experts[0].actor[0].weight,before[0]['actor.0.weight'])
    for i in (1,2):
        for k,v in before[i].items():torch.testing.assert_close(experts[i].state_dict()[k],v,rtol=0,atol=0)
        assert not optim[i].state
    bad=copy.deepcopy(source);bad['params']['Dense_1']['bias'][0]=np.nan
    saved=copy.deepcopy(experts[1].state_dict())
    with pytest.raises(ValueError):init(experts[1],bad)
    for k,v in saved.items():torch.testing.assert_close(experts[1].state_dict()[k],v,rtol=0,atol=0)

def test_compact_review_retains_all_csv_and_does_not_touch_originals(tmp_path):
    import zipfile
    pack=extract('tracking_diagnostics.py',{'pack_mode_review'},{'Path':Path,'json':json})['pack_mode_review']
    root=tmp_path/'review';root.mkdir();(root/'status.json').write_text('{"complete":true}')
    raw=root/'trace.npz';raw.write_bytes(b'preserved');(root/'steps.csv').write_text('t,r\n0,1\n1,-200\n')
    archive=pack(root)
    with zipfile.ZipFile(archive) as z:
        assert z.read('steps.csv').decode().endswith('1,-200\n') and 'trace.npz' not in z.namelist()
        assert json.loads(z.read('transfer_manifest.json'))['time_downsampling'] is False
    assert raw.read_bytes()==b'preserved'
