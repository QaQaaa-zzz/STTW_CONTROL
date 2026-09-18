"""Dependency-light checks for optimizer, comparison and small delivery contracts."""
import ast,copy,io,json,math,zipfile
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pytest

SRC=Path(__file__).parents[1]/'src/sttw_control'
def extract(file,names,ns):
    tree=ast.parse((SRC/file).read_text())
    nodes=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names]
    assert len(nodes)==len(names)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),file,'exec'),ns)
    return ns

def test_new_budget_and_fixed_learning_rate_without_reward_changes():
    cls=extract('training.py',{'TrainingConfig'},{'dataclass':dataclass,'math':math})['TrainingConfig']
    root=Path(__file__).parents[1]/'configs'
    old=cls(**json.loads((root/'ppo_geometric_reward.json').read_text()))
    new=cls(**json.loads((root/'ppo_geometric_stable.json').read_text()))
    assert new.trainer=='rsl' and new.rsl_schedule=='fixed' and new.learning_rate==.0003
    assert new.rsl_kl_limit==.02 and new.training_reward_selection
    assert new.num_envs*new.rollout_steps*new.updates==old.num_envs*old.rollout_steps*old.updates==6291456
    assert new.rollout_steps==128 and new.epochs*new.num_envs*new.rollout_steps/new.minibatch_size==16
    for kw in ({'rsl_schedule':'bad'},{'rsl_kl_limit':-1},{'rsl_kl_limit':float('nan')},{'rsl_schedule':'adaptive'}):
        with pytest.raises(ValueError):cls(trainer='rsl',**kw)

def test_candidate_subset_includes_zero_first_last_and_sampled_best():
    choose=extract('selection.py',{'comparison_updates'},{})['comparison_updates']
    assert choose(range(49),1)==[0,1,16,32,48]
    assert choose(range(65),7)==[0,1,7,21,43,64]
    assert choose(range(65),1,[0,1,16,32,64])==[0,1,16,32,64]
    with pytest.raises(ValueError):choose(range(5),requested=[9])

def test_comparison_does_not_force_later_or_unqualified_winner():
    ns=extract('selection.py',{'fixed_comparison_rank'},{'np':np,'rank_tracking_candidate':lambda *a,**kw:(None,'not held')})
    rank=ns['fixed_comparison_rank']
    b={k:[False] for k in ['failed','nominal_failed','terminal_tracking_hold','nominal_terminal_tracking_hold',
                           'return_deadline_missed','nominal_return_deadline_missed']}
    b.update(episode_return=[1.],nominal_speed_rmse=[.1],nominal_radial_rmse=[.1],nominal_heading_rmse=[.1])
    bad={**b,'failed':[True],'episode_return':[100.]}
    rb,q,_=rank(b,b);rd,qd,_=rank(bad,b)
    assert rb<rd and not q and not qd
    with pytest.raises(ValueError):rank({**b,'failed':[]},b)
    assert rank({**b,'episode_return':[float('nan')]},b)[0] is None

def test_compact_archive_is_lossless_in_time_not_large_raw_state(tmp_path):
    import hashlib
    compact=extract('tracking_diagnostics.py',{'compact_review_archive'},{'Path':Path,'np':np,'json':json,'hashlib':hashlib})['compact_review_archive']
    root=tmp_path/'review';path=root/'case/alpha_0';path.mkdir(parents=True)
    n=5
    np.savez_compressed(path/'trace.npz',time=np.arange(n)*.005,reward=np.arange(n),observation=np.zeros((n,280)),qpos=np.ones((n,11)),terminated=np.array([0,0,0,0,1],bool))
    (root/'summary.json').write_text('{}')
    (path/'declaration.json').write_text('{}')
    before=(path/'trace.npz').read_bytes()
    archive=compact(root)
    with zipfile.ZipFile(archive) as z:
        names=z.namelist();assert 'case/alpha_0/trace_compact.npz' in names
        tr=np.load(io.BytesIO(z.read('case/alpha_0/trace_compact.npz')))
        np.testing.assert_array_equal(tr['reward'],np.arange(n));assert tr['terminated'][-1]
        assert 'observation' not in tr and 'qpos' not in tr
        assert json.loads(z.read('compact_manifest.json'))['traces'][0]['samples']==n
    assert (path/'trace.npz').read_bytes()==before

def test_logger_exposes_deadlines_true_kl_phase_and_learning_rate():
    import numbers
    f=extract('tensorboard_logging.py',{'scalar_values'},{'numbers':numbers,'math':math})['scalar_values']
    r={'learning_rate':.0003,'reward_components_mean_step':{'deadline':-.1,'return_overdue':-.2},
       'optimizer_audit':{'candidate_exact_kl':.03,'final_exact_kl':0.,'full_update_rolled_back':True,'accepted_minibatches':0,'kl_limit':.02},
       'sample_phase':{'fraction_0':.1,'physical_failures':3},'sampling_policy_update':16}
    tags=f(r,profile='core')
    assert tags['reward_components/deadline']==-.1 and tags['optimizer/retained_minibatches']==0
    assert tags['optimizer/candidate_exact_kl']==.03 and tags['optimizer/final_exact_kl']==0.
    assert tags['sample_phase/physical_failures']==3
    assert tags['train/sampling_policy_update']==16 and tags['train/learning_rate']==.0003


def test_guard_restores_weights_optimizer_moments_and_learning_rate():
    torch=pytest.importorskip('torch')
    from types import SimpleNamespace
    guard=extract('rsl_training.py',{'guarded_update'},{'torch':torch,'copy':copy,'math':math})['guarded_update']
    class Policy(torch.nn.Module):
        def __init__(self):
            super().__init__();self.actor=torch.nn.Linear(3,2);self.log_std=torch.nn.Parameter(torch.log(torch.full((2,),.15)))
        def act_inference(self,x):return self.actor(x)
    class Algo:
        def __init__(self):
            self.policy=Policy();self.optimizer=torch.optim.Adam(self.policy.parameters(),lr=.0003);self.learning_rate=.0003;self.schedule='fixed'
            self.obs=torch.ones(4,2,3);self.refresh()
        def refresh(self):
            with torch.no_grad():mu=self.policy.act_inference(self.obs).clone()
            self.storage=SimpleNamespace(observations=self.obs,mu=mu,sigma=torch.full_like(mu,.15))
        def update(self):
            self.optimizer.zero_grad();loss=self.policy.act_inference(self.obs).square().mean();loss.backward();self.optimizer.step()
            with torch.no_grad():self.policy.actor.bias.add_(10.)
            return dict(value=float(loss.detach()),surrogate=0.,entropy=-1.)
    a=Algo();a.update();a.refresh()  # non-empty optimizer moments
    weights=copy.deepcopy(a.policy.state_dict());optimizer=copy.deepcopy(a.optimizer.state_dict())
    _,audit=guard(a,.02)
    assert audit['full_update_rolled_back'] and audit['candidate_exact_kl']>.02 and audit['final_exact_kl']==0
    for k,v in weights.items():torch.testing.assert_close(a.policy.state_dict()[k],v,rtol=0,atol=0)
    current=a.optimizer.state_dict()
    assert optimizer['param_groups']==current['param_groups']
    for key,state in optimizer['state'].items():
        for k,v in state.items():torch.testing.assert_close(current['state'][key][k],v,rtol=0,atol=0)
    assert a.learning_rate==.0003
