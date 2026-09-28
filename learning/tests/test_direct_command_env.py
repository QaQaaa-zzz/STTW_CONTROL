from importlib.util import find_spec

def test_independent_task_environment_exists():
    assert find_spec('sttw_control.direct_command_env') is not None

def test_budget_persists(tmp_path):
    import json
    from pathlib import Path
    from sttw_control.direct_command_budget import ComputeBudget,BudgetStop
    spec=json.loads((Path(__file__).parents[1]/'configs/STTW_Direct_Command_V3.json').read_text())
    b=ComputeBudget(tmp_path,spec);b.reserve(b.tick_limit,'reservation only')
    try:ComputeBudget(tmp_path,spec).reserve(1,'must reject')
    except BudgetStop:pass
    else:raise AssertionError('budget reset')

def test_nonfinite_optimizer_stops_before_review(monkeypatch,tmp_path):
    import pytest
    from contextlib import nullcontext
    from sttw_control import direct_command_training as training
    from sttw_control import direct_command_reporting as reporting
    called=[]
    class Budget:
        total_limit=1800
        def measure(self,*args):return nullcontext()
    class Campaign:
        def __init__(self,*args):
            self.status={};self.budget=Budget();self.out=tmp_path
            self.spec={'ppo':{'default_updates':20},'budget':{'additional_compute_wall_seconds':1800}}
        def initialize(self):pass
        def preflight(self):pass
        def train(self,phase,updates):
            if phase=='pilot':self.status['training_stop_reason']='nonfinite_optimizer_epoch_rolled_back'
            return object(),1
        def review(self,*args):called.append('review')
        def _status(self,**kwargs):self.status.update(kwargs)
    monkeypatch.setattr(training,'Campaign',Campaign)
    monkeypatch.setattr(training,'notify',lambda *args:None)
    monkeypatch.setattr(reporting,'generate_report',lambda *args:called.append('saved_report'))
    with pytest.raises(RuntimeError,match='no review'):
        training.run('unused',tmp_path)
    assert called==['saved_report']


def test_explicit_500_profile_and_method_freeze(tmp_path):
    import json
    from pathlib import Path
    import pytest
    from sttw_control.direct_command_training import load_spec
    p=Path(__file__).parents[1]/'configs/direct_command_500.json'
    spec=load_spec(p)
    assert spec['ppo']['default_updates']==500
    assert spec['budget']['default_control_transitions_upper']==131072000
    changed=json.loads(p.read_text());changed['ppo']['actor_learning_rate']*=2
    path=tmp_path/'changed.json';path.write_text(json.dumps(changed))
    with pytest.raises(ValueError):load_spec(path)
    changed=json.loads(p.read_text());changed['budget']['default_policy_transitions']=1310720
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError):load_spec(path)


def test_old_profile_cannot_silently_train_500():
    from pathlib import Path
    import pytest
    from sttw_control.direct_command_training import load_spec,validate_run_limits
    root=Path(__file__).parents[1]/'configs'
    old=load_spec(root/'STTW_Direct_Command_V3.json')
    new=load_spec(root/'direct_command_500.json')
    with pytest.raises(ValueError):validate_run_limits(old,500,16200)
    assert validate_run_limits(new,None,None)==(500,16200)
    with pytest.raises(ValueError):validate_run_limits(new,501,16200)
