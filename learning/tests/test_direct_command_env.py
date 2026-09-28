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
            self.status={};self.budget=Budget();self.out=tmp_path;self.spec={}
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
