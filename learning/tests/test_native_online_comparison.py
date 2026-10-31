"""Paired outcome accounting must retain gains and losses despite equal totals."""
import importlib.util
from pathlib import Path


def test_equal_totals_do_not_hide_opposite_paired_changes(monkeypatch):
    cli=Path(__file__).parents[1]/'cli'
    monkeypatch.syspath_prepend(str(cli))
    spec=importlib.util.spec_from_file_location('native_online_comparison',cli/'compare_native_online.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    success=dict(goal_arrived=True,physical_outcome='survived')
    failure=dict(goal_arrived=False,physical_outcome='contact')
    arrived_then_fell=dict(goal_arrived=True,physical_outcome='contact')
    pairs=[dict(online=success,fixed=failure),dict(online=failure,fixed=success),dict(online=arrived_then_fell,fixed=arrived_then_fell)]
    result=module.summarize_pairs(pairs)
    assert result['methods']['online']==dict(n=3,arrivals=2,survivals=1,arrival_and_survival=1)
    assert result['methods']['online']==result['methods']['fixed']
    assert result['paired_changes']['arrival']==dict(gained=1,lost=1,retained=1,neither=0)
    assert result['paired_changes']['survival']==dict(gained=1,lost=1,retained=0,neither=1)
