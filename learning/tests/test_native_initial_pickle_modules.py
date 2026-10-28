"""Frozen sidecar pickle class identity can be loaded without vehicle physics."""
import hashlib
import importlib.util
from pathlib import Path
import pickle
import sys
import pytest


def helper():
    path=Path(__file__).parents[1]/'cli/audit_native_goal_initial.py'
    spec=importlib.util.spec_from_file_location('_audit_initial_test',path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod


def test_frozen_sidecar_unpickles_after_alias_registration(tmp_path):
    directory=tmp_path/'source';directory.mkdir()
    source=directory/'tolerance_coordination.py'
    source.write_text('from collections import namedtuple\nOnlineCarry=namedtuple("OnlineCarry", "native latched prev_alpha")\n')
    plan={'source_sha256':{'/original/tolerance_coordination.py':hashlib.sha256(source.read_bytes()).hexdigest()}}
    alias='_sidecar_pickle_test'
    try:
        mod=helper().load_frozen_pickle_module(tmp_path,plan,source.name,alias)
        raw=pickle.dumps(mod.OnlineCarry('native',False,1.))
        del sys.modules[alias]
        helper().load_frozen_pickle_module(tmp_path,plan,source.name,alias)
        result=pickle.loads(raw)
        assert tuple(result)==('native',False,1.)
        source.write_text(source.read_text()+'# drift\n')
        with pytest.raises(ValueError,match='identity mismatch'):
            helper().load_frozen_pickle_module(tmp_path,plan,source.name,alias)
    finally:sys.modules.pop(alias,None)


@pytest.mark.parametrize('mode,count,arms',[('h75_online_confirmation100',100,5),('tolerance_online120',120,6),('tolerance_online40',40,2),('tolerance_broaden120',120,5)])
def test_initial_confirmation_layout_is_defined_for_each_panel(mode,count,arms):
    audit=helper();plan=dict(mode=mode,protocol=dict(expected_cases=count,arms=list(range(arms))))
    assert audit.confirmation_initial_layout(plan)==(count,arms)
    plan['protocol']['expected_cases']+=1
    with pytest.raises(ValueError):audit.confirmation_initial_layout(plan)
