import copy
from types import SimpleNamespace

import jax.numpy as j
from flax import struct

from sttw_control.forward_entry_contexts import prefix_inventory,prefix_config,task_relative_state


def test_prefix_declaration_is_fixed_and_config_isolation():
    specs=prefix_inventory()
    assert len(specs)==6 and len({s['name'] for s in specs})==6
    assert sum(s['lower_ticks'] for s in specs)==1920
    config={'path':{'other':'retain'},'follower':{'retain':True}}
    original=copy.deepcopy(config)
    proposed=prefix_config(config,specs[0])
    assert config==original
    assert proposed['path']['other']=='retain'
    assert proposed['follower']==config['follower']
    assert proposed['path']['turn_angle_rad']>0


def test_relative_task_clock_preserves_all_dynamic_carry():
    @struct.dataclass
    class State:
        tick:object
        physical:object
        lower:object
        history:object
        filtered:object
        fault:object
        domain_exit:object
    state=State(j.int32(320),SimpleNamespace(physical_tick=1020),object(),object(),object(),True,True)
    relative=task_relative_state(state)
    assert int(relative.tick)==0 and int(state.tick)==320
    for name in State.__dataclass_fields__:
        if name!='tick':assert getattr(relative,name) is getattr(state,name)
    assert relative.fault and relative.domain_exit
