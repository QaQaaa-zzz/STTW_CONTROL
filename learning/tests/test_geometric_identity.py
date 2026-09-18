from dataclasses import asdict
import copy
from sttw_control.env import config_from_dict
from sttw_control.network import make_policy_identity
import json
from pathlib import Path


def test_new_defaults_preserve_frozen_timed_checkpoint_identity():
    raw=json.loads(Path('learning/configs/timed_reference_standard_range.json').read_text())
    rebuilt=asdict(config_from_dict(raw))
    model={'test':'model'}
    # Compare canonical pre-extension task, retaining all historical defaults.
    raw=copy.deepcopy(rebuilt)
    for k in ['geometric','reward_mode','shrink_tolerances','deadline_penalty','over_deadline_rate']:raw['tracking'].pop(k)
    for k in ['recovery_probability','recovery_start','conflict_start_window']:raw['timed_reference'].pop(k)
    assert make_policy_identity(model,raw,10)==make_policy_identity(model,rebuilt,10)
    changed=copy.deepcopy(rebuilt);changed['tracking']['geometric']=True
    assert make_policy_identity(model,raw,10)!=make_policy_identity(model,changed,10)
    changed=copy.deepcopy(rebuilt);changed['timed_reference']['recovery_probability']=.8
    assert make_policy_identity(model,raw,10)!=make_policy_identity(model,changed,10)
