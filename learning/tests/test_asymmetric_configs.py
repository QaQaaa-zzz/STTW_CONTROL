from dataclasses import asdict
from dataclasses import replace
import json
from pathlib import Path
import numpy as np

from sttw_control.env import RecoveryEnv, load_config
from sttw_control.actuator import apply_residual
from sttw_control.training import TrainingConfig


def test_main_and_control_tasks_differ_only_in_declared_priority_ratio():
    main=asdict(load_config('learning/configs/asymmetric_priority_rho34.json'))
    control=asdict(load_config('learning/configs/asymmetric_priority_rho10.json'))
    assert main['tracking'].pop('priority_ratio')==34.2
    assert control['tracking'].pop('priority_ratio')==10.0
    assert main==control


def test_formal_budget_network_optimizer_and_fixed_development_panel():
    c=TrainingConfig(**json.loads(Path('learning/configs/ppo_asymmetric_priority.json').read_text()))
    assert c.num_envs*c.rollout_steps*c.updates==6_291_456
    assert c.hidden_sizes==(128,128,128) and c.activation=='elu'
    assert c.rsl_schedule=='fixed' and c.rsl_kl_limit==.02 and c.target_kl is None
    assert tuple(c.validation_updates)==(1,24,48)
    assert c.development_scenarios==('ordinary_accel','core_left','core_right','disturbance_left')


def test_learning_zero_residual_is_point_eight_base_then_external_offset():
    c=load_config('learning/configs/asymmetric_priority_rho34.json')
    c=replace(c,preparation_seconds=0.,horizon_seconds=.025,
              timed_reference=replace(c.timed_reference,fixed_scenario='core_left'))
    env=RecoveryEnv(c);state=env.reset(3)
    state=state.replace(event=np.array([0.,1.,.25,0.,0.,0.]))
    base,action=env.prepare_action(state,np.zeros(2))
    np.testing.assert_allclose(base,[.8*float(state.base[0])+.25,.8*float(state.base[1])],atol=1e-6)
    _,command=apply_residual(state.actuator,base,action,state.measurement[2],c.actuator)
    np.testing.assert_allclose(command,base,atol=1e-6)
