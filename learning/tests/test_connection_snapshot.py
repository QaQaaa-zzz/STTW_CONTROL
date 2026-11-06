from types import SimpleNamespace
import torch
from sttw_control.isaac_path_rollout import IsaacPathRollout


def fixture(n):
    env=IsaacPathRollout.__new__(IsaacPathRollout);env.n=n;env.device='cpu';env.randomization=None;env.switch=None
    env.upper_sha256='upper';env.lower_sha256='lower';env.cc={'dt':.005};env.tick=7
    env.controller=SimpleNamespace();env.lower=SimpleNamespace();env.policy=SimpleNamespace(channel_intervention_mode='hold');env.middle=SimpleNamespace(tick=28)
    groups=[(env.controller,['eso','disturbance','gains']),(env.lower,['frames','mask','previous_applied','yaw_rate']),
        (env.policy,['filtered','offsets','execution_mode','hold_channels']),
        (env.middle,['frames','mask','pose','yaw_rate','nominal','rates','fault','domain_exit']),
        (env,['previous_reference','previous_command','measurement','alive','task_tick','physical_tick','since_reset_tick','alpha','speed','switched','switch_tick','preference_progress'])]
    for obj,names in groups:
        for i,name in enumerate(names):setattr(obj,name,torch.arange(n*2).reshape(n,2).float()+i+1)
    env.middle.projection={'progress':torch.arange(n).float()};env.middle.follower={'command':torch.ones(n,2)}
    root=torch.zeros(n,13);root[:,0]=torch.arange(n)*20+3;root[:,6]=1;root[:,7]=2
    env.plant=SimpleNamespace(pendulum_rest=None,roots=root,origin=torch.stack((torch.arange(n)*20,torch.zeros(n),torch.zeros(n)),-1),dofs=torch.ones(n,4,2),joints=['a','b','c','d'],contract={'exported_xml_sha256':'asset'})
    def restore(ids,state):env.plant.restored={k:v.clone() for k,v in state.items()}
    env.plant.reset_idx=restore
    return env


def test_connection_snapshot_preserves_histories_and_translates_world_origin():
    source=fixture(2);snapshot=source.capture_connection_state(1);target=fixture(3)
    expected=source.lower.frames[1].clone();source.lower.frames.zero_()
    target.restore_connection_state(snapshot)
    torch.testing.assert_close(target.lower.frames,expected.expand(3,-1))
    torch.testing.assert_close(target.controller.eso,source.controller.eso[1].expand(3,-1))
    torch.testing.assert_close(target.plant.restored['root_pose'],torch.tensor([3.,0.,0.,1.,0.,0.,0.]))
    assert target.middle.tick==28 and target.tick==7
    torch.testing.assert_close(target.policy.filtered,source.policy.filtered[1].expand(3,-1))
