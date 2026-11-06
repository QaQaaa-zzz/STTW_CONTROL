"""Exercise snapshot restore against the installed upstream delay-buffer behavior."""
import importlib.util
import sys
from pathlib import Path
from types import ModuleType,SimpleNamespace
import torch
from sttw_control.isaac_path_snapshot import FIELDS,capture,restore


def test_delayed_branch_preserves_history_and_untouched_lane():
    base=Path('/home/qy/projects/IsaacLab/source/isaaclab/isaaclab/utils/buffers')
    package=ModuleType('_snapshot_test_buffers');package.__path__=[str(base)]
    sys.modules[package.__name__]=package
    spec=importlib.util.spec_from_file_location(package.__name__+'.delay_buffer',base/'delay_buffer.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    delay=module.DelayBuffer(3,3,'cpu');delay.set_time_lag(torch.tensor([2,0,1],dtype=torch.int))
    for k in range(7):delay.compute(torch.tensor([[k],[k+100],[k+200]],dtype=torch.float32))
    env=SimpleNamespace(device='cpu',tick=1)
    for group,names in FIELDS.items():
        obj=env if group=='rollout' else SimpleNamespace()
        if group!='rollout':setattr(env,group,obj)
        for name in names:setattr(obj,name,torch.arange(3,dtype=torch.float32))
    env.middle.tick=4;env.middle.projection={'progress':torch.arange(3.)};env.middle.follower={'command':torch.ones(3,2)}
    data=SimpleNamespace(root_link_pose_w=torch.tensor([[0.,0,0,1,0,0,0]]*3),root_com_vel_w=torch.arange(18.).reshape(3,6))
    robot=SimpleNamespace(data=data)
    def write_com(value,env_ids):robot.written_com=value.clone()
    robot.write_root_com_velocity_to_sim=write_com
    plant=SimpleNamespace(robot=robot,origin=torch.zeros(3,3),roots=torch.zeros(3,13),dofs=torch.zeros(3,4,2),servos=SimpleNamespace(gain=torch.ones(3,2)))
    plant.reset_idx=lambda ids,state:None;plant.refresh=lambda:None;env.plant=plant
    env.randomization=SimpleNamespace(steer_offset=torch.arange(3.),buffers={'command':delay})
    state=capture(env,0)
    # Advance the shared pointer before restoration: unchanged lane2 must keep its own history.
    delay.compute(torch.tensor([[7.],[107.],[207.]]))
    restore(env,state,[1])
    value=delay.compute(torch.tensor([[8.],[8.],[208.]]))
    torch.testing.assert_close(value,torch.tensor([[6.],[5.],[207.]]))
    torch.testing.assert_close(robot.written_com,data.root_com_vel_w[0:1])
    torch.testing.assert_close(env.controller.eso,torch.tensor([0.,0.,2.]))
    assert delay.time_lags.tolist()==[2,2,1]
