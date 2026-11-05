"""Convert saved MuJoCo physical states to the existing Lab reset contract.

Only kinematics are evaluated. Contact/solver caches are not transferred.
"""
import numpy as np
import mujoco


def export_original_policy(checkpoint,output):
    """Save tensors only, without the original NumPy RNG/optimizer payload.

    Call in the original checkpoint environment. Lab can then load the weights
    without requiring the original NumPy version or restoring an old optimizer.
    """
    import torch,hashlib
    from pathlib import Path
    checkpoint=Path(checkpoint);saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    torch.save(dict(policy={key:value.detach().cpu() for key,value in saved['policy'].items()},source_checkpoint=str(checkpoint.resolve()),source_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),source_update=saved['update'],initialization='Actor/Critic/log_std weights only; optimizer and RNG not restored'),output)


def export_physical_states(bank,bundle,indices):
    model=bundle.model;data=mujoco.MjData(model)
    names=['rearsuspension_joint','steering_joint','rearwheel_joint','frontwheel_joint']
    joints=[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,name) for name in names]
    states=[]
    for i in indices:
        data.qpos[:]=bank.data.qpos[i];data.qvel[:]=bank.data.qvel[i];mujoco.mj_forward(model,data)
        velocity=np.zeros(6);mujoco.mj_objectVelocity(model,data,mujoco.mjtObj.mjOBJ_BODY,bundle.chassis,velocity,0)
        states.append(dict(root_pose=np.r_[data.xpos[bundle.chassis],data.xquat[bundle.chassis]],root_velocity=np.r_[velocity[3:],velocity[:3]],joint_position=data.qpos[model.jnt_qposadr[joints]].copy(),joint_velocity=data.qvel[model.jnt_dofadr[joints]].copy(),eso=np.asarray(bank.controller.eso[i]),disturbance=np.asarray(bank.controller.disturbance[i]),gains=np.asarray(bank.controller.gains[i]),previous_command=np.asarray(bank.actuator.previous[i]),previous_reference=np.asarray(bank.governor.current_reference[i]),physical_tick=np.asarray(bank.physical_tick[i]),raw=np.asarray(bank.raw[i])))
    return {**{key:np.stack([state[key] for state in states]) for key in states[0]},'joint_names':np.array(names)}
