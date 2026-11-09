"""Opt-in solved-step evidence for event modelling; no controller reset or action edits."""
import numpy as np
import torch


class PhysicsContactRecorder:
    def __init__(self,plant,context):
        self.plant=plant;self.context=context;self.rows=[];self.original_step=plant.step
        plant.step=self.step

    def step(self,*args,**kwargs):
        result=self.original_step(*args,**kwargs)
        p=self.plant
        # Fast policy steps omit diagnostic contact refresh. Sample after the solved step.
        if not kwargs.get('refresh_diagnostics',True):
            p.sensor.update(p.physics_dt,force_recompute=True)
        body=p.rigid.clone();body[:,:,:3]-=p.origin[:,None,:]
        root=p.robot.data.root_link_state_w.clone();root[:,:3]-=p.origin
        joints=torch.stack((p.robot.data.joint_pos,p.robot.data.joint_vel),-1)
        force=p.sensor.data.net_forces_w[:,p.contact_order]
        alive,phase=self.context()
        self.rows.append((body,force.clone(),root,joints.clone(),alive.clone(),phase))
        return result

    def save(self,path):
        p=self.plant
        arrays=[torch.stack([r[i] for r in self.rows]).cpu().numpy() for i in range(5)]
        np.savez_compressed(path,time=np.arange(1,len(self.rows)+1)*p.physics_dt,
            body_state=arrays[0],contact=arrays[1],root_state_wxyz=arrays[2],joint_state=arrays[3],
            valid=arrays[4],phase=np.array([r[5] for r in self.rows]),body_names=p.bodies,joint_names=p.joints,
            physics_dt_s=p.physics_dt,semantics='Solved-step samples; valid is alive before step, terminal transition retained. Net forces do not identify terrain collider. Controller hidden states are not saved by this recorder.')
