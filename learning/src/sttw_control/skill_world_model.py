"""Full-joint recurrent skill dynamics; predicts state, never robot commands.

Controller/RB histories below are learned estimates after initialization, not
measurements of the real future. qvel uses MuJoCo world linear/body angular axes.
"""
from dataclasses import dataclass
import torch
from torch import nn
from .playground_prediction import multiply


JOINT_POSITION_SCALES = {'steering_joint': .2, 'frontwheel_joint': 1., 'rearwheel_joint': 1., 'hip_joint': .3, 'knee_joint': .3}
JOINT_VELOCITY_SCALES = {'steering_joint': 1., 'frontwheel_joint': 5., 'rearwheel_joint': 5., 'hip_joint': 1., 'knee_joint': 1.}


@dataclass
class SkillWorldState:
    qpos: torch.Tensor
    qvel: torch.Tensor
    history: torch.Tensor
    contact: torch.Tensor
    phase: torch.Tensor
    hidden: torch.Tensor
    elapsed_steps: int = 0


class SkillWorldModel(nn.Module):
    def __init__(self, history_dim, context_dim, input_mean, input_scale,
                 velocity_scale, history_scale, hidden_dim=128, dt=.02):
        super().__init__()
        self.dt=dt
        self.history_dim=history_dim
        for name,value in [('input_mean',input_mean),('input_scale',input_scale),
                           ('velocity_scale',velocity_scale),('history_scale',history_scale)]:
            self.register_buffer(name,torch.as_tensor(value,dtype=torch.float32))
        size=12+11+history_dim+3+5+context_dim
        self.initial=nn.Sequential(nn.Linear(size,hidden_dim),nn.Tanh())
        self.memory=nn.GRUCell(size,hidden_dim)
        self.decoder=nn.Sequential(nn.Linear(hidden_dim,128),nn.SiLU(),nn.Linear(128,11+history_dim+3+5))

    def features(self,qpos,qvel,history,contact,phase,context):
        return (torch.cat((qpos,qvel,history,contact,phase,context),-1)-self.input_mean)/self.input_scale

    def initialize(self,qpos,qvel,controller,rb,contact,phase,context):
        history=torch.cat((controller,rb),-1)
        features=self.features(qpos,qvel,history,contact,phase,context)
        return SkillWorldState(qpos,qvel,history,contact,phase,self.initial(features))

    def step(self,state,context):
        hidden=self.memory(self.features(state.qpos,state.qvel,state.history,state.contact,state.phase,context),state.hidden)
        output=self.decoder(hidden)
        dv,dh,contact_logits,phase_logits=torch.split(output,[11,self.history_dim,3,5],-1)
        velocity=state.qvel+dv*self.velocity_scale
        mid=(state.qvel+velocity)*.5
        rv=mid[:,3:6]*self.dt
        angle=rv.norm(dim=-1,keepdim=True)
        dq=torch.cat(((angle*.5).cos(),rv*.5*torch.sinc(angle/(2*torch.pi))),-1)
        quaternion=multiply(state.qpos[:,3:7],dq)
        quaternion=quaternion/quaternion.norm(dim=-1,keepdim=True)
        pose=torch.cat((state.qpos[:,:3]+mid[:,:3]*self.dt,quaternion,
                        state.qpos[:,7:]+mid[:,6:]*self.dt),-1)
        nxt=SkillWorldState(pose,velocity,state.history+dh*self.history_scale,
                            contact_logits.sigmoid(),phase_logits.softmax(-1),hidden,state.elapsed_steps+1)
        return nxt,contact_logits,phase_logits

    def rollout(self,state,context,steps):
        poses=[state.qpos];velocities=[state.qvel];histories=[state.history]
        contacts=[];phases=[]
        for _ in range(steps):
            state,contact,phase=self.step(state,context)
            poses.append(state.qpos);velocities.append(state.qvel);histories.append(state.history)
            contacts.append(contact);phases.append(phase)
        return dict(qpos=torch.stack(poses,1),qvel=torch.stack(velocities,1),
                    history=torch.stack(histories,1),contact_logits=torch.stack(contacts,1),
                    phase_logits=torch.stack(phases,1),duration_s=steps*self.dt,next_state=state)

    def rollout_until(self,state,context,max_steps,stop_phase=2):
        """GPU-batched predicted event stop; no actual event time supplied.

        A lane without a predicted event keeps the max-horizon state and is
        explicitly marked unsupported. The full fixed-size trace has a mask.
        """
        states=[];contacts=[];phases=[]
        initial_steps=state.elapsed_steps
        for _ in range(max_steps):
            state,contact,phase=self.step(state,context)
            states.append(state);contacts.append(contact);phases.append(phase)
        phase_logits=torch.stack(phases,1)
        hit=phase_logits.argmax(-1)==stop_phase
        found=hit.any(-1)
        index=torch.where(found,hit.to(torch.int64).argmax(-1),torch.full_like(found,max_steps-1,dtype=torch.int64))
        lane=torch.arange(len(index),device=index.device)
        values={name:torch.stack([getattr(s,name) for s in states],1) for name in ['qpos','qvel','history','contact','phase','hidden']}
        next_state=SkillWorldState(**{name:value[lane,index] for name,value in values.items()},elapsed_steps=initial_steps+index+1)
        return dict(qpos=values['qpos'],qvel=values['qvel'],history=values['history'],contact_logits=torch.stack(contacts,1),phase_logits=phase_logits,
                    valid=torch.arange(max_steps,device=index.device)[None,:]<=index[:,None],event_found=found,event_step=index+1,duration_s=(index+1)*self.dt,next_state=next_state)
