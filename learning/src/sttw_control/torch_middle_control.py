"""H75 causal history and 50/200 Hz reference scheduler, independent of physics.

Call reset on the initial physical state, before() before each lower update,
and after() exactly once after its 25 physical steps. Controller/actuator state
must already reflect that completed lower interval when calling after().
"""
import torch
from .torch_local_lower import local_frame
from .torch_path_reference import wrap


class TorchMiddleControl:
    def __init__(self, policy, route, frame_scales, alpha, v_user):
        c=policy.config
        if (c['timing']['lower_dt_s'],c['timing']['upper_dt_s'],c['timing']['path_follower_dt_s']) != (.005,.02,.02):
            raise ValueError('frozen H75 scheduler requires 200/50 Hz timing')
        if len(frame_scales)!=20 or c['network']['history_frames']!=16:
            raise ValueError('frozen H75 requires 16 by 20 history')
        self.policy=policy;self.route=route;self.alpha=alpha;self.v_user=v_user
        self.scales=torch.as_tensor(frame_scales,device=alpha.device,dtype=torch.float32)
        self.context_scales=torch.tensor([f['scale'] for f in c['network']['context_fields']],device=alpha.device)
        self.frames=torch.zeros(len(alpha),16,20,device=alpha.device)
        self.mask=torch.zeros(len(alpha),16,device=alpha.device)

    def record(self,measurement,speed,previous_reference,previous_command,applied,controller):
        frame=local_frame(measurement,speed,self.nominal,previous_reference,previous_command,
            applied,self.yaw_rate,self.rates,controller)
        self.frames=torch.cat((self.frames[:,1:],frame[:,None]),dim=1)
        self.mask=torch.cat((self.mask[:,1:],torch.ones_like(self.mask[:,:1])),dim=1)

    def observation(self,pose):
        p=self.projection;s=p['progress'];r=self.route;f=self.follower
        eh=wrap(p['heading']-pose[:,2]);offset=self.policy.offsets;filtered=self.policy.filtered
        context=torch.stack((self.alpha,p['cross_track'],torch.sin(eh),torch.cos(eh),
            r.at(s)[2],r.at(s+.5)[2],r.at(s+1)[2],f['preview_body'][:,0],f['preview_body'][:,1],
            self.v_user,offset[:,0],offset[:,1],filtered[:,0],filtered[:,1],f['lookahead']),dim=-1)
        obs=torch.cat(((self.frames/self.scales).flatten(1),self.mask,context/self.context_scales),dim=-1)
        return obs.clamp(-5,5),~torch.isfinite(obs).all(-1)

    def reset(self,measurement,pose,speed,raw,previous_reference,previous_command,controller):
        self.policy.reset(slice(None));self.frames.zero_();self.mask.zero_()
        self.tick=0;self.pose=pose.clone();self.yaw_rate=torch.zeros_like(speed)
        self.projection=self.route.project(pose[:,:2],torch.zeros_like(speed),torch.zeros_like(speed))
        self.follower=self.route.pursuit(self.projection['progress'],pose,speed,self.v_user)
        self.nominal,self.rates=self.route.publish(raw,self.follower['command'])
        self.fault=self.follower['numerical_fault'].clone();self.domain_exit=self.follower['invalid'].clone()
        self.record(measurement,speed,previous_reference,previous_command,torch.zeros_like(raw),controller)

    def before(self,pose,zero_middle=False,noise_std=None):
        if self.tick%4==0:
            self.obs,fault=self.observation(pose)
            self.latent=torch.zeros_like(self.nominal) if zero_middle else self.policy.latent(self.obs)
            if noise_std is not None:
                self.latent=self.latent+noise_std*torch.randn_like(self.latent)
            self.fault|=fault|~torch.isfinite(self.latent).all(-1)
        governed,flags=self.policy.correct(self.latent,self.nominal)
        self.fault|=flags['policy_fault']
        return governed

    def after(self,measurement,pose,speed,governed,command,applied,controller):
        self.yaw_rate=wrap(pose[:,2]-self.pose[:,2])/.005
        self.projection=self.route.project(pose[:,:2],self.projection['progress'],torch.linalg.vector_norm(pose[:,:2]-self.pose[:,:2],dim=-1))
        self.pose=pose.clone();self.tick+=1
        if self.tick%4==0:
            self.follower=self.route.pursuit(self.projection['progress'],pose,speed,self.v_user)
        self.nominal,self.rates=self.route.publish(self.nominal,self.follower['command'])
        self.fault|=self.follower['numerical_fault']|~torch.isfinite(measurement).all(-1)
        self.domain_exit|=self.follower['invalid']&~self.follower['numerical_fault']
        if self.tick%4==0:self.record(measurement,speed,governed,command,applied,controller)
