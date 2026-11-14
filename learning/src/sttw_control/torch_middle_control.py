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

    def reset(self,measurement,pose,speed,raw,previous_reference,previous_command,controller,initial_progress=None):
        self.policy.reset(slice(None));self.frames.zero_();self.mask.zero_()
        self.tick=0;self.lane_ticks=None;self.pose=pose.clone();self.yaw_rate=torch.zeros_like(speed)
        self.projection=self.route.project(pose[:,:2],torch.zeros_like(speed) if initial_progress is None else initial_progress,torch.zeros_like(speed))
        self.follower=self.route.pursuit(self.projection['progress'],pose,speed,self.v_user)
        self.nominal,self.rates=self.route.publish(raw,self.follower['command'])
        self.fault=self.follower['numerical_fault'].clone();self.domain_exit=self.follower['invalid'].clone()
        self.record(measurement,speed,previous_reference,previous_command,torch.zeros_like(raw),controller)

    def reset_lanes(self,indices,measurement,pose,speed,raw,previous_reference,previous_command,controller,initial_progress=None):
        """Initialize returning lanes at a policy boundary without resetting peers."""
        self.policy.reset(indices);self.frames[indices]=0.;self.mask[indices]=0.
        self.pose[indices]=pose[indices];self.yaw_rate[indices]=0.
        projection=self.route.project(pose[:,:2],torch.zeros_like(speed) if initial_progress is None else initial_progress,torch.zeros_like(speed))
        follower=self.route.pursuit(projection['progress'],pose,speed,self.v_user)
        nominal,rates=self.route.publish(raw,follower['command'])
        for key,value in projection.items():self.projection[key][indices]=value[indices]
        for key,value in follower.items():self.follower[key][indices]=value[indices]
        self.nominal[indices]=nominal[indices];self.rates[indices]=rates[indices]
        self.fault[indices]=follower['numerical_fault'][indices]
        self.domain_exit[indices]=follower['invalid'][indices]
        frame=local_frame(measurement,speed,self.nominal,previous_reference,previous_command,
            torch.zeros_like(raw),self.yaw_rate,self.rates,controller)
        self.frames[indices,-1]=frame[indices];self.mask[indices,-1]=1.

    def reset_mask(self, mask, measurement, pose, speed, controller):
        """Reset returning GPU lanes without dynamic indexing or host synchronization."""
        choose=lambda new,old:torch.where(mask.reshape((-1,)+(1,)*(old.ndim-1)),new,old)
        self.policy.offsets=choose(0.,self.policy.offsets)
        self.policy.filtered=choose(0.,self.policy.filtered)
        self.frames=choose(0.,self.frames);self.mask=choose(0.,self.mask)
        self.pose=choose(pose,self.pose);self.yaw_rate=choose(0.,self.yaw_rate)
        selected=self.route.xy[self.route.route_ids] if self.route.batched else self.route.xy[None]
        nearest=(selected-pose[:,None,:2]).square().sum(-1).argmin(-1)
        projection=self.route.project(pose[:,:2],self.route.s[nearest],torch.zeros_like(speed))
        follower=self.route.pursuit(projection['progress'],pose,speed,self.v_user)
        raw=torch.stack((speed,measurement[:,2]),-1)
        nominal,rates=self.route.publish(raw,follower['command'])
        for key,value in projection.items():self.projection[key]=choose(value,self.projection[key])
        for key,value in follower.items():self.follower[key]=choose(value,self.follower[key])
        self.nominal=choose(nominal,self.nominal);self.rates=choose(rates,self.rates)
        self.fault=choose(follower['numerical_fault'],self.fault);self.domain_exit=choose(follower['invalid'],self.domain_exit)
        controller.eso=choose(torch.cat((measurement[:,[2,0,1]],torch.zeros_like(speed[:,None])),1),controller.eso)
        controller.filtered_roll_rate=choose(measurement[:,1],controller.filtered_roll_rate)
        controller.disturbance=choose(0.,controller.disturbance);controller.gains=choose(controller.initial_gains,controller.gains)
        previous=torch.stack((measurement[:,3],measurement[:,5]),-1)
        frame=local_frame(measurement,speed,nominal,raw,previous,torch.zeros_like(raw),self.yaw_rate,rates,controller)
        self.frames[:,-1]=choose(frame,self.frames[:,-1]);self.mask[:,-1]=choose(1.,self.mask[:,-1])
        self.lane_ticks=torch.where(mask,0,self.lane_ticks)

    def update_plan(self, route, committed_until):
        """Publish a same-grid route at a 20 ms boundary without a skill reset.

        The caller supplies a smooth route in the original world frame. Its
        committed prefix includes the current pursuit preview and next tick's
        forward motion. Controller, policy, history and clock state survive.
        """
        if self.tick % 4:
            raise ValueError('publish plans at a 20 ms boundary')
        old=self.route
        if old.batched != route.batched or not torch.equal(old.s,route.s) or (
                old.batched and not torch.equal(old.route_ids,route.route_ids)):
            raise ValueError('plan updates require the same progress grid and lane mapping')
        required=self.projection['progress']+self.follower['lookahead']+self.v_user*.02
        if not torch.all(committed_until>=required.clamp_max(old.s[-1])):
            raise ValueError('committed prefix must cover the pursuit preview')
        # Include the segment endpoint used when interpolating the prefix boundary.
        end=torch.searchsorted(old.s,committed_until.contiguous()).clamp_max(len(old.s)-1)
        prefix=torch.arange(len(old.s),device=old.s.device)[None,:]<=end[:,None]
        for key in ('xy','heading','curvature'):
            a=getattr(old,key);b=getattr(route,key)
            a=a[old.route_ids] if old.batched else a[None]
            b=b[route.route_ids] if route.batched else b[None]
            mask=prefix[...,None] if key=='xy' else prefix
            if not torch.all(torch.where(mask,(a-b).abs()<=1e-6,True)):
                raise ValueError('plan update changed the committed prefix')
        self.route=route

    def before(self,pose,zero_middle=False,noise_std=None,zero_mask=None,latent_override=None):
        if self.lane_ticks is not None or self.tick%4==0:
            self.obs,fault=self.observation(pose)
            candidate=torch.zeros_like(self.nominal) if zero_middle else self.policy.latent(self.obs)
            self.latent=candidate if self.lane_ticks is None else torch.where((self.lane_ticks%4==0)[:,None],candidate,self.latent)
            if latent_override is not None:self.latent=latent_override
            if noise_std is not None:
                self.latent=self.latent+noise_std*torch.randn_like(self.latent)
            if zero_mask is not None:self.latent=torch.where(zero_mask[:,None],0.,self.latent)
            self.fault|=fault|~torch.isfinite(self.latent).all(-1)
        governed,flags=self.policy.correct(self.latent,self.nominal)
        self.flags=flags
        self.fault|=flags['policy_fault']
        return governed

    def after(self,measurement,pose,speed,governed,command,applied,controller):
        self.yaw_rate=wrap(pose[:,2]-self.pose[:,2])/.005
        self.projection=self.route.project(pose[:,:2],self.projection['progress'],torch.linalg.vector_norm(pose[:,:2]-self.pose[:,:2],dim=-1))
        self.pose=pose.clone();self.tick+=1
        if self.lane_ticks is not None:
            self.lane_ticks+=1
            refresh=self.lane_ticks%4==0
            candidate=self.route.pursuit(self.projection['progress'],pose,speed,self.v_user)
            for key,value in candidate.items():
                old=self.follower[key];self.follower[key]=torch.where(refresh.reshape((-1,)+(1,)*(old.ndim-1)),value,old)
        elif self.tick%4==0:
            self.follower=self.route.pursuit(self.projection['progress'],pose,speed,self.v_user)
        self.nominal,self.rates=self.route.publish(self.nominal,self.follower['command'])
        self.fault|=self.follower['numerical_fault']|~torch.isfinite(measurement).all(-1)
        self.domain_exit|=self.follower['invalid']&~self.follower['numerical_fault']
        if self.lane_ticks is not None:
            frames,mask=self.frames,self.mask
            self.record(measurement,speed,governed,command,applied,controller)
            self.frames=torch.where(refresh[:,None,None],self.frames,frames)
            self.mask=torch.where(refresh[:,None],self.mask,mask)
        elif self.tick%4==0:self.record(measurement,speed,governed,command,applied,controller)
