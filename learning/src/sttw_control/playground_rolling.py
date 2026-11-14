"""Measured-state screening and delayed publication of finite motion candidates."""
from copy import copy
import numpy as np
import torch


def controller_features(middle, controller, recovering, stable_time, upper_ready_time, lock):
    state=dict(frames=middle.frames,mask=middle.mask,nominal=middle.nominal,rates=middle.rates,
        latent=middle.latent,pose=middle.pose,yaw_rate=middle.yaw_rate,lane_ticks=middle.lane_ticks,
        policy_offsets=middle.policy.offsets,policy_filtered=middle.policy.filtered,eso=controller.eso,
        disturbance=controller.disturbance,gains=controller.gains,filtered_roll_rate=controller.filtered_roll_rate,
        recovering=recovering,stable_time=stable_time,upper_ready_time=upper_ready_time,lock=lock)
    state.update({'projection_'+k:value for k,value in middle.projection.items()})
    state.update({'follower_'+k:value for k,value in middle.follower.items()})
    return state


class RollingMotionLibrary:
    def __init__(self, bank, route, clearance, device):
        with np.load(bank) as data:
            for key in data.files:setattr(self,key,torch.as_tensor(data[key],device=device))
        self.clearance=torch.as_tensor(clearance,device=device)
        self.route_bank={k:getattr(route,k) for k in ('xy','heading','curvature')}
        n=len(route.route_ids);self.current_route=route.route_ids.clone()
        self.active=torch.full((n,),-1,device=device,dtype=torch.long)
        self.failed=torch.zeros(n,device=device,dtype=torch.bool)
        # Give each environment its own route row; policy/state tensors stay resident.
        for key,value in self.route_bank.items():setattr(route,key,value[self.current_route].clone())
        route.route_ids=torch.arange(n,device=device)
        self.common_end=torch.empty((len(self.route_bank['xy']),)*2,device=device,dtype=torch.long)
        for i in range(len(self.common_end)):
            for j in range(len(self.common_end)):
                same=(self.route_bank['xy'][i]-self.route_bank['xy'][j]).abs().amax(-1)<=1e-6
                for key in ('heading','curvature'):same &= (self.route_bank[key][i]-self.route_bank[key][j]).abs()<=1e-6
                self.common_end[i,j]=same.long().cumprod(0).sum()-1

    def screen(self,q,v,history,phase,required,route,jump_start):
        features=torch.cat((q[:,1:7],q[:,8:9],q[:,10:12],v,history),1)
        index=(torch.searchsorted(self.x,q[:,0].contiguous(),right=True)-1).clamp(0,len(self.x)-2)
        weight=((q[:,0]-self.x[index])/(self.x[index+1]-self.x[index])).clamp(0,1)
        low=self.low[:,index]*(1-weight[None,:,None])+self.low[:,index+1]*weight[None,:,None]
        high=self.high[:,index]*(1-weight[None,:,None])+self.high[:,index+1]*weight[None,:,None]
        region=(torch.searchsorted(self.edges,q[:,0].contiguous(),right=True)-1).clamp(0,len(self.margin)-1)
        margin=self.margin[region]
        miss=torch.maximum(low-features,features-high).clamp_min(0)
        ratio=(miss/margin[None]).amax(-1)
        supported=(ratio<=1)&self.valid[:,index]&self.valid[:,index+1]
        supported &= (self.phase_low[:,index]==phase[None])&(self.phase_high[:,index+1]==phase[None])
        end=torch.searchsorted(route.s,required.contiguous()).clamp_max(len(route.s)-1)
        prefix=self.common_end[self.routes[:,None],self.current_route[None]]>=end[None]
        activation=(phase[None]==0)|((self.activation[:,None]-jump_start[None]).abs()<1e-5)
        valid=supported&prefix&activation&(self.clearance[:,None]>=0)
        return valid,ratio

    def select(self,*args):
        valid,ratio=self.screen(*args)
        selected=torch.where(valid,self.cost[:,None],torch.inf).argmin(0)
        return selected,valid.any(0),ratio

    def proposal(self,old,selected,accept):
        chosen=torch.where(accept,self.routes[selected],self.current_route)
        route=copy(old)
        for key,value in self.route_bank.items():setattr(route,key,value[chosen])
        return route,chosen
