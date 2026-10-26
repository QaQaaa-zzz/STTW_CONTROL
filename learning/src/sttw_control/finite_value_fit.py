"""Offline finite-window value parameterizations; no Actor is constructed."""
import copy,math
import torch
from torch import nn


def discount_budget(remaining,*,gamma=.997,steps=500):
    return -torch.expm1(torch.clamp(remaining,0,1)*(steps*math.log(gamma)))/(1-gamma)


class FiniteValue(nn.Module):
    def __init__(self,source,mode,*,rate):
        super().__init__()
        if mode not in ['inherited_refit','reset_free','time_free','time_nonpositive'] or not math.isfinite(rate) or rate<=0:raise ValueError('declared mode and positive rate required')
        self.network=copy.deepcopy(source);self.mode=mode;self.rate=float(rate)
        if mode!='inherited_refit':
            nn.init.zeros_(self.network[6].weight)
            bias=0. if mode=='reset_free' else -rate if mode=='time_free' else math.log(math.expm1(rate))
            nn.init.constant_(self.network[6].bias,bias)

    def forward(self,x):
        raw=self.network(x).squeeze(-1);budget=discount_budget(x[:,-1])
        if self.mode=='inherited_refit':return raw
        if self.mode=='reset_free':return raw-self.rate*budget
        if self.mode=='time_free':return budget*raw
        return -budget*torch.nn.functional.softplus(raw)
