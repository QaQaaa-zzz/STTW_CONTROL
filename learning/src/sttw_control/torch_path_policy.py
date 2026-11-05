"""Frozen H75 alpha-modulated Actor and 200 Hz reference correction."""
import hashlib
import io
from pathlib import Path
import numpy as np
import torch


class TorchPathPolicy:
    def __init__(self, actor, sha256, config, count, device='cpu'):
        data = Path(actor).read_bytes()
        if hashlib.sha256(data).hexdigest() != sha256:
            raise ValueError('path actor identity mismatch')
        if config['network']['architecture'] != 'alpha_modulated_v1':
            raise ValueError('H75 alpha-modulated architecture required')
        shapes = [(351,128),(128,128),(128,64),(64,2)]
        expected = {f'Dense_{i}_{name}' for i in range(4) for name in ('kernel','bias')}
        expected |= {'gain_2','shift_2','gain_3','shift_3'}
        archive = np.load(io.BytesIO(data), allow_pickle=False)
        if set(archive.files) != expected or any(
                archive[f'Dense_{i}_kernel'].shape != shape or archive[f'Dense_{i}_bias'].shape != (shape[1],)
                for i,shape in enumerate(shapes)) or any(archive[name].shape != (width,)
                for name,width in [('gain_2',128),('shift_2',128),('gain_3',64),('shift_3',64)]):
            raise ValueError('H75 parameter shapes mismatch')
        self.weights = {k:torch.tensor(archive[k],device=device,dtype=torch.float32) for k in archive.files}
        self.config = config
        self.filtered = torch.zeros(count,2,device=device)
        self.offsets = torch.zeros_like(self.filtered)
        c=config['residual_interface'];dt=config['timing']['lower_dt_s']
        self.decay=torch.exp(-dt*torch.ones(2,device=device)/torch.tensor(c['filter_tau_s'],device=device))
        self.increment_low=dt*torch.tensor([-c['correction_speed_down_m_s2'],-c['correction_steer_rate_rad_s']],device=device)
        self.increment_high=dt*torch.tensor([c['correction_speed_up_m_s2'],c['correction_steer_rate_rad_s']],device=device)
        self.low=torch.tensor([c['speed_reference_bounds_m_s'][0],-c['steer_reference_bound_rad']],device=device)
        self.high=torch.tensor([c['speed_reference_bounds_m_s'][1],c['steer_reference_bound_rad']],device=device)

    def reset(self,indices):
        self.filtered[indices]=0
        self.offsets[indices]=0

    @torch.no_grad()
    def latent(self,obs):
        x=obs;q=2*obs[:,336:337]-1;w=self.weights
        for i in range(3):
            x=x@w[f'Dense_{i}_kernel']+w[f'Dense_{i}_bias']
            if i>0:x=(1+q*w[f'gain_{i+1}'])*x+q*w[f'shift_{i+1}']
            x=torch.nn.functional.elu(x)
        return x@w['Dense_3_kernel']+w['Dense_3_bias']

    @torch.no_grad()
    def correct(self,z,nominal):
        c=self.config['residual_interface'];a=torch.tanh(z)
        target=torch.stack((a[:,0]*torch.where(a[:,0]>=0,c['speed_positive_scale_m_s'],c['speed_negative_scale_m_s']),a[:,1]*c['steer_scale_rad']),dim=-1)
        f=self.decay*self.filtered+(1-self.decay)*target
        increment=torch.maximum(self.increment_low,torch.minimum(self.increment_high,f-self.offsets))
        rate_clipped=(increment!=f-self.offsets).any(-1)
        before=nominal+self.offsets+increment
        governed=torch.maximum(self.low,torch.minimum(self.high,before))
        clipped=governed!=before
        self.offsets=governed-nominal
        self.filtered=torch.where(clipped,self.offsets,f)
        return governed,dict(reference_clipped=clipped.any(-1),reference_clip_channels=clipped,
            rate_clipped=rate_clipped,
            policy_fault=~torch.isfinite(z).all(-1),target=target)
