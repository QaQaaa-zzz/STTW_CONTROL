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
        self.execution_mode = torch.zeros(count,device=device,dtype=torch.long)
        self.hold_channels = torch.zeros(count,2,device=device,dtype=torch.bool)
        c=config['residual_interface'];dt=config['timing']['lower_dt_s']
        self.decay=torch.exp(-dt*torch.ones(2,device=device)/torch.tensor(c['filter_tau_s'],device=device))
        self.increment_low=dt*torch.tensor([-c['correction_speed_down_m_s2'],-c['correction_steer_rate_rad_s']],device=device)
        self.increment_high=dt*torch.tensor([c['correction_speed_up_m_s2'],c['correction_steer_rate_rad_s']],device=device)
        self.low=torch.tensor([c['speed_reference_bounds_m_s'][0],-c['steer_reference_bound_rad']],device=device)
        self.high=torch.tensor([c['speed_reference_bounds_m_s'][1],c['steer_reference_bound_rad']],device=device)

    def reset(self,indices):
        self.filtered[indices]=0
        self.offsets[indices]=0
        self.hold_channels[indices]=False

    @torch.no_grad()
    def latent(self,obs):
        return path_latent(self.weights,obs)

    @torch.no_grad()
    def correct(self,z,nominal):
        c=self.config['residual_interface'];a=torch.tanh(z)
        target=torch.stack((a[:,0]*torch.where(a[:,0]>=0,c['speed_positive_scale_m_s'],c['speed_negative_scale_m_s']),a[:,1]*c['steer_scale_rad']),dim=-1)
        f=self.decay*self.filtered+(1-self.decay)*target
        delta=f-self.offsets
        increment=torch.maximum(self.increment_low,torch.minimum(self.increment_high,delta))
        budget=torch.where(delta>=0,self.increment_high,-self.increment_low)
        eta=torch.where(delta.abs()>0,(budget/delta.abs()).clamp_max(1),torch.ones_like(delta))
        common=eta.min(-1).values
        joint=common[:,None]*delta
        steer_priority=torch.stack((joint[:,0],increment[:,1]),dim=-1)
        increment=torch.where((self.execution_mode==1)[:,None],joint,increment)
        increment=torch.where((self.execution_mode==2)[:,None],steer_priority,increment)
        rate_clipped=(increment!=f-self.offsets).any(-1)
        increment=torch.where(self.hold_channels,torch.zeros_like(increment),increment)
        before=nominal+self.offsets+increment
        governed=torch.maximum(self.low,torch.minimum(self.high,before))
        clipped=governed!=before
        self.offsets=governed-nominal
        self.filtered=torch.where(clipped,self.offsets,f)
        return governed,dict(reference_clipped=clipped.any(-1),reference_clip_channels=clipped,
            rate_clipped=rate_clipped,
            policy_fault=~torch.isfinite(z).all(-1),target=target,channel_progress=eta,held_channels=self.hold_channels.clone())


def path_latent(weights,obs):
    """Shared mean network for frozen evaluation and gradient-based finetuning."""
    x=obs;q=2*obs[:,336:337]-1;w=weights
    for i in range(3):
        x=x@w[f'Dense_{i}_kernel']+w[f'Dense_{i}_bias']
        if i>0:x=(1+q*w[f'gain_{i+1}'])*x+q*w[f'shift_{i+1}']
        x=torch.nn.functional.elu(x)
    return x@w['Dense_3_kernel']+w['Dense_3_bias']


class TrainablePathActor(torch.nn.Module):
    """Weights-only H75 finetuning; original artifacts remain immutable."""
    def __init__(self,frozen,initial_std=.1):
        super().__init__()
        self.weights=torch.nn.ParameterDict({k:torch.nn.Parameter(v.clone()) for k,v in frozen.weights.items()})
        self.log_std=torch.nn.Parameter(torch.full_like(frozen.offsets[0],float(np.log(initial_std))))

    def forward(self,obs):
        return path_latent(self.weights,obs)

    def distribution(self,obs):
        return torch.distributions.Normal(self(obs),self.log_std.exp())

    def export_arrays(self):
        return {k:v.detach().cpu().numpy().copy() for k,v in self.weights.items()}


def make_torch_path_network(input_dim, output_dim, config):
    """Trainable Torch MLP preserving historical numeric Dense state keys."""
    architecture = config.get('network',{}).get('architecture','legacy')
    if architecture not in ('legacy','alpha_modulated_v1'):raise ValueError('unsupported path network architecture')
    MODULATED_ARCHITECTURE='alpha_modulated_v1';ALPHA_INDEX=336
    MODULATION_SHAPES={'gain_2':(128,),'shift_2':(128,),'gain_3':(64,),'shift_3':(64,)}
    if (input_dim, output_dim) not in ((351, 2), (352, 1)):
        raise ValueError('path network requires Actor351/2 or Critic352/1')

    class AlphaModulatedNetwork(torch.nn.Sequential):
        def __init__(self, *layers):
            super().__init__(*layers)
            self.architecture = architecture
            for key, shape in MODULATION_SHAPES.items():
                self.register_parameter(key, torch.nn.Parameter(torch.zeros(shape)))

        def forward(self, obs):
            q = 2 * obs[..., ALPHA_INDEX:ALPHA_INDEX+1] - 1
            x = self[1](self[0](obs))
            for dense_index, suffix in ((2, 2), (4, 3)):
                z = self[dense_index](x)
                z = (1 + q * getattr(self, f'gain_{suffix}')) * z + q * getattr(self, f'shift_{suffix}')
                x = self[dense_index+1](z)
            return self[6](x)

    widths = (input_dim, 128, 128, 64, output_dim)
    layers = []
    for index in range(4):
        layers.append(torch.nn.Linear(widths[index], widths[index+1]))
        if index < 3:
            layers.append(torch.nn.ELU())
    if architecture == MODULATED_ARCHITECTURE:
        return AlphaModulatedNetwork(*layers)
    return torch.nn.Sequential(*layers)
