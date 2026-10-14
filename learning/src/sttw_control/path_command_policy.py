"""Geometric schema v1; physical correction filter shared by train/evaluate."""
import jax.numpy as j
from flax import linen as nn
SCHEMA='sttw_geometric_path_actor351_v1'

def map_latent(z,config):
    c=config['residual_interface'];a=j.tanh(z)
    return j.stack([a[0]*j.where(a[0]>=0,c['speed_positive_scale_m_s'],c['speed_negative_scale_m_s']),a[1]*c['steer_scale_rad']])

def correction_tick(filtered,offsets,z,nominal,config):
    c=config['residual_interface'];dt=config['timing']['lower_dt_s']
    target=map_latent(z,config);a=j.exp(-dt/j.array(c['filter_tau_s']))
    f=a*filtered+(1-a)*target
    increment=j.clip(f-offsets,j.array([-c['correction_speed_down_m_s2'],-c['correction_steer_rate_rad_s']])*dt,j.array([c['correction_speed_up_m_s2'],c['correction_steer_rate_rad_s']])*dt)
    before=nominal+offsets+increment
    governed=j.clip(before,j.array([c['speed_reference_bounds_m_s'][0],-c['steer_reference_bound_rad']]),j.array([c['speed_reference_bounds_m_s'][1],c['steer_reference_bound_rad']]))
    clipped=governed!=before;out=governed-nominal
    return j.where(clipped,out,f),out,governed,dict(reference_clipped=j.any(clipped),reference_clip_channels=clipped,rate_clipped=j.any(increment!=f-offsets),policy_fault=~j.all(j.isfinite(z)),target=target)

def validate_actor_input(obs):
    if obs.shape[-1]!=351:raise ValueError('geometric schema requires 351 inputs; legacy345 incompatible')

def assemble_observation(frames,mask,context,frame_scales,config):
    scales=j.array([x['scale'] for x in config['network']['context_fields']])
    obs=j.concatenate([(frames/frame_scales).reshape(-1),mask,context/scales]);validate_actor_input(obs)
    return j.clip(obs,-5,5),~j.all(j.isfinite(obs))

LEGACY_ARCHITECTURE = 'legacy'
MODULATED_ARCHITECTURE = 'alpha_modulated_v1'
ALPHA_INDEX = 336
MODULATION_SHAPES = {'gain_2': (128,), 'shift_2': (128,),
                     'gain_3': (64,), 'shift_3': (64,)}


def architecture_identity(config):
    """Resolve the explicit network identity; absent means the historical MLP."""
    architecture = (config or {}).get('network', {}).get('architecture', LEGACY_ARCHITECTURE)
    if architecture not in (LEGACY_ARCHITECTURE, MODULATED_ARCHITECTURE):
        raise ValueError(f'unsupported path network architecture: {architecture}')
    return architecture


def _validate_parameters(params, architecture, input_dim, output_dim):
    values = params['params'] if 'params' in params else params
    expected = {f'Dense_{i}' for i in range(4)}
    if architecture == MODULATED_ARCHITECTURE:
        expected |= set(MODULATION_SHAPES)
    if set(values) != expected:
        raise ValueError(f'path network architecture {architecture} parameter keys mismatch')
    widths = (input_dim, 128, 128, 64, output_dim)
    for i in range(4):
        layer = values[f'Dense_{i}']
        if (set(layer) != {'kernel', 'bias'} or
                layer['kernel'].shape != (widths[i], widths[i+1]) or
                layer['bias'].shape != (widths[i+1],)):
            raise ValueError(f'path network architecture {architecture} Dense_{i} shape mismatch')
    if architecture == MODULATED_ARCHITECTURE:
        for key, shape in MODULATION_SHAPES.items():
            if values[key].shape != shape:
                raise ValueError(f'path network architecture {architecture} {key} shape mismatch')


def validate_actor_parameters(params, config):
    """Reject incomplete or cross-architecture checkpoint parameters before apply."""
    _validate_parameters(params, architecture_identity(config), 351, 2)


class PathCommandActor(nn.Module):
    architecture: str = LEGACY_ARCHITECTURE

    @nn.compact
    def __call__(self, obs):
        validate_actor_input(obs)
        architecture_identity({'network': {'architecture': self.architecture}})
        if not self.is_initializing():
            _validate_parameters(self.variables, self.architecture, 351, 2)
        x = obs
        q = 2 * obs[..., ALPHA_INDEX:ALPHA_INDEX+1] - 1
        for index, width in enumerate((128, 128, 64)):
            x = nn.Dense(width)(x)
            if self.architecture == MODULATED_ARCHITECTURE and index > 0:
                gain = self.param(f'gain_{index+1}', nn.initializers.zeros, (width,))
                shift = self.param(f'shift_{index+1}', nn.initializers.zeros, (width,))
                x = (1 + q * gain) * x + q * shift
            x = nn.elu(x)
        return nn.Dense(2, kernel_init=nn.initializers.zeros,
                        bias_init=nn.initializers.zeros)(x)


class PathCommandCritic(nn.Module):
    architecture: str = LEGACY_ARCHITECTURE

    @nn.compact
    def __call__(self, obs):
        if obs.shape[-1] != 352:
            raise ValueError('geometric critic352 required')
        architecture_identity({'network': {'architecture': self.architecture}})
        if not self.is_initializing():
            _validate_parameters(self.variables, self.architecture, 352, 1)
        x = obs
        q = 2 * obs[..., ALPHA_INDEX:ALPHA_INDEX+1] - 1
        for index, width in enumerate((128, 128, 64)):
            x = nn.Dense(width)(x)
            if self.architecture == MODULATED_ARCHITECTURE and index > 0:
                gain = self.param(f'gain_{index+1}', nn.initializers.zeros, (width,))
                shift = self.param(f'shift_{index+1}', nn.initializers.zeros, (width,))
                x = (1 + q * gain) * x + q * shift
            x = nn.elu(x)
        return nn.Dense(1)(x)


def make_path_actor(config):
    return PathCommandActor(architecture=architecture_identity(config))


def make_path_critic(config):
    return PathCommandCritic(architecture=architecture_identity(config))


def make_torch_path_network(input_dim, output_dim, config):
    """Trainable Torch MLP preserving historical numeric Dense state keys."""
    import torch
    architecture = architecture_identity(config)
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


def export_torch_actor(actor, config=None):
    """Export all learned Actor parameters, including feature modulation."""
    import torch
    inferred = getattr(actor, 'architecture', LEGACY_ARCHITECTURE)
    architecture = inferred if config is None else architecture_identity(config)
    if architecture != inferred:
        raise ValueError('Torch Actor architecture/config mismatch')
    expected_keys = {f'{index}.{field}' for index in (0, 2, 4, 6)
                     for field in ('weight', 'bias')}
    if architecture == MODULATED_ARCHITECTURE:
        expected_keys |= set(MODULATION_SHAPES)
    if set(actor.state_dict()) != expected_keys:
        raise ValueError('Torch Actor architecture parameter keys mismatch')
    values = {}
    for i, index in enumerate((0, 2, 4, 6)):
        layer = actor[index]
        if not isinstance(layer, torch.nn.Linear):
            raise ValueError('Torch Actor architecture requires numeric Dense layers')
        values[f'Dense_{i}'] = {
            'kernel': j.asarray(layer.weight.detach().cpu().numpy().T),
            'bias': j.asarray(layer.bias.detach().cpu().numpy())}
    if architecture == MODULATED_ARCHITECTURE:
        for key in MODULATION_SHAPES:
            values[key] = j.asarray(getattr(actor, key).detach().cpu().numpy())
    params = {'params': values}
    _validate_parameters(params, architecture, 351, 2)
    return params
