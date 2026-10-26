"""Explicit inference-only embedding of frozen legacy middle policies.

This does not convert a PPO checkpoint or change a reference interface. It
allows a legacy Actor's output to be evaluated with the modulated architecture
without inventing learned preference modulation.
"""
import copy

import numpy as np

from .path_command_policy import MODULATION_SHAPES, validate_actor_parameters


def embed_legacy_actor(params):
    """Copy a validated legacy tree, adding zero modulation parameters.

    Already-modulated inputs are rejected so learned gains cannot be erased.
    Original weights and frozen registry aliases remain untouched.
    """
    if set(params) != {'params'} or set(params['params']) != {f'Dense_{i}' for i in range(4)}:
        raise ValueError('only an unmodulated four-Dense legacy Actor may be embedded')
    validate_actor_parameters(params, {'network': {'architecture': 'legacy'}})
    result = copy.deepcopy(params)
    for name, shape in MODULATION_SHAPES.items():
        result['params'][name] = np.zeros(shape, dtype=np.float32)
    validate_actor_parameters(result, {'network': {'architecture': 'alpha_modulated_v1'}})
    return result
