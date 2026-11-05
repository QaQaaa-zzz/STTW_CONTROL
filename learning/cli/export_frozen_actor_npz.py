"""Convert a trusted, hash-bound NumPy pickle to a portable inference NPZ.

Run with the source NumPy environment. This avoids NumPy 2 pickle module names
in the Isaac Python 3.8 environment; tensors and policy architecture do not change.
"""
import argparse
import hashlib
import json
import pickle
from pathlib import Path
import numpy as np

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--actor', type=Path, required=True)
p.add_argument('--sha256', required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
data = a.actor.read_bytes()
if hashlib.sha256(data).hexdigest() != a.sha256:
    raise ValueError('source actor identity mismatch')
params = pickle.loads(data)['params']
a.output.mkdir(parents=True, exist_ok=False)
np.savez(a.output/'actor.npz', **{f'{layer}_{name}':np.asarray(value)
    for layer, values in params.items() for name, value in values.items()})
with np.load(a.output/'actor.npz', allow_pickle=False) as archive:
    for layer, values in params.items():
        for name, value in values.items():
            np.testing.assert_array_equal(archive[f'{layer}_{name}'], value)
(a.output/'identity.json').write_text(json.dumps(dict(source=str(a.actor.resolve()),
    source_sha256=a.sha256, artifact='actor.npz',
    artifact_sha256=hashlib.sha256((a.output/'actor.npz').read_bytes()).hexdigest(),
    conversion='exact arrays, no optimization or architecture change'), indent=2))
