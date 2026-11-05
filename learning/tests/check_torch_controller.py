"""Compare every controller state/output against the JAX fixture in Isaac's env."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from sttw_control.torch_controller import TorchController, TorchVelocityServos

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--fixture', type=Path, required=True)
parser.add_argument('--device', default='cpu')
parser.add_argument('--precision', choices=('float64', 'float32'), default='float64')
args = parser.parse_args()
dtype = getattr(torch, args.precision)
def tensor(array):
    return torch.as_tensor(array, device=args.device, dtype=torch.bool if array.dtype == bool else dtype)
errors = {}
for label in ('dynamic', 'fixed'):
    data = np.load(args.fixture/(label+'.npz'))
    controller = TorchController(json.loads((args.fixture/(label+'.json')).read_text()), 8, dtype=dtype, device=args.device)
    trace = {key: [] for key in ('output', 'eso', 'disturbance', 'gains')}
    for measurement, enable, reset in zip(data['measurements'], data['enable'], data['reset']):
        controller.reset(tensor(reset))
        trace['output'].append(controller.step(tensor(measurement), tensor(enable)).cpu().numpy().copy())
        for key in ('eso', 'disturbance', 'gains'):
            trace[key].append(getattr(controller, key).cpu().numpy().copy())
    errors[label] = {}
    for key, values in trace.items():
        actual = np.stack(values)
        np.testing.assert_allclose(actual, data[key], atol=1e-9 if dtype == torch.float64 else 5e-4,
                                   rtol=1e-9 if dtype == torch.float64 else 2e-5)
        errors[label][key] = float(np.max(np.abs(actual-data[key])))
contract = json.loads((args.fixture/'asset/contract.json').read_text())
servos = TorchVelocityServos(contract, [j['name'] for j in contract['joints']], device=args.device, dtype=dtype)
data = np.load(args.fixture/'servos.npz')
force = servos.torque(tensor(data['command']), tensor(data['velocity'])).cpu().numpy()
np.testing.assert_allclose(force, data['force'], atol=1e-10 if dtype == torch.float64 else 5e-5, rtol=1e-6)
errors['servo_force_max_error_Nm'] = float(np.max(np.abs(force-data['force'])))
(args.fixture/('torch_validation_'+args.device.replace(':', '_')+'_'+args.precision+'.json')).write_text(json.dumps(dict(passed=True, errors=errors,
    torch_version=torch.__version__, physics_steps=0, device=args.device, precision=args.precision,
    scope='controller arithmetic and selective reset against float64 JAX'), indent=2))
print(json.dumps(errors, indent=2))
