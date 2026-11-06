"""Batched Torch translation of controller.py for the Isaac runtime.

Measurements: speed, steer angle/rate, left-positive roll/rate, steer request.
Configuration is the validated ControllerConfig dictionary. ESO updates even
while its correction is disabled, matching the original getOutput-before-update.
"""
import math
import torch


def _floor_magnitude(value, floor):
    return torch.where(value < 0, -1., 1.) * value.abs().clamp_min(floor)


class TorchController:
    def __init__(self, config, count, device='cpu', dtype=torch.float32):
        self.config = dict(config)
        self.eso = torch.zeros(count, 4, device=device, dtype=dtype)
        self.disturbance = torch.zeros(count, device=device, dtype=dtype)
        self.gains = torch.zeros(count, 3, device=device, dtype=dtype)
        self.initial_gains = self._system(torch.ones_like(self.disturbance))[-1]
        self.reset(slice(None))

    def _system(self, speed):
        c = self.config
        v = speed.clamp_min(c['minimum_speed'])
        inertia = c['mass']*c['cg_height']**2 + c.get('roll_inertia_com',.043929*c['mass']/5.434)
        m1 = -c['mass']*c['cg_forward']*c['cg_height']*v*math.cos(c['caster'])/c['wheelbase']
        m2 = _floor_magnitude(-(c['mass']*v*v*c['cg_height'] -
                              c['mass']*c['cg_forward']*c['trail']*c['gravity'])*
                             math.cos(c['caster'])/c['wheelbase'], .1)
        m4 = -c['mass']*c['gravity']*c['cg_height']
        a1, a2 = m1/inertia, m2/inertia
        a4 = torch.full_like(v, -m4/inertia)
        matrix = torch.stack((-a2*a2, a1*a2, -a1*a1,
                              -a2*a4, a1*a4, -a2,
                              a1*a4, -a2, a1), dim=-1).reshape(-1, 3, 3)
        det = _floor_magnitude(a4*a1*a1-a2*a2, 2.)
        target = torch.stack((torch.ones_like(v)*(3*c['wc']), 3*c['wc']**2+a4,
                              torch.ones_like(v)*c['wc']**3), dim=-1)
        gains = torch.where((det.abs() > 5.)[:, None],
                            (matrix @ target[:, :, None]).squeeze(-1)/det[:, None], self.gains)
        return a1, a2, a4, m4/m2, gains

    def reset(self, indices):
        self.eso[indices] = 0
        self.disturbance[indices] = 0
        self.gains[indices] = self.initial_gains[indices]

    def step(self, measurement, enable_eso):
        c = self.config
        speed, steer, steer_rate, roll, roll_rate, reference = measurement.unbind(-1)
        a1, a2, a4, ratio, gains = self._system(speed)
        shift = torch.where(enable_eso, -self.disturbance/a4, 0.)
        if c['fixed_roll_reference'] is not None:
            reference = ratio*c['fixed_roll_reference']
        reference_roll = reference/ratio if c['fixed_roll_reference'] is None else torch.full_like(speed, c['fixed_roll_reference'])
        error = torch.stack((reference-steer, reference_roll-roll+shift, -roll_rate), dim=-1)
        u = (gains*error).sum(-1).clamp(-c['max_steer_rate'], c['max_steer_rate'])
        e0, e1 = steer-self.eso[:, 0], roll-self.eso[:, 1]
        derivative = torch.stack((steer_rate+30*e0,
                                  self.eso[:, 2]+3*c['wo']*e1,
                                  a2*self.eso[:, 0]+a4*self.eso[:, 1]+self.eso[:, 3]+
                                  a1*steer_rate+(3*c['wo']**2+a4)*e1,
                                  c['wo']**3*e1), dim=-1)
        output = torch.stack((u, reference_roll, self.disturbance, -self.disturbance/a4), dim=-1)
        self.disturbance = self.disturbance+c['dt']*c['disturbance_filter']*(self.eso[:, 3]-self.disturbance)
        self.eso = self.eso+c['dt']*derivative
        self.gains = gains
        return output


class TorchVelocityServos:
    """Physical-step explicit velocity servos, driven by controller commands.

    Command order is [steer rate, forward-positive rear rate]; the rear sign is
    converted exactly once to the original MuJoCo joint convention. Apply at
    each 0.0002 s physics step, holding commands across 25 steps. This returns
    actuator torque only; passive suspension and joint damping remain separate.
    """
    def __init__(self, contract, joint_names, device='cpu', dtype=torch.float32, additional_actuators=()):
        actuators = {a['name']: a for a in contract['actuators']}
        selected = [actuators[name] for name in ('cmd_steering_v', 'cmd_rearwheel_v')]
        for a in selected:
            if (not a['enabled'] or not a['control_limited'] or not a['force_limited'] or
                    a['gear'] != [1., 0., 0., 0., 0., 0.] or a['gain'][0] <= 0 or
                    a['bias'][:3] != [0., 0., -a['gain'][0]]):
                raise ValueError('unsupported source velocity-servo contract')
        if any(a['enabled'] and a not in selected and a['name'] not in additional_actuators and any(a['gain']) for a in actuators.values()):
            raise ValueError('unmapped active actuator')
        self.indices = [joint_names.index(a['joint']) for a in selected]
        self.joint_count = len(joint_names)
        self.gain = torch.tensor([a['gain'][0] for a in selected], device=device, dtype=dtype)
        self.control = torch.tensor([a['ctrlrange'] for a in selected], device=device, dtype=dtype)
        self.force = torch.tensor([a['forcerange'] for a in selected], device=device, dtype=dtype)
        self.sign = torch.tensor([1., -1.], device=device, dtype=dtype)

    def torque(self, command, joint_velocity):
        target = torch.maximum(self.control[:, 0], torch.minimum(self.control[:, 1], command*self.sign))
        force = self.gain*(target-joint_velocity[:, self.indices])
        force = torch.maximum(self.force[:, 0], torch.minimum(self.force[:, 1], force))
        result = torch.zeros_like(joint_velocity)
        result[:, self.indices] = force
        return result


class TorchPositionServos:
    """Original affine position actuators, with source target and force bounds."""
    def __init__(self, contract, joint_names, names, device='cpu', *, allow_inactive=False):
        rows={a['name']:a for a in contract['actuators']}
        selected=[rows[n] for n in names]
        for a in selected:
            if (not a['enabled'] and not allow_inactive) or a['gear'] != [1.,0.,0.,0.,0.,0.] or a['bias'][0] != 0 or a['bias'][1] != -a['gain'][0]:
                raise ValueError('Expected enabled unit-gear affine position servo')
        self.indices=[joint_names.index(a['joint']) for a in selected]
        self.kp=torch.tensor([a['gain'][0] for a in selected],device=device)
        self.kd=torch.tensor([-a['bias'][2] for a in selected],device=device)
        self.control=torch.tensor([a['ctrlrange'] for a in selected],device=device)
        self.force=torch.tensor([a['forcerange'] for a in selected],device=device)

    def torque(self, target, position, velocity):
        target=torch.maximum(self.control[:,0],torch.minimum(self.control[:,1],target))
        force=self.kp*(target-position[:,self.indices])-self.kd*velocity[:,self.indices]
        force=torch.maximum(self.force[:,0],torch.minimum(self.force[:,1],force))
        result=torch.zeros_like(position);result[:,self.indices]=force
        return result
