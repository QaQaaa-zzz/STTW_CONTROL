"""STTW measurement conventions and passive joint forces in Torch."""
import math
import torch


def rotate_inverse(quat_xyzw, vector):
    xyz, w = quat_xyzw[..., :3], quat_xyzw[..., 3:]
    cross = torch.cross(xyz, vector, dim=-1)
    return vector - 2*w*cross + 2*torch.cross(xyz, cross, dim=-1)


class TorchVehicleSignals:
    def __init__(self, contract, joint_names, device='cpu', dtype=torch.float32):
        rows = {j['name']: j for j in contract['joints']}
        self.steer = joint_names.index('steering_joint')
        self.front = joint_names.index('frontwheel_joint')
        self.rear = joint_names.index('rearwheel_joint')
        q = contract['imu_site_quat_wxyz']
        self.site = torch.tensor(q[1:]+q[:1], device=device, dtype=dtype)
        self.stiffness = torch.tensor([rows[n]['stiffness'] for n in joint_names], device=device, dtype=dtype)
        self.damping = torch.tensor([rows[n]['damping'] for n in joint_names], device=device, dtype=dtype)
        self.spring_reference = torch.tensor([rows[n]['spring_reference'] for n in joint_names], device=device, dtype=dtype)

    def observe(self, root_state, position, velocity):
        # Isaac root_state: world position, xyzw quaternion, world linear/angular velocity.
        q = root_state[:, 3:7]
        x, y, z, w = q.unbind(-1)
        raw_roll = torch.atan2(2*(w*x+y*z), 1-2*(x*x+y*y))-math.pi/2
        body_gyro = rotate_inverse(q, root_state[:, 10:13])
        gyro = rotate_inverse(self.site.expand_as(q), body_gyro)
        measurement = torch.stack((-raw_roll, -gyro[:, 0], position[:, self.steer],
                                   velocity[:, self.steer], gyro[:, 2],
                                   -velocity[:, self.rear], -velocity[:, self.front]), dim=-1)
        yaw = torch.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))
        pose = torch.stack((root_state[:, 0], root_state[:, 1], yaw), dim=-1)
        speed = root_state[:, 7]*yaw.cos()+root_state[:, 8]*yaw.sin()
        return measurement, pose, speed

    def passive_torque(self, position, velocity):
        return -self.stiffness*(position-self.spring_reference)-self.damping*velocity
