"""Seeded fixed rough banks separated by a real open trench (no backing plane)."""
import numpy as np


def build_ditch_terrain(spec):
    front, width = spec['front_x_m'], spec['width_m']
    if width <= 0 or not 0 < spec['stone_height_max_m'] <= .15:
        raise ValueError('Positive trench width and stone height in (0, .15] required')
    rng = np.random.default_rng(spec['seed'])
    vertices, triangles = [], []
    def patch(xs, ys, rough, bank_height):
        x, y = np.meshgrid(xs, ys, indexing='ij')
        z = np.zeros_like(x)
        if rough:
            count = int((xs[-1]-xs[0])*(ys[-1]-ys[0])*spec['stones_per_m2'])
            for _ in range(count):
                cx, cy = rng.uniform(xs[0],xs[-1]), rng.uniform(ys[0],ys[-1])
                radius = rng.uniform(.055,.09)
                height = rng.uniform(spec['stone_height_min_m'],spec['stone_height_max_m'])
                z = np.maximum(z, height*np.maximum(0,1-((x-cx)**2+(y-cy)**2)/radius**2)**2)
            # Match the surrounding flat mesh at its outer seams.
            z *= np.clip((y-ys[0])/.1,0,1)*np.clip((ys[-1]-y)/.1,0,1)
            z *= np.clip((x-xs[0])/.1,0,1)*np.clip((xs[-1]-x)/.1,0,1)
        start = sum(len(v) for v in vertices)
        vertices.append(np.stack((x,y,z+bank_height),-1).reshape(-1,3))
        a = np.arange(x.size).reshape(x.shape)[:-1,:-1].ravel()+start
        stride = len(ys)
        triangles.extend((np.stack((a,a+stride,a+1),-1),np.stack((a+1,a+stride,a+stride+1),-1)))
    for left,right,bank_height in ((-5.,front,0.),(front+width,spec['ground_end_x_m'],spec.get('far_bank_height_m',0.))):
        rough_left,rough_right = max(left,2.),min(right,8.)
        patch(np.linspace(rough_left,rough_right,int(np.ceil((rough_right-rough_left)/.025))+1),np.linspace(-1.,1.,81),True,bank_height)
        if left < rough_left: patch(np.array([left,rough_left]),np.array([-1.,1.]),False,bank_height)
        if rough_right < right: patch(np.array([rough_right,right]),np.array([-1.,1.]),False,bank_height)
        for bottom,top in ((-4.,-1.),(1.,4.)):
            patch(np.array([left,right]),np.array([bottom,top]),False,bank_height)
    return np.concatenate(vertices).astype(np.float32),np.concatenate(triangles).astype(np.int32)


def ditch_support_mask(wheels, forces, roll, pitch, forward_velocity, spec):
    on = ((wheels[:,:,0] > spec['front_x_m']+spec['width_m'])
          & (wheels[:,:,0] < spec['ground_end_x_m'])
          & (wheels[:,:,1].abs() < spec.get('lateral_limit_m',.9))
          & (wheels[:,:,2] >= .045+spec.get('far_bank_height_m',0.))
          & (wheels[:,:,2] <= .095+spec['stone_height_max_m']+spec.get('far_bank_height_m',0.))).all(-1)
    return on & (forces[:,:,2]>1.).all(-1) & (roll.abs()<.15) & (pitch.abs()<.15) & (forward_velocity>=1.)
