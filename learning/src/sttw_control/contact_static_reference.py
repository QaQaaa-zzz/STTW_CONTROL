"""Two-support quasi-static load reference at the exported asset reference pose."""
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation


def static_wheel_loads(contract):
    asset=json.loads(Path(contract['settings']['asset']).with_name('contract.json').read_text())
    bodies={body['name']:body for body in asset['bodies']}
    names=contract['bodies'];n=contract['settings']['num_envs']
    position=np.array([bodies[name]['position'] for name in names])
    rotations=np.array([bodies[name]['quat_wxyz'] for name in names])[:,[1,2,3,0]]
    matrix=Rotation.from_quat(rotations).as_matrix()
    if 'randomization_readback' in contract:
        masses=np.array(contract['randomization_readback']['masses'])
        local_com=np.array(contract['randomization_readback']['coms'])[...,:3]
    else:
        masses=np.broadcast_to([bodies[name]['mass'] for name in names],(n,len(names)))
        local_com=np.broadcast_to([bodies[name]['inertial_pos'] for name in names],(n,len(names),3))
    world_com=position[None]+np.einsum('bij,nbj->nbi',matrix,local_com)
    total=masses.sum(1);com=(world_com*masses[...,None]).sum(1)/total[:,None]
    rear=position[names.index('rearwheel'),:2];front=position[names.index('frontwheel'),:2]
    wheelbase=float(np.linalg.norm(front-rear));forward=(front-rear)/wheelbase
    front_load=total*9.81*((com[:,:2]-rear)*forward).sum(1)/wheelbase
    return dict(scope='quasi-static two-support balance at exported reference pose; actual frozen masses/local COM readback; axle horizontal projections used as support points; not solved dynamic equilibrium',wheelbase_m=wheelbase,total_mass_kg=total.tolist(),front_load_N=front_load.tolist(),rear_load_N=(total*9.81-front_load).tolist())
