"""Generate standard Isaac Gym terrain families used by legged_gym.

Calls the installed NVIDIA SDK; does not copy or modify its implementation.
These are 8 m source tiles, not automatically positioned rollout environments.
"""
import argparse
import hashlib
import json
from pathlib import Path
from isaacgym import terrain_utils
import numpy as np

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--seed',type=int,default=20261105)
p.add_argument('--slope-deg',type=float,default=6.)
p.add_argument('--roughness',type=float,default=.01)
p.add_argument('--step-height',type=float,default=.02)
a=p.parse_args()
if not 0<a.slope_deg<=20 or not .005<=a.roughness<=.05 or not .005<=a.step_height<=.1:
    raise ValueError('declared terrain library range exceeded')
a.output.mkdir(parents=True,exist_ok=False)
recipes={
 'rough': [('random_uniform_terrain',dict(min_height=-a.roughness,max_height=a.roughness,step=.005,downsampled_scale=.2))],
 'slope': [('sloped_terrain',dict(slope=float(np.tan(np.deg2rad(a.slope_deg)))))],
 'rough_slope': [('sloped_terrain',dict(slope=float(np.tan(np.deg2rad(a.slope_deg))))),('random_uniform_terrain',dict(min_height=-a.roughness,max_height=a.roughness,step=.005,downsampled_scale=.2))],
 'stairs': [('stairs_terrain',dict(step_width=.5,step_height=a.step_height))],
 'obstacles': [('discrete_obstacles_terrain',dict(max_height=a.step_height,min_size=.4,max_size=1.,num_rects=20,platform_size=2.))],
 'waves': [('wave_terrain',dict(num_waves=3,amplitude=2*a.roughness))],
}
records={}
for index,(name,recipe) in enumerate(recipes.items()):
    seed=a.seed+index;np.random.seed(seed)
    tile=terrain_utils.SubTerrain(name,width=80,length=80,vertical_scale=.005,horizontal_scale=.1)
    for function,kwargs in recipe:getattr(terrain_utils,function)(tile,**kwargs)
    vertices,triangles=terrain_utils.convert_heightfield_to_trimesh(tile.height_field_raw,.1,.005,.75)
    assert np.isfinite(vertices).all() and triangles.max()<len(vertices)
    np.savez_compressed(a.output/(name+'.npz'),vertices=vertices,triangles=triangles,height_field=tile.height_field_raw)
    records[name]=dict(seed=seed,recipe=recipe,file=name+'.npz',sha256=hashlib.sha256((a.output/(name+'.npz')).read_bytes()).hexdigest(),height_min_m=float(vertices[:,2].min()),height_max_m=float(vertices[:,2].max()),vertices=len(vertices),triangles=len(triangles))
source=Path(terrain_utils.__file__)
meta=dict(schema='sttw_isaac_standard_terrain_library_v1',source_module=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),references=['https://github.com/leggedrobotics/legged_gym/blob/master/legged_gym/utils/terrain.py','https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab/isaaclab/terrains/config/rough.py'],horizontal_scale_m=.1,vertical_scale_m=.005,slope_threshold=.75,tile_samples=[80,80],world_extent_m=[0,7.9],role='DEV terrain generator samples; not yet rollout placement or trained capability',physics_steps=0,optimizer_updates=0,terrains=records)
(a.output/'identity.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta,indent=2))
