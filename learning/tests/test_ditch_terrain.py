import numpy as np
import torch
from sttw_control.ditch_terrain import build_ditch_terrain, ditch_support_mask
from sttw_control.phase_u_events import classify_terminal

SPEC=dict(front_x_m=3.6,width_m=.6,ground_end_x_m=25.,stone_height_min_m=.01,stone_height_max_m=.04,stones_per_m2=30,seed=20261109)


def test_chain_has_two_meter_flat_takeoff_and_landing_buffers():
    import json
    from pathlib import Path
    config=json.loads(Path('learning/configs/playground_sttw_rb_sttw.json').read_text())
    vertices,faces=build_ditch_terrain(config['ditch_terrain'])
    xyz=vertices[faces]
    for lo,hi in config['flat_intervals_m']:
        overlaps=(xyz[:,:,0].max(1)>lo+1e-6)&(xyz[:,:,0].min(1)<hi-1e-6)
        np.testing.assert_allclose(xyz[overlaps,:,2],0.,atol=1e-8)
        assert hi-lo >= 2.-1e-6
    assert vertices[:,2].max() > .039


def test_gap_is_open_and_both_banks_are_rough():
    vertices, faces = build_ditch_terrain(SPEC)
    xs=vertices[faces,0]
    assert np.all((xs.max(1)<=3.60001)|(xs.min(1)>=4.19999))
    assert not ((vertices[:,0]>3.60001)&(vertices[:,0]<4.19999)).any()
    for lo,hi in ((2.,3.6),(4.2,8.)):
        z=vertices[(vertices[:,0]>lo)&(vertices[:,0]<hi),2]
        assert z.max()>.025 and z.max()<=.04
    assert np.array_equal(vertices,build_ditch_terrain(SPEC)[0])
    normal=np.cross(vertices[faces[:,1]]-vertices[faces[:,0]],vertices[faces[:,2]]-vertices[faces[:,0]])
    assert (normal[:,2]>0).all()


def test_landing_requires_both_wheels_across_contact_and_motion():
    wheels=torch.tensor([[[4.6,0,.09],[4.3,0,.08]]]).repeat(5,1,1)
    forces=torch.zeros(5,2,3);forces[:,:,2]=15
    wheels[1,1,0]=4.1
    forces[2,1,2]=0
    wheels[3,:,2]=.23 # old raised platform is not this task
    speed=torch.tensor([1.5,1.5,1.5,1.5,0.])
    assert ditch_support_mask(wheels,forces,torch.zeros(5),torch.zeros(5),speed,SPEC).tolist()==[True,False,False,False,False]


def test_trench_failure_is_terminal_and_retains_legacy_codes():
    import json
    from pathlib import Path
    config=json.loads(Path('learning/configs/isaac_phase_u_task_ditch.json').read_text())['task']
    inputs={k:torch.zeros(2,dtype=torch.bool) for k in ('nonfinite','illegal_contact','backward_exit','stuck','jump_zone_seen')}
    inputs.update(roll=torch.tensor([0.,1.]),pitch=torch.zeros(2),yaw=torch.zeros(2),episode_step=torch.zeros(2,dtype=torch.long),terrain_failure=torch.tensor([True,False]))
    result=classify_terminal(inputs,config)
    assert result['terminated'].tolist()==[True,True]
    assert result['end_code'].tolist()==[12,3]


def test_crossing_skill_can_land_off_center_with_declared_bank_boundary():
 p=torch.tensor([[[4.6,1.2,.07],[4.3,1.2,.07]]]);f=torch.zeros_like(p);f[:,:,2]=10
 z=torch.zeros(1);v=torch.ones(1)*1.5
 assert not ditch_support_mask(p,f,z,z,v,SPEC).item()
 assert ditch_support_mask(p,f,z,z,v,{**SPEC,'lateral_limit_m':3.9}).item()


def test_full_width_ridges_share_collision_height_and_leave_gap_open():
 from sttw_control.ditch_event_model import KnownHeightMap
 spec={**SPEC,'front_x_m':10.3,'width_m':.3,'full_width_rough':True,'rough_x_intervals_m':[[2.5,8.5],[12.,20.]],'stone_height_min_m':.001,'stone_height_max_m':.004,'stones_per_m2':0,'land_bridge_y_interval_m':[1.4,4.], 'ridges':[{'x_interval_m':[13.,15.],'height_m':.08}]}
 v,f=build_ditch_terrain(spec)
 assert not ((v[:,0]>10.30001)&(v[:,0]<10.59999)&(abs(v[:,1])<1.4)).any()
 h=KnownHeightMap(v,spec);h.local=np.zeros((1,2));r=np.zeros((4,13));r[:,6]=1;r[:,:2]=[[14.,2.2],[10.45,0],[10.45,2.2],[9.5,0]]
 crop=h.crop(r)
 np.testing.assert_allclose(crop[:,0],[80,0,0,0],atol=.001)
 assert crop[:,1].tolist()==[1,0,1,1]


def test_playground_patches_preserve_isaac_vertices_and_open_trench():
 spec={**SPEC,'stones_per_m2':1,'land_bridge_y_interval_m':[1.4,4.]}
 vertices,faces,patches=build_ditch_terrain(spec,return_patches=True)
 original_vertices,original_faces=build_ditch_terrain(spec)
 np.testing.assert_array_equal(vertices,original_vertices)
 np.testing.assert_array_equal(faces,original_faces)
 rebuilt=[]
 for xs,ys,z in patches:
  x,y=np.meshgrid(xs,ys,indexing='ij')
  rebuilt.append(np.stack((x,y,z),-1).reshape(-1,3))
  # Rotate grid 90 degrees to preserve the Isaac cell diagonal.
  bottom=z.min();scale=max(z.max()-bottom,1e-6)
  heightfield=((z[::-1,:]-bottom)/scale).astype(np.float32)
  np.testing.assert_allclose(heightfield[::-1,:]*scale+bottom,z,atol=1e-8)
 np.testing.assert_array_equal(np.concatenate(rebuilt).astype(np.float32),vertices)


def test_rotated_heightfield_uses_isaac_triangle_diagonal():
 import mujoco
 # One source cell with z=x*y at vertices. Isaac splits along x+y=1.
 m=mujoco.MjModel.from_xml_string('<mujoco><asset><hfield name="h" nrow="2" ncol="2" size=".5 .5 1 .1"/></asset><worldbody><geom type="hfield" hfield="h" pos=".5 .5 0" quat=".7071067811865476 0 0 .7071067811865476"/></worldbody></mujoco>')
 z=np.array([[0.,0.],[0.,1.]])
 m.hfield_data[:]=z[::-1,:].ravel()
 d=mujoco.MjData(m);mujoco.mj_forward(m,d)
 for x,y in ((.25,.25),(.75,.75),(.25,.75),(.75,.25),(.5,.5)):
  gid=np.zeros(1,dtype=np.int32)
  distance=mujoco.mj_ray(m,d,np.array([x,y,2.]),np.array([0.,0.,-1.]),None,1,-1,gid)
  np.testing.assert_allclose(2-distance,max(0.,x+y-1),atol=1e-7)
