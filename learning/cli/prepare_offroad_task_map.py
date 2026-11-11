"""Build one physical height map and reusable route/speed/alpha task candidates."""
import argparse,json
from pathlib import Path
import numpy as np
from sttw_control.ditch_terrain import build_ditch_terrain
from sttw_control.ditch_event_model import route_parameters,KnownHeightMap
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--spec',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=False,parents=True)
spec=json.loads(a.spec.read_text());cfg=json.loads((a.source/'config.json').read_text());old=np.load(cfg['ground_handoff']['reference_path']);s=old['s'];dense_x=np.linspace(-5,50,22001)
# An S curve returns to the same straight jump corridor before x=9.
geometry=spec.get('route_geometry',{});lo,hi=geometry.get('s_interval_m',[2.5,8.5]);t=np.clip((dense_x-lo)/(hi-lo),0,1);sy=geometry.get('s_amplitude_m',1.1)*np.sin(2*np.pi*t)*np.sin(np.pi*t)**2
# Fillet a rectangular offset with actual 90 degree changes of tangent.
offset=geometry.get('corner_offset_m',2.2);exit_x=geometry.get('corner_exit_x_m',6.8)
vertices=np.array([[-5,0],[3.2,0],[3.2,offset],[exit_x,offset],[exit_x,0],[50,0]],float);rounded=[vertices[0]];radius=geometry.get('corner_radius_m',.65)
for i in range(1,len(vertices)-1):
 u=vertices[i]-vertices[i-1];u/=np.linalg.norm(u);v=vertices[i+1]-vertices[i];v/=np.linalg.norm(v)
 start=vertices[i]-radius*u;end=vertices[i]+radius*v;center=start+radius*v
 angle0=np.arctan2(*(start-center)[::-1]);turn=np.sign(np.cross(u,v));angles=angle0+turn*np.linspace(0,np.pi/2,101)
 rounded.extend(center+radius*np.column_stack((np.cos(angles),np.sin(angles))))
rounded.append(vertices[-1]);rounded=np.asarray(rounded)
rise=.5-.5*np.cos(np.pi*np.clip((dense_x-2)/6,0,1));fall=.5+.5*np.cos(np.pi*np.clip((dense_x-11)/5,0,1))
raw=[np.column_stack((dense_x,np.zeros_like(dense_x))),np.column_stack((dense_x,sy)),rounded,np.column_stack((dense_x,2.2*rise*fall))]
route=[]
for xy in raw:
 arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(xy,axis=0),axis=1))];route.append(np.column_stack([np.interp(s,arc,xy[:,j]) for j in (0,1)]))
route=np.asarray(route);choices=[(i,v,alpha) for i in range(4) for v in spec['speeds_m_s'] for alpha in spec['alphas']];ids=[c[0] for c in choices];xy=route[ids]
heading=np.unwrap(np.arctan2(np.gradient(xy[:,:,1],s,axis=1),np.gradient(xy[:,:,0],s,axis=1)),axis=1);curvature=np.gradient(heading,s,axis=1)
np.savez_compressed(a.output/'issued.npz',s=s,xy=xy,heading=heading,curvature=curvature);np.savez_compressed(a.output/'roads.npz',s=s,xy=xy)
cfg['ditch_terrain'].update(spec['terrain']);cfg['ditch_terrain']['land_bridge_y_interval_m']=[1.4,4.]
mesh,tri=build_ditch_terrain(cfg['ditch_terrain']);np.savez_compressed(a.output/'collision_mesh.npz',vertices=mesh,triangles=tri)
# Query the exact mesh triangles using the same local-map interpolator as the predictor.
height_map=KnownHeightMap(mesh,cfg['ditch_terrain']);xx,yy=np.meshgrid(np.arange(-5,25.001,.05),np.arange(-4,4.001,.05),indexing='ij');root=np.zeros((xx.size,13));root[:,:2]=np.column_stack((xx.ravel(),yy.ravel()));root[:,6]=1
height_map.local=np.zeros((1,2));sample=height_map.crop(root);height=sample[:,0].reshape(xx.shape)/1000;support=sample[:,1].reshape(xx.shape).astype(bool)
np.savez_compressed(a.output/'height_map.npz',x=xx,y=yy,height_m=np.where(support,height,np.nan),support=support)
params=[np.r_[route_parameters(route[i],v,alpha),float(i!=3)].tolist() for i,v,alpha in choices];names=[f'{spec["routes"][i]}_v{v}_a{alpha}' for i,v,alpha in choices]
rng=np.random.default_rng(spec['seed']);roots=[dict(y_m=float(rng.uniform(-.03,.03)),yaw_rad=float(rng.uniform(-.035,.035)),roll_rad=float(rng.uniform(-.02,.02)),speed_m_s=float(rng.uniform(1.9,2.1))) for _ in range(spec['roots'])]
logical=[dict(root=k,candidate=names[j],candidate_id=j,logical_case=k*len(choices)+j,role='DEV',x=1.8,method='forced') for k in range(len(roots)) for j in range(len(choices))];cases=[dict(logical[i],lane=lane) for lane,i in enumerate(rng.permutation(len(logical)))];n=len(cases)
cfg.update(num_envs=n,ground_initial_conditions=[roots[c['root']] for c in cases],task_reference_path=str(a.output/'roads.npz'),event_candidate_parameters=params,data_role=spec['role'])
cfg['ground_handoff'].update(reference_path=str(a.output/'issued.npz'),reference_route_ids=[7]*n,ground_only_lanes=[False]*n)
cfg['moving_event_planner']=dict(model=cfg['moving_event_planner']['model'],candidate_parameters=params,decision_x_m=1.8,methods=['forced']*n,forced_candidates=[c['candidate_id'] for c in cases],fixed_jump_candidate=7)
cfg.pop('evaluation_checkpoint',None)
(a.output/'config.json').write_text(json.dumps(cfg,indent=2));(a.output/'protocol.json').write_text(json.dumps(dict(roots=roots,cases=cases,candidates=names,route_names=names,parameters=params,budget=spec['budget'],scope=spec['role']),indent=2))
contract=dict(map_spec=cfg['ditch_terrain'],goal=cfg['ground_handoff']['goal'],road_half_width_m=.6,routes=spec['routes'],speeds_m_s=spec['speeds_m_s'],alphas=spec['alphas'],candidate_count=len(choices),flat_jump_interval_m=[8.5,12],height_grid_spacing_m=.05,height_source='piecewise-linear actual collision mesh, missing trench cells NaN with support=false',corner_radius_m=radius,physics_status='prepared, not yet executed',predictor_status='requires new data and calibration; old two-route score not used')
(a.output/'map_contract.json').write_text(json.dumps(contract,indent=2))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,(ax,bx)=plt.subplots(2,1,figsize=(14,8),layout='constrained');im=ax.pcolormesh(xx,yy,np.where(support,height,np.nan)*100,shading='auto',cmap='terrain');fig.colorbar(im,ax=ax,label='Collision surface height (cm)')
for i,name in enumerate(spec['routes']):ax.plot(route[i,:,0],route[i,:,1],label=name)
ax.set(xlim=(0,19),ylim=(-4,4),xlabel='x (m)',ylabel='y (m)',title='One map: reusable routes over physical stones and ridges; blank trench has no support');ax.set_aspect('equal');ax.legend(ncol=2)
for y in (0,2.2):
 j=np.argmin(abs(yy[0]-y));bx.plot(xx[:,j],np.where(support[:,j],height[:,j],np.nan)*100,label=f'y={y} m')
bx.set(xlim=(0,20),xlabel='x (m)',ylabel='height (cm)',title='Height profiles (metres in data; centimetres on plot)');bx.legend();fig.savefig(a.output/'map_overview.png',dpi=160)
print(a.output)
