"""Generate and audit local fixed embedded pebbles, with existing terrain NPZ I/O."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
from sttw_control.route_pebbles import build_mesh,cap_height
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--scene-config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
cfg=json.loads(a.scene_config.read_text());route=np.load(cfg['route'])
if not (0<cfg['horizontal_scale_m']<=.01 and cfg['vertical_scale_m']==.001 and cfg['half_width_m']>=.7):raise ValueError('G0 requires local <=1cm mesh,1mm vertical and>=.7m half width')
v,t,stones,grid=build_mesh(cfg,route);a.output.mkdir(parents=True,exist_ok=False)
np.savez_compressed(a.output/'terrain.npz',vertices=v,triangles=t,height_field=np.rint(grid['height']/cfg['vertical_scale_m']).astype(np.int16),grid_x=grid['x'],grid_y=grid['y'])
with (a.output/'stone_centers.csv').open('w') as f:
 w=csv.writer(f);w.writerow(['id','s_m','n_m','radius_s_m','radius_n_m','height_m','world_x_m','world_y_m'])
 for i,(s,n,rs,rn) in enumerate(stones):
  yaw=np.interp(s,route['s'],route['heading']);xy=np.array([np.interp(s,route['s'],route['xy'][:,k]) for k in (0,1)])+n*np.array([-np.sin(yaw),np.cos(yaw)])
  w.writerow([i,s,n,rs,rn,cfg['height_m'],*xy])
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
identity=dict(cfg,sha256=sha(a.output/'terrain.npz'),route_sha256=sha(Path(cfg['route'])),terrain_kind='route_pebbles',terrain_support_end_s_m=cfg['s_range_m'][1],profile='fixed embedded caps; max overlap; cosine entry/exit; world local patch stitched to coarse flat mesh',vertices=len(v),triangles=len(t))
(a.output/'identity.json').write_text(json.dumps(identity,indent=2)+'\n');(a.output/'scene_config.json').write_text(json.dumps(cfg,indent=2)+'\n')
e=np.diff(grid['height'],axis=0)/cfg['horizontal_scale_m'];f=np.diff(grid['height'],axis=1)/cfg['horizontal_scale_m'];boundary=np.r_[grid['height'][0],grid['height'][-1],grid['height'][:,0],grid['height'][:,-1]]
normal=np.cross(v[t[:,1]]-v[t[:,0]],v[t[:,2]]-v[t[:,0]])
audit=dict(stones=len(stones),height_min_m=float(v[:,2].min()),height_max_m=float(v[:,2].max()),slope_abs_percentiles=np.percentile(abs(np.r_[e.ravel(),f.ravel()]),[0,50,95,100]).tolist(),raised_fraction_of_local_patch=float((grid['height']>0).mean()),local_patch_bounds_xy_m=grid['bounds'].tolist(),patch_boundary_max_abs_height_m=float(abs(boundary).max()),upward_triangles=bool((normal[:,2]>0).all()),zero_area_triangles=int((np.linalg.norm(normal,axis=1)==0).sum()),collision_mesh_sha256=identity['sha256'],route_sha256=identity['route_sha256'],centers_sha256=sha(a.output/'stone_centers.csv'),center_radius_sha256=hashlib.sha256(stones.tobytes()).hexdigest(),duplicate_base_plane=False,scope='geometry audit only; runtime contact and1cm/5mm comparison pending')
(a.output/'terrain_geometry_audit.json').write_text(json.dumps(audit,indent=2)+'\n');(a.output/'geometry_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig=plt.figure(figsize=(13,4),layout='constrained');ax=fig.add_subplot(131);im=ax.pcolormesh(grid['x'],grid['y'],grid['height'].T*1000,shading='auto',cmap='viridis',vmin=0,vmax=cfg['height_m']*1000);ax.plot(*route['xy'].T,'--',color='white',lw=.8);ax.set(xlim=(grid['x'][0],grid['x'][-1]),ylim=(grid['y'][0],grid['y'][-1]),xlabel='World x (m)',ylabel='World y (m)');ax.set_aspect('equal');fig.colorbar(im,ax=ax,label='Height (mm)')
ax=fig.add_subplot(132);ss=np.linspace(*cfg['s_range_m'],2000);ax.plot(ss,cap_height(ss,np.zeros_like(ss),stones,cfg)*1000,label='Analytic cap max');ax.set(xlabel='Route s (m)',ylabel='Centerline height (mm)');ax.legend(fontsize=8);ax.grid(alpha=.2)
ax=fig.add_subplot(133,projection='3d');skip=max(1,round(.02/cfg['horizontal_scale_m']));x,y=np.meshgrid(grid['x'][::skip],grid['y'][::skip],indexing='ij');ax.plot_surface(x,y,grid['height'][::skip,::skip],cmap='viridis',rstride=1,cstride=1,linewidth=0);ax.set(xlabel='World x (m)',ylabel='World y (m)',zlabel='Height (m)')
fig.suptitle(f"{cfg['name']} | fixed caps, seed{cfg['seed']} | horizontal {cfg['horizontal_scale_m']*1000:g}mm / vertical1mm | geometry DEV");fig.savefig(a.output/'terrain_preview.png',dpi=150);plt.close(fig)
print(json.dumps(audit))
