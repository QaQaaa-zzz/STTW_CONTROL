"""Show frozen collision height fields and replay recorded link poses; never step physics."""
import argparse,json,os,subprocess,copy,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from PIL import Image,ImageDraw

p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--cases',type=Path,help='JSON list of saved run, terrain, lane, label and output name');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
ridge=[Path('runs/h75_terrain_speed_sweep/oblique_2cm'),Path('runs/h75_selected_local300_bend/bump_terrain'),Path('runs/h75_terrain_speed_sweep/oblique_6cm')]
density=[Path('runs/h75_gravel_p1/assets/S1'),Path('runs/h75_gravel_g0/assets/S2_5mm'),Path('runs/h75_gravel_p1/assets/S3')]
heights=[Path('runs/h75_gravel_p1/assets/S4'),density[1],Path('runs/h75_gravel_g0/assets/S5_5mm')]

def terrain(path):
 meta=json.loads((path/'identity.json').read_text())
 with np.load(path/'terrain.npz') as ar:
  h=ar['height_field'].astype(float)*meta['vertical_scale_m']
  x=ar['grid_x'] if 'grid_x' in ar else np.linspace(*meta['bounds_m'],h.shape[0]);y=ar['grid_y'] if 'grid_y' in ar else np.linspace(*meta['bounds_m'],h.shape[1])
 return meta,x,y,h

if a.cases is None:
 route=np.load('runs/h75_selected_local300_flat/bundle/route.npz');routexy=route['xy'];rs=route['s'];center=np.array([np.interp(5,rs,routexy[:,j]) for j in range(2)])
 fig,axs=plt.subplots(3,3,figsize=(15,11),layout='constrained')
 for row,paths in enumerate([ridge,density,heights]):
  for col,path in enumerate(paths):
   meta,x,y,h=terrain(path);ax=axs[row,col];ii=(x>center[0]-3)&(x<center[0]+3);jj=(y>center[1]-3)&(y<center[1]+3);vmax=60 if row==0 else (10 if row==1 else 20)
   img=ax.pcolormesh(x[ii],y[jj],h[np.ix_(ii,jj)].T*1000,cmap='YlOrBr',norm=Normalize(0,vmax),shading='nearest',rasterized=True)
   ids=(rs>=2)&(rs<=8);ax.plot(*routexy[ids].T,'--',color='#0072B2',lw=1,label='Original reference')
   if row==0:title=f'Single ridge | height {meta["height_m"]*100:.0f} cm'
   else:
    count=len((path/'stone_centers.csv').read_text().splitlines())-1;title=f'Pebbles | {meta["height_m"]*1000:.0f} mm | {meta["density_per_m2"]:.0f}/m² | N={count}'
   ax.set(title=title,xlabel='World x (m)',ylabel='World y (m)');ax.set_aspect('equal');ax.legend(fontsize=7,loc='upper left');fig.colorbar(img,ax=ax,label='Actual mesh height (mm)',shrink=.75)
 fig.suptitle('Actual experiment terrain height fields | rows: ridge height / pebble density / pebble height\nNo height magnification in the data; colors show height; rows use different color ranges',fontsize=13)
 fig.savefig(a.output/'terrain_conditions.png',dpi=160);plt.close(fig)
 # Side-view centerline profiles, evaluated on the actual stored grid.
 fig,axs=plt.subplots(1,2,figsize=(13,4),layout='constrained')
 from scipy.interpolate import RegularGridInterpolator
 s=np.linspace(2.5,7.5,1800);xy=np.column_stack([np.interp(s,rs,routexy[:,j]) for j in range(2)])
 for ax,paths in zip(axs,[ridge,heights]):
  for path in paths:
   meta,x,y,h=terrain(path);z=RegularGridInterpolator((x,y),h)(xy);ax.plot(s,z*1000,label=f'{meta["height_m"]*1000:.0f} mm nominal')
  ax.set(xlabel='Original route distance s (m)',ylabel='Actual centerline mesh height (mm)');ax.legend();ax.grid(alpha=.2)
 axs[0].set_title('Single ridge: 20 / 40 / 60 mm');axs[1].set_title('Pebbles: 5 / 10 / 20 mm, same centers and radii')
 fig.savefig(a.output/'height_profiles.png',dpi=160);plt.close(fig)

os.environ.setdefault('MUJOCO_GL','egl')
import mujoco

def video(name,run,ground,label,lane=2):
 contract=json.loads((run/'contract.json').read_text());names=contract['bodies'];alpha=contract['settings']['alphas'][lane];speed=contract['settings'].get('user_speed_m_s',2.3)
 with np.load(run/'trajectory.npz') as ar:
  poses=ar['body_poses'][:,:,lane].reshape(-1,len(names),7).copy();trace=ar['lower_trace'][:,:,lane].reshape(-1,11)
 # Isaac adapter stores link orientations as xyzw; MuJoCo mocap requires wxyz.
 poses[...,3:]=poses[...,[6,3,4,5]]
 origin=np.array(contract['origin'][lane]);poses[...,:3]-=origin
 live=np.flatnonzero(trace[:,6]);poses=poses[live];times=live*.005+trace[live,7]*contract['physics_dt']
 if contract.get('reference_routes'):
  route=np.load(run/contract['reference_routes']);case=json.loads((run/'initial_cases.json').read_text());rid=case['route_ids'][lane];routexy=route['xy'][rid];speed=float(route['speed'][rid])
 else:
  route=np.load('runs/h75_selected_local300_flat/bundle/route.npz');routexy=route['xy']
 raw=np.load(ground/'terrain.npz');is_mesh='vertices' in raw
 if is_mesh:meta=json.loads((ground/'identity.json').read_text())
 else:meta,x,y,h=terrain(ground);zmax=float(h.max())
 doc=ET.Element('mujoco');ET.SubElement(doc,'visual');asset=ET.SubElement(doc,'asset');world=ET.SubElement(doc,'worldbody')
 ET.SubElement(world,'geom',type='plane',size='35 35 .01',pos='0 0 -.0001',rgba='.66 .57 .38 1')
 ET.SubElement(world,'light',pos='0 0 10',dir='0 0 -1',diffuse='.8 .8 .8',castshadow='false')
 if is_mesh:
  mesh_file=(a.output/(name+'_terrain.obj')).resolve()
  with mesh_file.open('w') as f:
   np.savetxt(f,raw['vertices'],fmt='v %.8f %.8f %.8f')
   np.savetxt(f,raw['triangles']+1,fmt='f %d %d %d')
  ET.SubElement(asset,'mesh',name='ground_mesh',file=str(mesh_file))
  ET.SubElement(world,'geom',type='mesh',mesh='ground_mesh',rgba='.48 .43 .34 1',contype='0',conaffinity='0')
  # Exact source triangles in local metres, with a 1-m scale and the fixed route.
  import matplotlib.tri as mtri
  vertices=raw['vertices'];triangles=raw['triangles'];local=(vertices[:,:2]>=-4).all(1)&(vertices[:,:2]<=4).all(1);selected=triangles[local[triangles].all(1)]
  fig,ax=plt.subplots(figsize=(5,5),layout='constrained');mesh=ax.tripcolor(mtri.Triangulation(vertices[:,0],vertices[:,1],selected),vertices[:,2]*1000,cmap='terrain',shading='flat',vmin=0,vmax=float(vertices[:,2].max()*1000));ax.plot(*routexy.T,color='#0072B2',lw=1.5);ax.plot([-3.5,-2.5],[-3.5,-3.5],'k-',lw=3);ax.text(-3,-3.3,'1 m',ha='center');ax.set(xlim=(-4,4),ylim=(-4,4),xlabel='x (m)',ylabel='y (m)',title='Exact collision mesh / reference');ax.set_aspect('equal');fig.colorbar(mesh,ax=ax,label='Height (mm)',shrink=.7);fig.savefig(a.output/(name+'_terrain.png'),dpi=140);plt.close(fig)
  inset=Image.open(a.output/(name+'_terrain.png')).convert('RGB');inset.thumbnail((270,270))
 else:
  ET.SubElement(asset,'hfield',name='ground',nrow=str(len(y)),ncol=str(len(x)),size=f'{(x[-1]-x[0])/2} {(y[-1]-y[0])/2} {zmax} .02')
  ET.SubElement(world,'geom',type='hfield',hfield='ground',pos=f'{(x[-1]+x[0])/2} {(y[-1]+y[0])/2} 0',rgba='.66 .57 .38 1')
 source_xml=Path('mujoco_ros_ws20260313/model/scalebike_scene_matlab.xml').resolve();source=ET.parse(source_xml).getroot();bodies={b.attrib['name']:b for b in source.findall('.//body')};mesh_defs={m.attrib['name']:m for m in source.findall('asset/mesh')}
 for n in names:
  b=ET.SubElement(world,'body',name=n,mocap='true')
  for g in bodies[n].findall('geom'):
   if g.attrib.get('type')!='mesh' or g.attrib.get('group')!='1':continue
   mesh_name=g.attrib['mesh'];definition=copy.deepcopy(mesh_defs[mesh_name]);definition.set('file',str(source_xml.parent/'meshes'/definition.attrib['file']));asset.append(definition)
   visual_geom=copy.deepcopy(g);visual_geom.set('contype','0');visual_geom.set('conaffinity','0');b.append(visual_geom)
 ET.SubElement(doc,'statistic',extent='4');visual=doc.find('visual');ET.SubElement(visual,'global',offwidth='960',offheight='640');ET.SubElement(visual,'map',znear='.005');ET.SubElement(visual,'headlight',ambient='.6 .6 .6',diffuse='.8 .8 .8',specular='.1 .1 .1')
 model=mujoco.MjModel.from_xml_string(ET.tostring(doc,encoding='unicode'));
 if not is_mesh:model.hfield_data[:]=(h.T/zmax).ravel()
 data=mujoco.MjData(model);render=mujoco.Renderer(model,height=640,width=960);camera=mujoco.MjvCamera();camera.azimuth=135;camera.elevation=-23;camera.distance=1.65
 fps=30;sample=np.arange(times[0],times[-1],1/fps);ids=np.searchsorted(times,sample);ids=np.minimum(ids,len(times)-1)
 path=a.output/(name+'.mp4');proc=subprocess.Popen(['ffmpeg','-loglevel','error','-y','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24','-s','960x640','-r',str(fps),'-i','-','-an','-c:v','libx264','-preset','fast','-crf','22','-pix_fmt','yuv420p',str(path)],stdin=subprocess.PIPE)
 for k,i in enumerate(ids):
  data.mocap_pos[:]=poses[i,:,:3];data.mocap_quat[:]=poses[i,:,3:];mujoco.mj_forward(model,data);camera.lookat[:]=poses[i,0,:3];camera.lookat[2]=.12;render.update_scene(data,camera)
  # Original fixed route, added as a render-only blue line.
  for j in range(0,len(routexy)-12,12):
   aa=np.r_[routexy[j],.003];bb=np.r_[routexy[j+12],.003]
   if np.linalg.norm(aa[:2]-poses[i,0,:2])>2:continue
   scene=render.scene;geom=scene.geoms[scene.ngeom];mujoco.mjv_initGeom(geom,mujoco.mjtGeom.mjGEOM_CAPSULE,np.zeros(3),np.zeros(3),np.eye(3).ravel(),np.array([.05,.4,.9,1.]));mujoco.mjv_connector(geom,mujoco.mjtGeom.mjGEOM_CAPSULE,.004,aa,bb);scene.ngeom+=1
  frame=Image.fromarray(render.render())
  if is_mesh:frame.paste(inset,(960-inset.width,640-inset.height))
  draw=ImageDraw.Draw(frame);draw.rectangle((0,0,960,64),fill=(245,245,245));draw.text((12,8),label,fill=(15,15,15));draw.text((12,27),f'Recorded Isaac Lab poses | H75 + orange Local300 | alpha={alpha:g} | request={speed:g} m/s | t={times[i]:.2f}s',fill=(15,15,15));draw.text((12,46),'True geometry scale; original vehicle STL visual meshes; no physics replay or new simulation',fill=(15,15,15));proc.stdin.write(np.asarray(frame).tobytes())
  if k==45:frame.save(a.output/(name+'_frame.png'))
 proc.stdin.close();assert proc.wait()==0;render.close()
 return dict(terrain_identity=meta,reference_route_id=(rid if contract.get('reference_routes') else None),video=str(path),source=str(run),terrain=str(ground),lane=lane,alpha=alpha,frames=len(ids),fps=fps,duration_s=len(ids)/fps,role='visual replay of saved link poses with source XML STL visual meshes; original physics was Isaac Lab; no physics integration',camera='tracking, actual scale, nearest recorded5ms pose,30fps',source_origin=origin.tolist(),source_quaternion_order='xyzw',render_quaternion_order='wxyz',visual_model=str(source_xml),visual_geoms='Original XML group1 STL meshes only; collision proxy boxes hidden; all joint origins are zero in the source vehicle')
records=[]
default_cases=[('ridge_6cm',Path('runs/h75_gravel_p0/evaluation/single_6cm_v2p3_seed101/evaluation'),ridge[2],'Single ridge: height60 mm, length0.8 m, half-width0.8 m'),('pebbles_10mm_density24',Path('runs/h75_gravel_p0/evaluation/S2_v2p3_seed101/evaluation'),density[1],'Continuous pebbles: height10 mm, density24/m2, N154, radii55-80 mm')]
cases=default_cases if a.cases is None else [(r['name'],Path(r['run']),Path(r['terrain']),r['label'],r['lane']) for r in json.loads(a.cases.read_text())]
for args in cases:
 records.append(video(*args));(a.output/'manifest.json').write_text(json.dumps(dict(physics_steps=0,optimizer_updates=0,records=records,figure_source='Actual saved terrain height fields; no synthetic rocks or height exaggeration'),indent=2)+'\n');print('completed',args[0],flush=True)
