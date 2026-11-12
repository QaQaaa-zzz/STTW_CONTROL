"""Freeze a paired one-factor contact matrix, assets and geometry before rollout."""
import argparse,copy,hashlib,itertools,json,os,shutil,subprocess,sys,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--resume-preparation',action='store_true');a=p.parse_args();plan=json.loads(a.config.read_text());root=Path(plan['root']);root.mkdir(parents=True,exist_ok=a.resume_preparation);source=Path(plan['source_theory']);base=json.loads((source/'calibration/single10_v2.6_seed101.json').read_text());old_protocol=json.loads((source/'layout_test/protocol.json').read_text());base_scene=json.loads((source/'calibration/single10_scene.json').read_text());bundle=Path(plan['base_bundle']).resolve();route=dict(np.load(bundle/'route.npz'));cpu=sys.executable;repo=Path.cwd()
for key in ('physics_contact_recording','alpha_switch','channel_intervention'):base.pop(key,None)
base.update(num_envs=6,trace_lanes=6,alphas=plan['alphas'],zero_middle=[False]*5+[True],case_labels=plan['case_labels'],zero_reference_channels=[[False,False]]*6,user_speed_m_s=plan['speed_m_s'],record_body_poses=False,record_controller_terms=True,compact_results=True,paired_reward_diagnostics=True,auto_reset_horizon=False,omit_diagnostic_fields=['observation','policy_measurement'])
base['evaluation_role']='Paired one-factor mechanism sweep; fixed original H75/Local300, no training or selector fitting'
groups=copy.deepcopy(old_protocol['physical_groups']);old=[json.loads((source/f'calibration/single10_v2.6_seed{s}.json').read_text())['randomization'] for s in (101,102,103)]
keys=('friction_range','mass_scale','servo_gain_scale','steer_zero_offset_rad');bounds={k:[min(c[k][0] for c in old),max(c[k][0] for c in old)] for k in keys}
for seed in plan['physical_groups']:
 if str(seed) not in groups:
  rng=np.random.default_rng(seed);g=copy.deepcopy(old[0])
  for key in keys:value=float(rng.uniform(*bounds[key]));g[key]=[value,value]
  groups[str(seed)]=g
# Cache the exact same nearest-segment projection used by place_isaac_terrain.py.
spacing=base_scene['horizontal_scale_m'];axis=np.arange(base_scene['bounds_m'][0],base_scene['bounds_m'][1]+spacing/2,spacing);xx,yy=np.meshgrid(axis,axis,indexing='ij');points=np.stack((xx.ravel(),yy.ravel()),axis=-1);ids=np.flatnonzero(route['s']<=float(route['turn_end'])+3);xy=route['xy'][ids];start,edge=xy[:-1],np.diff(xy,axis=0);ss=[];nn=[]
for chunk in np.array_split(points,32):
 delta=chunk[:,None,:]-start;fraction=np.clip((delta*edge).sum(-1)/(edge*edge).sum(-1),0,1);offset=delta-fraction[...,None]*edge;nearest=(offset*offset).sum(-1).argmin(-1);row=np.arange(len(chunk));ss.append(route['s'][ids[:-1]][nearest]+fraction[row,nearest]*np.diff(route['s'][ids])[nearest]);tangent=edge[nearest]/np.linalg.norm(edge[nearest],axis=-1)[:,None];nn.append(tangent[:,0]*offset[row,nearest,1]-tangent[:,1]*offset[row,nearest,0])
cache=root/'route_projection.npz';np.savez_compressed(cache,s=np.concatenate(ss),n=np.concatenate(nn));print('cached original route projection',flush=True)
base_asset=Path(base['asset']).resolve();asset_contract=json.loads(base_asset.with_name('contract.json').read_text());body_names={b['name']:b for b in asset_contract['bodies']};original_axle=float(np.linalg.norm(np.asarray(body_names['frontwheel']['position'])[:2]-np.asarray(body_names['rearwheel']['position'])[:2]));geometry=[];assets={1.:str(base_asset)}

def vehicle_summary(contract,scale,asset,delta):
 masses=np.array([b['mass'] for b in contract['bodies']]);positions=[];rotations=[]
 for b in contract['bodies']:
  q=np.asarray(b['quat_wxyz']);rot=Rotation.from_quat(q[[1,2,3,0]]).as_matrix();rotations.append(rot);positions.append(np.asarray(b['position'])+rot@np.asarray(b['inertial_pos']))
 positions=np.array(positions);com=np.average(positions,axis=0,weights=masses);inertia=np.zeros((3,3))
 for mass,pos,rot,b in zip(masses,positions,rotations,contract['bodies']):
  d=pos-com;inertia+=rot@np.asarray(b['inertia_body'])@rot.T+mass*(np.eye(3)*np.dot(d,d)-np.outer(d,d))
 records={b['name']:b for b in contract['bodies']};axle=np.asarray(records['frontwheel']['position'])-np.asarray(records['rearwheel']['position'])
 return dict(scale=scale,asset=asset,steering_head_delta_local_x_m=delta,zero_pose_projected_axle_distance_m=float(np.linalg.norm(axle[:2])),controller_wheelbase_m=json.loads(Path(base['controller_config']).read_text())['wheelbase'],total_mass_kg=float(masses.sum()),zero_pose_com_world_m=com.tolist(),zero_pose_inertia_com_world_kg_m2=inertia.tolist(),changed_parameter='steering-head local x mount only; fixed local masses and inertias')
geometry.append(vehicle_summary(asset_contract,1.,str(base_asset),0.))
for scale in plan['wheelbase_scales']:
 if scale==1:continue
 dest=root/'assets'/f'wheelbase_{scale:g}';dest.mkdir(parents=True,exist_ok=a.resume_preparation);tree=ET.parse(base_asset);mount=tree.find("joint[@name='steering_joint']/origin");position=np.fromstring(mount.get('xyz'),sep=' ');delta=original_axle*(scale-1);position[0]+=delta;mount.set('xyz',' '.join(format(x,'.17g') for x in position));tree.write(dest/'vehicle.urdf',encoding='unicode')
 for name in asset_contract['meshes']:shutil.copyfile(base_asset.parent/name,dest/name)
 contract=copy.deepcopy(asset_contract);q=np.asarray(contract['root_quat_wxyz']);translation=Rotation.from_quat(q[[1,2,3,0]]).apply([delta,0,0])
 for b in contract['bodies']:
  if b['name'] in ('frontfork','frontsuspension','frontwheel'):b['position']=(np.asarray(b['position'])+translation).tolist()
 contract['exported_xml_sha256']=hashlib.sha256((dest/'vehicle.urdf').read_bytes()).hexdigest();contract['geometry_variant']=dict(parent_asset=str(base_asset),scale=scale,delta_steering_head_x_m=delta,controller_updated=False)
 (dest/'contract.json').write_text(json.dumps(contract,indent=2));assets[scale]=str(dest/'vehicle.urdf');geometry.append(vehicle_summary(contract,scale,assets[scale],delta))
 assert [b['mass'] for b in contract['bodies']]==[b['mass'] for b in asset_contract['bodies']]
 assert [b['inertia_body'] for b in contract['bodies']]==[b['inertia_body'] for b in asset_contract['bodies']]
(root/'vehicle_geometry.json').write_text(json.dumps(geometry,indent=2));print('prepared geometry variants',flush=True)
terrains={};terrain_jobs=[];terrain_parameters=[]
def terrain(center=4.2,length=.8,oblique=.25,height=.1):
 key=(center,length,oblique,height)
 if key in terrains:return terrains[key]
 name=f's{center:g}_length{length:g}_oblique{oblique:g}_h{height:g}';out=root/'terrains'/name;cfg=copy.deepcopy(base_scene);cfg.update(center_s_m=center,length_m=length,oblique_s_per_lateral_m=oblique,height_m=height,projection_cache=str(cache),role='Frozen paired one-factor geometry sweep');path=root/'scene_configs'/(name+'.json');path.parent.mkdir(exist_ok=True);path.write_text(json.dumps(cfg,indent=2));terrains[key]=str(out);terrain_jobs.append((path,out));terrain_parameters.append(dict(terrain=str(out),center_s_m=center,reference_heading_deg=float(np.degrees(np.interp(center,route['s'],route['heading']))),reference_curvature_per_m=float(np.interp(center,route['s'],route['curvature'])),length_m=length,height_m=height,oblique_s_per_n=oblique,nominal_peak_grade=height*np.pi/length,nominal_peak_slope_deg=float(np.degrees(np.arctan(height*np.pi/length))),nominal_peak_second_derivative_per_m=2*np.pi**2*height/length**2))
 return str(out)
stages=[];cells=[]
def stage(name,asset=None,custom_bundle=None,initial_state=None):
 path=root/name;path.mkdir(exist_ok=a.resume_preparation);item=dict(name=name,path=str(path),asset=asset or str(base_asset),bundle=custom_bundle or str(bundle),initial_state=initial_state or base['initial_state'],cells=[]);stages.append(item);return item

def cell(item,cell_id,factor,value,center=4.2,length=.8,oblique=.25,height=.1,scale=1.,terrain_path=None):
 row=dict(id=cell_id,stage=item['name'],factor=factor,value=value,center_s_m=center,length_m=length,oblique_s_per_n=oblique,height_m=height,wheelbase_scale=scale,world_rotation_deg=value if factor=='world_rotation' else 0,terrain=terrain_path or terrain(center,length,oblique,height),reference_heading_deg=float(np.degrees(np.interp(center,route['s'],route['heading'])))+(value if factor=='world_rotation' else 0))
 item['cells'].append(row);cells.append(row)
A=stage('position')
for center in plan['position_centers_s_m']:cell(A,f'position_s{center:g}','position',center,center=center)
cell(A,'flat_original','flat',0,height=0)
B=stage('slope')
for center,length in itertools.product(plan['paired_centers_s_m'],plan['lengths_m']):
 if length!=plan['base_length_m']:cell(B,f'length{length:g}_s{center:g}','length',length,center=center,length=length)
for scale in plan['wheelbase_scales']:
 if scale==1:continue
 C=stage(f'wheelbase_{scale:g}',asset=assets[scale])
 for center in plan['paired_centers_s_m']:cell(C,f'wheelbase{scale:g}_s{center:g}','wheelbase',scale,center=center,scale=scale)
 cell(C,f'flat_wheelbase{scale:g}','flat_wheelbase',scale,height=0,scale=scale)
D=stage('obliquity')
for oblique in plan['obliquities_s_per_n']:
 if oblique!=plan['base_obliquity_s_per_n']:cell(D,f'oblique{oblique:g}','obliquity',oblique,oblique=oblique)
# Build exact source meshes once per unique surface; no simulator startup.
env=os.environ.copy();env['LD_LIBRARY_PATH']='/home/qy/miniconda3/envs/go2_isaacgym/lib'+(':'+env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '');gym='/home/qy/miniconda3/envs/go2_isaacgym/bin/python'
with (root/'terrain_build.log').open('w') as log:
 for path,out in terrain_jobs:
  if a.resume_preparation and (out/'terrain.npz').exists():continue
  subprocess.run([gym,'learning/cli/place_isaac_terrain.py','--terrain','bend_bump','--scene-config',str(path),'--output',str(out)],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
for angle in plan['world_rotations_deg']:
 if angle==0:continue
 rot=Rotation.from_euler('z',angle,degrees=True);pivot=np.load(base['initial_state'])['root_pose'][:3];pivot=pivot.copy();pivot[2]=0
 def rotate_xyz(x):return rot.apply(np.asarray(x)-pivot)+pivot
 derived=root/'bundles'/f'yaw{angle:g}';derived.mkdir(parents=True,exist_ok=a.resume_preparation)
 for filename in ('actor.npz','config.json'):shutil.copyfile(bundle/filename,derived/filename)
 ar=copy.deepcopy(route);ar['xy']=rotate_xyz(np.column_stack((ar['xy'],np.zeros(len(ar['xy'])))))[:,:2];ar['heading']=(ar['heading']+np.radians(angle)+np.pi)%(2*np.pi)-np.pi;np.savez_compressed(derived/'route.npz',**ar)
 identity=json.loads((bundle/'identity.json').read_text());identity['files']['route.npz']=hashlib.sha256((derived/'route.npz').read_bytes()).hexdigest();identity['world_rotation_deg']=angle;(derived/'identity.json').write_text(json.dumps(identity,indent=2))
 initial=dict(np.load(base['initial_state']));pose=initial['root_pose'].copy();pose[:3]=rotate_xyz(pose[:3]);q=pose[3:7];qnew=(rot*Rotation.from_quat(q[[1,2,3,0]])).as_quat();pose[3:7]=qnew[[3,0,1,2]];initial['root_pose']=pose;initial['root_velocity']=np.r_[rot.apply(initial['root_velocity'][:3]),rot.apply(initial['root_velocity'][3:])];np.savez_compressed(derived/'initial_state.npz',**initial)
 original=Path(terrain());raw=dict(np.load(original/'terrain.npz'));raw['vertices']=rotate_xyz(raw['vertices']).astype(np.float32);out=root/'terrains'/f'world_yaw{angle:g}';out.mkdir(exist_ok=a.resume_preparation);np.savez_compressed(out/'terrain.npz',**raw);meta=json.loads((original/'identity.json').read_text());meta.update(route=str(derived/'route.npz'),world_rotation_deg=angle,rotation_pivot=pivot.tolist(),sha256=hashlib.sha256((out/'terrain.npz').read_bytes()).hexdigest());(out/'identity.json').write_text(json.dumps(meta,indent=2))
 E=stage(f'world_yaw{angle:g}',custom_bundle=str(derived),initial_state=str(derived/'initial_state.npz'));cell(E,f'world_yaw{angle:g}','world_rotation',angle,terrain_path=str(out))
# Quantify the actual discretized meshes, rather than only requested dimensions.
for row in terrain_parameters:
 z=np.load(Path(row['terrain'])/'terrain.npz');v=z['vertices'];tri=z['triangles'];normal=np.cross(v[tri[:,1]]-v[tri[:,0]],v[tri[:,2]]-v[tri[:,0]]);raised=(v[tri,2]>0).any(1);nz=normal[:,2];finite=raised&(abs(nz)>1e-10);grades=np.linalg.norm(normal[finite,:2],axis=1)/abs(nz[finite]);row.update(actual_max_height_m=float(v[:,2].max()),actual_max_finite_face_grade=float(grades.max()) if len(grades) else 0.,vertical_raised_faces=int((raised&(abs(nz)<=1e-10)).sum()))
(root/'terrain_geometry.json').write_text(json.dumps(terrain_parameters,indent=2));(root/'cells.json').write_text(json.dumps(cells,indent=2));(root/'physical_groups.json').write_text(json.dumps(groups,indent=2))
queue=root/'queue';queue.mkdir(exist_ok=a.resume_preparation);total=0;queue_stages=[]
for item in stages:
 folder=Path(item['path']);jobs=[]
 for cellrow,seed in itertools.product(item['cells'],plan['physical_groups']):
  name=f'{cellrow["id"]}_v{plan["speed_m_s"]:g}_seed{seed}';cfg=copy.deepcopy(base);cfg.update(asset=item['asset'],initial_state=item['initial_state'],seed=seed,randomization=groups[str(seed)])
  cfgpath=folder/(name+'.json');cfgpath.write_text(json.dumps(cfg,indent=2));jobs.append(dict(name=name,config=str(cfgpath),terrain=cellrow['terrain'],physical_seed=seed,family='ridge' if cellrow['height_m'] else 'flat',height_m=cellrow['height_m'],speed_m_s=plan['speed_m_s'],cell=cellrow['id']))
 spec=copy.deepcopy(old_protocol);spec.update(experiment_id=plan['experiment_id']+'_'+item['name'],title='H75单因素接触机制：'+item['name'],case_labels=plan['case_labels'],seeds=plan['physical_groups'],bundle=item['bundle'],scenes=len(jobs),lanes=len(jobs)*6,batch_cases=len(jobs),max_batch_envs=plan['max_batch_envs'],plot_workers=2,terrains=[c['id'] for c in item['cells']],duplicate_pairs=[],data_role='Paired one-factor mechanism experiment; post-S4 development, not independent selector TEST',role='Original common start; fixed preference for whole rollout',retention=plan['retention'],physical_groups=groups,stop_rule=plan['stop_rule'],estimated_wall_time_minutes=plan['estimated_wall_time_minutes'],physical_parameter_envelope=bounds)
 for name,obj in [('jobs.json',jobs),('protocol.json',spec),('status.json',dict(stage='prepared',planned=len(jobs),episodes=len(jobs)*6))]:(folder/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2))
 queue_stages.append(dict(name=item['name'],path=str(folder),contexts=len(jobs),episodes=len(jobs)*6,analysis_command=[cpu,'learning/cli/analyze_contact_oat.py','--root',str(root),'--stage',item['name']]));total+=len(jobs)*6
assert total==plan['expected_episodes'],total
plan.update(total_episodes=total,cells=len(cells),unique_base_terrains=len(terrain_parameters),code_revision_before_freeze=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),baselines_reused_from_position_stage='length0.8 at s3.9/4.5, wheelbase1 flat/ridge, obliquity0.25 and world yaw0 at s4.2',source_policy_identity=json.loads((bundle/'identity.json').read_text()),physical_groups_path=str(root/'physical_groups.json'))
(root/'protocol.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2));(root/'config.json').write_text(a.config.read_text())
(queue/'plan.json').write_text(json.dumps(dict(stages=queue_stages,total_episodes=total,optimizer_updates=0,final_analysis_command=[cpu,'learning/cli/analyze_contact_oat.py','--root',str(root)]),indent=2))
(root/'README.md').write_text('# H75接触机制单因素实验\n\n2112回合，44参数单元×8固定物理组×五alpha及Zero。固定原H75/Local300与ECBC，0训练，10秒/回合。\n\n因素：13障碍位置、7坡长（含复用基准）、5轴距（含基准，带平地对照）、5斜交角（含复用基准）、整体旋转0/45/90度。各配对只改一个因素，原始观测/物理频率及失败判据不变。\n\n[参数与预算](protocol.json) · [地形实测](terrain_geometry.json) · [车辆几何/质量/惯量](vehicle_geometry.json) · [排队状态](queue/status.json) · [结果与趋势（结束后）](analysis/README.md)\n\n轴距变化通过移动前转向总成安装位置实现，保持各刚体局部质量/惯量、轮胎及执行器不变；整车COM/惯量随布局产生的变化已记录。保持原控制器参数的几何迁移试验；平地对照用于辨别控制器迁移问题。\n\n位置变化对应转弯阶段；整体旋转只检验坐标对称性，不能把绝对参考航向当作地形致因。保留全部失败、完整窗口与最终保持分开报告。只保留必要逐步NPZ、数值表和PNG，不默认制作视频或5kHz接触。\n')
(queue/'README.md').write_text('# 有限实验队列\n\n[实验合同](../protocol.json) · [实时状态](status.json) · [全实验入口](../README.md)\n\n'+''.join(f'- {s["name"]}: {s["contexts"]}工况/{s["episodes"]}回合\n' for s in queue_stages)+'\n阶段结束发送Bark；异常停止，全部结束自动统计。\n')
print(json.dumps(dict(root=str(root),episodes=total,stages=queue_stages),indent=2),flush=True)
