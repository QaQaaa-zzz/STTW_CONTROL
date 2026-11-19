"""Reconstruct the recorded native staged handback gates without changing policy or physics."""
import argparse,csv,json
from pathlib import Path
import mujoco,numpy as np
from audit_playground_jump import measure

p=argparse.ArgumentParser(description=__doc__);p.add_argument('evaluation',type=Path);p.add_argument('output',type=Path);a=p.parse_args();a.output.mkdir(exist_ok=True,parents=True)
src=a.evaluation;m=mujoco.MjModel.from_binary_path(str(src/'terrain.mjb'));md=mujoco.MjData(m);cfg=json.loads((src/'resolved_config.json').read_text());hand=cfg['ground_handoff'];rows=json.loads((src/'status.json').read_text())['results']
assert hand['require_both_far_wheels'] and hand['staged_native_handoff'] and not hand['require_settled_pendulum']
home=m.key('initial_state').qpos[-2:];gyro=m.sensor('gyro_local').adr.item();records=[];failures=[]
robot=np.flatnonzero((m.geom_bodyid>0)&((m.geom_contype|m.geom_conaffinity)!=0))
geom_type=m.geom_type[robot];size=m.geom_size[robot]
assert set(geom_type)<=set([4,5,6])  # ellipsoid, cylinder, box
for r in rows:
 path=src/r['case_dir']/'trajectory.npz';flights,_=measure(m,path,cfg['ditch_terrain']);valid=any(e['bank_to_bank_flight'] or e['solid_bank_rear_launch'] for e in flights)
 with np.load(path) as z:
  tr=z['trace'];t=tr[:,0];q=z['qpos'];v=z['qvel'];far=z['wheel_contact_flags'][:,2].astype(bool);force=z['contact_substep'][:,-1,:2]
 with np.load(path) as z:wallflags=z['world_flags'][:,1].astype(bool)
 loaded=far&(force>1e-5).all(1);edges=np.diff(np.r_[False,loaded,False].astype(int));spans=np.flatnonzero(edges==-1)-np.flatnonzero(edges==1)
 ix=np.flatnonzero(far);rollrate=np.zeros(len(q))
 for k in ix:
  md.qpos[:]=q[k];md.qvel[:]=v[k];mujoco.mj_forward(m,md);rollrate[k]=abs(md.sensordata[gyro])
 var=r['variant'];speed=v[:,0]*np.cos(tr[:,6])+v[:,1]*np.sin(tr[:,6])
 body=dict(both_far=far,roll=abs(tr[:,4])<var['gate_roll_rad'],pitch=abs(tr[:,5])<.2,vertical_speed=abs(v[:,2])<.5,gyro_x=rollrate<var['gate_roll_rate'],forward_speed=speed>=hand.get('homing_min_forward_m_s',1.5))
 lock=dict(both_far=far,roll=abs(tr[:,4])<.08,pitch=abs(tr[:,5])<.15,vertical_speed=abs(v[:,2])<.4,gyro_x=rollrate<.3,joint_position=np.max(abs(q[:,-2:]-home),1)<hand['lock_position_tolerance_rad'],joint_rate=np.max(abs(v[:,-2:]),1)<hand['lock_velocity_tolerance_rad_s'])
 prepared=np.isin(tr[:,19],[3,4,2]);homing=np.isin(tr[:,19],[4,2]);gate=lock if homing.any() else body;window=(homing if homing.any() else prepared)&far
 blockers='';best_t=None
 if window.any():
  keys=list(gate);matrix=np.column_stack([gate[k] for k in keys]);ks=np.flatnonzero(window);best=ks[matrix[ks].sum(1).argmax()];best_t=float(t[best]);blockers=';'.join(k for k,w in zip(keys,matrix[best]) if not w)
 elif r['far_contact']:blockers='no_prepared_both_far_frame'
 else:blockers='no_first_far_touch'
 hasload=bool(loaded.any());land=float(t[np.flatnonzero(loaded)[0]]) if hasload else None
 recorded_flight_land=any((e['solid_bank_rear_launch'] or e['bank_to_bank_flight']) and np.any(t[loaded]>=e['end_s']) for e in flights)
 entry=dict(skill=r['skill'],case=r['case_dir'],flight=valid,touch_s=r['first_far_contact_s'],dual_load_s=land,max_consecutive_loaded_ticks_ms=5*int(max(spans,default=0)),flight_then_load=recorded_flight_land,homing_phase=bool(homing.any()),handoff_s=r['handoff_s'],continued_after_handoff=r['handoff_s'] is not None and t[-1]>r['handoff_s']+.001,goal=r['goal'],termination=r['termination'],end_s=float(t[-1]),post_touch_abs_roll_max_deg=None if not r['far_contact'] else float(np.rad2deg(abs(tr[t>=r['first_far_contact_s'],4])).max()),gate_examined='lock' if homing.any() else 'prepare_to_home',closest_tick_s=best_t,unsatisfied_at_closest_tick=blockers)
 lip=cfg['ditch_terrain']['front_x_m']+cfg['ditch_terrain']['width_m']
 trigger=r['variant']['jump_start_x_m'];touch=r['first_far_contact_s']
 post=wallflags & (t>=touch) if touch is not None else np.zeros(len(t),bool)
 entry.update(jump_control_trigger_x_m=trigger,jump_control_entry_s=r['rb_entry_s'],verified_takeoff_s=next((e['start_s'] for e in flights if e['solid_bank_rear_launch'] or e['bank_to_bank_flight']),None),far_lip_x_m=lip,base_ever_past_lip=bool((q[:,0]>lip).any()),base_ever_past_10p8_m=bool((q[:,0]>10.8).any()),final_x_m=float(q[-1,0]),final_y_m=float(q[-1,1]),post_touch_s=None if touch is None else float(t[-1]-touch),nonwheel_ticks=int(wallflags.sum()),post_touch_nonwheel_ticks=int(post.sum()),first_post_touch_nonwheel_s=float(t[post][0]) if post.any() else None,first_dual_load_front_N=float(force[loaded][0,0]) if hasload else None,first_dual_load_rear_N=float(force[loaded][0,1]) if hasload else None)
 # Exact x support of each collision primitive under its measured transform.
 passed=None
 for k in np.flatnonzero(q[:,0]>lip):
  md.qpos[:]=q[k];mujoco.mj_kinematics(m,md)
  axis=md.geom_xmat[robot,:3]
  radius=np.where(geom_type==4,np.linalg.norm(axis*size,axis=1),
          np.where(geom_type==5,size[:,0]*np.linalg.norm(axis[:,:2],axis=1)+size[:,1]*abs(axis[:,2]),
                   (abs(axis)*size).sum(1)))
  if np.min(md.geom_xpos[robot,0]-radius)>lip:
   passed=float(t[k]);break
 entry['first_whole_robot_past_lip_s']=passed
 records.append(entry)
 if r['skill']=='rb_run1' and r['far_contact'] and r['handoff_s'] is None:failures.append(entry)
with (a.output/'landing_funnel.csv').open('w') as f:
 w=csv.DictWriter(f,records[0],lineterminator='\n');w.writeheader();w.writerows(records)
# Saved-pose contact sheets: one per distinct failure pattern, then longest dual support.
from PIL import Image,ImageDraw
renderer=mujoco.Renderer(m,height=270,width=400);opt=mujoco.MjvOption();opt.geomgroup[1]=0
seen=set();sheets=[]
for r in sorted(failures,key=lambda r:-r['max_consecutive_loaded_ticks_ms']):
 if r['unsatisfied_at_closest_tick'] in seen:continue
 seen.add(r['unsatisfied_at_closest_tick']);imgs=[]
 with np.load(src/r['case']/'trajectory.npz') as z:t=z['trace'][:,0];q=z['qpos']
 for tm in [r['touch_s'],r['closest_tick_s'] or (r['touch_s']+r['end_s'])/2,r['end_s']]:
  k=np.argmin(abs(t-tm));md.qpos[:]=q[k];mujoco.mj_forward(m,md);cam=mujoco.MjvCamera();cam.lookat[:]=[q[k,0],q[k,1],.24];cam.distance=2.2;cam.elevation=-12;cam.azimuth=92
  renderer.update_scene(md,cam,scene_option=opt);im=Image.fromarray(renderer.render().copy());draw=ImageDraw.Draw(im);draw.rectangle((0,0,400,19),fill='black');draw.text((4,3),f"{r['case']}  t={t[k]:.3f}s",fill='white');imgs.append(im)
 sheet=Image.new('RGB',(1200,270))
 for i,img in enumerate(imgs):sheet.paste(img,(i*400,0))
 name=r['case']+'_contact.jpg';sheet.save(a.output/name,quality=85);sheets.append(dict(case=r['case'],file=name))
 if len(sheets)==3:break
renderer.close();(a.output/'frame_index.json').write_text(json.dumps(sheets,indent=2))
