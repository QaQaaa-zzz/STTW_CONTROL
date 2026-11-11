"""Render saved complete MuJoCo configurations; never integrate new physics."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
import argparse,json
from pathlib import Path
import numpy as np
import mujoco
import subprocess
from PIL import Image,ImageDraw,ImageFont
p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args()
declaration=json.loads((a.root/'declaration.json').read_text());status=json.loads((a.root/'status.json').read_text())
model=mujoco.MjModel.from_binary_path(declaration['terrain_model']);data=mujoco.MjData(model)
renderer=mujoco.Renderer(model,height=540,width=960);opt=mujoco.MjvOption();opt.geomgroup[1]=0
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',22)
small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
records=[]
for row in status['results']:
 name=f"{row['skill']}_route{row['route']}_{row['continuation']}";root=a.root/name
 archive=np.load(root/'trajectory.npz');trace=archive['trace'];poses=archive['qpos'];dt=.005
 file=a.root/(row['skill']+'_failure.mp4');writer=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','1920x540','-r','30','-i','-','-an','-c:v','libx264','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(file)],stdin=subprocess.PIPE)
 # 0.5x playback: 30 frames per video-second, 60 per simulated second.
 indices=np.unique(np.r_[np.round(np.arange(0,trace[-1,0],1/60)/dt).astype(int),len(trace)-1])
 for j in list(indices)+[len(trace)-1]*30:
  data.qpos[:]=poses[j];data.qvel[:]=0.;mujoco.mj_forward(model,data)
  images=[]
  for wide in (False,True):
   camera=mujoco.MjvCamera();camera.type=mujoco.mjtCamera.mjCAMERA_FREE
   camera.lookat[:]=[9.5,0,.15] if wide else [float(poses[j,0])+.5,float(poses[j,1]),.25]
   camera.distance=16 if wide else 3.3;camera.azimuth=110 if wide else 90;camera.elevation=-40 if wide else -12
   renderer.update_scene(data,camera,scene_option=opt);images.append(renderer.render().copy())
  image=Image.fromarray(np.concatenate(images,axis=1));draw=ImageDraw.Draw(image)
  draw.rectangle((0,0,1920,95),fill=(15,18,24));draw.rectangle((0,490,1920,540),fill=(15,18,24))
  phase='STTW continuation' if trace[j,10] else ('H75 approach' if trace[j,16] else 'RB jump control')
  draw.text((18,8),f"{row['skill']} | fresh recording of frozen-model trial | 0.5x playback",font=font,fill='white')
  draw.text((18,42),f"Simulation t={trace[j,0]:.3f}s | {phase} | vx={trace[j,7]:.2f} m/s | roll={np.rad2deg(trace[j,4]):.1f} deg | pitch={np.rad2deg(trace[j,5]):.1f} deg",font=font,fill='white')
  text=f"Near bank edge x=10.30 m; far bank x=10.60 m | End: {row['termination']} | Far-bank handoff: {row['far_contact']}"
  if j==len(trace)-1:text='RECORDED TERMINAL STATE (held image) | '+text
  draw.text((18,503),text,font=small,fill=(255,210,130));writer.stdin.write(np.asarray(image).tobytes())
 writer.stdin.close();code=writer.wait();assert code==0
 records.append(dict(video=file.name,source=name,frames=len(indices)+30,simulation_duration_s=float(trace[-1,0]),playback_speed=.5,terminal_hold_s=1,termination=row['termination'],far_contact=row['far_contact']))
 Image.fromarray(np.asarray(image)).save(a.root/(row['skill']+'_terminal.png'))
renderer.close();(a.root/'video_index.json').write_text(json.dumps(records,indent=2))
(a.root/'README.md').write_text('# Three RB failure recordings\n\nFresh matched-configuration recording attempts, not exact reproductions of the earlier trajectories. Frozen RB Run1/2/3; route61; H75 approach and planned H75 continuation. Actual full qpos recorded every 5ms and rendered without re-integrating physics. Left: side following camera; right: map overview. Playback 0.5x; final image held 1s, explicitly labeled.\n\n'+'\n'.join(f"- [{r['video']}]({r['video']}): {r['simulation_duration_s']:.3f}s simulation, {r['termination']}, far-bank contact={r['far_contact']}." for r in records)+'\n')
print(json.dumps(records,indent=2))
