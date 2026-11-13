"""Render saved STTW poses at real time; no physics rerun or trajectory smoothing."""
import argparse,json,os,re,subprocess
from pathlib import Path
os.environ.setdefault('MUJOCO_GL','egl')
import numpy as np
import mujoco
from PIL import Image,ImageDraw,ImageFont
p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('--method',default='alpha0');p.add_argument('--case-regex',default='.*');p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
d=json.loads((a.run/'suite_declaration.json').read_text());status=json.loads((a.run/'status.json').read_text());z=np.load(a.run/'trajectory.npz');model=mujoco.MjModel.from_binary_path(str(a.run/'terrain.mjb'));data=mujoco.MjData(model);renderer=mujoco.Renderer(model,height=480,width=800);font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18);records=[]
for row in status['results']:
 if row['method']!=a.method or not re.search(a.case_regex,row['case_name']):continue
 lane=row['lane'];valid=z['valid'][:,lane];poses=z['qpos'][valid,lane];times=z['time'][valid];duration=float(times[-1]);file=a.output/(row['case_name']+'.mp4');indices=np.clip(np.round(np.arange(0,duration,1/30)/.005).astype(int),0,len(poses)-1)
 writer=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','1600x480','-r','30','-i','-','-an','-c:v','libx264','-crf','22','-pix_fmt','yuv420p','-movflags','+faststart',str(file)],stdin=subprocess.PIPE)
 for j in np.r_[indices,np.repeat(len(poses)-1,30)]:
  data.qpos[:]=poses[j];data.qvel[:]=0;mujoco.mj_forward(model,data);images=[]
  for overview in [False,True]:
   camera=mujoco.MjvCamera();camera.type=mujoco.mjtCamera.mjCAMERA_FREE;camera.lookat[:]=[(poses[0,0]+poses[-1,0])/2,(poses[0,1]+poses[-1,1])/2,.1] if overview else [poses[j,0],poses[j,1],.2];camera.distance=6 if overview else 1.8;camera.azimuth=120;camera.elevation=-70 if overview else -20;renderer.update_scene(data,camera);images.append(renderer.render().copy())
  frame=Image.fromarray(np.concatenate(images,axis=1));draw=ImageDraw.Draw(frame);draw.rectangle((0,0,1600,63),fill=(20,20,25));draw.text((12,6),f"{row['case_name']} | {a.method} | locked hip/knee | saved-pose replay 1x",font=font,fill='white');draw.text((12,34),f"t={times[j]:.3f}s | speed={z['speed'][valid,lane][j]:.2f}m/s | roll={np.rad2deg(z['measurement'][valid,lane,0][j]):.1f}deg | end={row['termination']}"+(' | TERMINAL FRAME HELD' if j==len(poses)-1 else ''),font=font,fill='white');writer.stdin.write(np.asarray(frame).tobytes())
 writer.stdin.close();assert writer.wait()==0
 frame.save(file.with_suffix('.png'));records.append(dict(file=file.name,lane=lane,case=row['case_name'],duration_s=duration,frames=len(indices)+30,terminal_hold_s=1,termination=row['termination']))
renderer.close();(a.output/'index.json').write_text(json.dumps(dict(source=str(a.run),role='Saved measured qpos replay, no re-integration, real-time30fps plus1s labeled final hold',videos=records),indent=2));(a.output/'README.md').write_text('# STTW saved-pose videos\n\n'+ '\n'.join(f"- [{r['case']}]({r['file']}): {r['termination']}, {r['duration_s']:.3f}s." for r in records)+'\n');print(f'Rendered {len(records)} videos')
