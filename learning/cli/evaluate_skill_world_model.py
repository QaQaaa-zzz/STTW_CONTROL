"""Free full-segment predictions from true RB entry; no future state feedback."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from sttw_control.skill_world_model import SkillWorldModel
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True);p.add_argument('--model',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
torch.set_num_threads(1);device='cuda';tensor=lambda x:torch.as_tensor(x,device=device,dtype=torch.float32)
m=json.loads((a.data/'manifest.json').read_text());z=np.load(a.data/'sequences.npz');n=np.load(a.data/'normalization.npz')
model=SkillWorldModel(537,5,*[n[k] for k in ['input_mean','input_scale','velocity_scale','history_scale']]).to(device);saved=torch.load(a.model,map_location=device,weights_only=True);model.load_state_dict(saved['model']);model.eval()
rows=[r for r in m['rows'] if r['split']=='DEV'];records=[];arrays={}
with torch.no_grad():
 for length in sorted({r['length'] for r in rows}):
  group=[r for r in rows if r['length']==length];starts=np.array([r['start'] for r in group]);idx=starts[:,None]+np.arange(length)[None,:]
  physical=tensor(z['qpos'][idx]);velocity=tensor(z['qvel'][idx]);history=tensor(z['history'][starts]);phase=F.one_hot(torch.as_tensor(z['phase'][starts],device=device,dtype=torch.long),5).float();context=tensor(z['context'][starts])
  state=model.initialize(physical[:,0],velocity[:,0],history[:,:381],history[:,381:],tensor(z['contact'][starts]),phase,context);pred=model.rollout(state,context,length-1)
  pose=pred['qpos'].cpu().numpy();vel=pred['qvel'].cpu().numpy();contacts=pred['contact_logits'].sigmoid().cpu().numpy();phases=pred['phase_logits'].argmax(-1).cpu().numpy()
  for i,r in enumerate(group):
   actual=z['qpos'][idx[i]];error=np.linalg.norm(pose[i,:,:3]-actual[:,:3],axis=-1);rotation=2*np.arccos(np.clip(np.abs((pose[i,:,3:7]*actual[:,3:7]).sum(-1)),0,1));joint=np.abs(pose[i,:,7:]-actual[:,7:]);times=z['time'][idx[i]]
   actual_contact=z['contact'][idx[i]];transition=np.r_[False,np.any(np.diff(actual_contact,axis=0)!=0,axis=-1)|(np.diff(z['phase'][idx[i]])!=0)]
   transition=np.convolve(transition.astype(int),np.ones(5,dtype=int),mode='same')>0
   far=np.flatnonzero(contacts[i,:,1]>=.5);handoff=np.flatnonzero(phases[i]==2)
   entry=dict(map=r['map'],case_dir=r['case_dir'],update=saved['update'],samples=length,termination=r['termination'],max_xyz_m=float(error.max()),mean_xyz_m=float(error.mean()),endpoint_xyz_m=float(error[-1]),rotation_mean_rad=float(rotation.mean()),joint_mae_rad=joint.mean(0).tolist(),velocity_rmse=np.sqrt(np.mean((vel[i]-z['qvel'][idx[i]])**2,axis=0)).tolist(),transition_xyz_mean_m=float(error[transition].mean()) if transition.any() else None,far_contact_time_pred_s=None if not len(far) else float(times[far[0]+1]),far_contact_time_actual_s=r['first_far_contact_s'],handoff_time_pred_s=None if not len(handoff) else float(times[handoff[0]+1]),handoff_time_actual_s=r['handoff_s'])
   records.append(entry);key=r['map']+'_'+r['case_dir'];arrays[key+'_actual']=actual;arrays[key+'_predicted']=pose[i];arrays[key+'_time']=times;arrays[key+'_phase']=np.r_[z['phase'][r['start']],phases[i]]
np.savez_compressed(a.output/'predictions.npz',**arrays)
(a.output/'cases.json').write_text(json.dumps(records,indent=2)+'\n')
summary=dict(cases=len(records),model=str(a.model),update=saved['update'],mean_max_xyz_m=float(np.mean([r['max_xyz_m'] for r in records])),worst_max_xyz_m=max(r['max_xyz_m'] for r in records),mean_endpoint_xyz_m=float(np.mean([r['endpoint_xyz_m'] for r in records])),missing_predicted_handoff=sum(r['handoff_time_pred_s'] is None for r in records),actual_handoffs=sum(r['handoff_time_actual_s'] is not None for r in records),scope='True entry, then free full RB segment through actual record duration; fixed horizon is evaluation only, never supplied to predictor as an input; no full-chain or decision benefit claim')
(a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary))
