"""Actual-checkpoint GPU serialization and frozen-physics parity for P3 events."""
import argparse,io,json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from sttw_control.skill_world_model import SkillWorldModel
from sttw_control.skill_event_readout import SkillEventReadout,EventWorldModel
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);a=p.parse_args();torch.set_num_threads(1);contract=json.loads((a.run/'contract.json').read_text());data=Path(contract['data']);manifest=json.loads((data/'manifest.json').read_text());row=next(r for r in manifest['rows'] if r['split']=='DEV');z=np.load(data/'sequences.npz');n=np.load(data/'normalization.npz');j=row['start'];t=lambda x:torch.as_tensor(x,device='cuda',dtype=torch.float32)
world=SkillWorldModel(537,5,*[n[k] for k in ['input_mean','input_scale','velocity_scale','history_scale']]).cuda();world.load_state_dict(torch.load(contract['world_model'],map_location='cuda',weights_only=True)['model']);saved=torch.load(a.run/'seed1/best.pt',map_location='cuda',weights_only=True);readout=SkillEventReadout(saved['model']['mean'],saved['model']['scale']).cuda();readout.load_state_dict(saved['model']);model=EventWorldModel(world,readout);context=t(z['context'][j:j+1]);state=model.initialize(t(z['qpos'][j:j+1]),t(z['qvel'][j:j+1]),t(z['history'][j:j+1,:381]),t(z['history'][j:j+1,381:]),t(z['contact'][j:j+1]),F.one_hot(torch.tensor([int(z['phase'][j])],device='cuda'),5).float(),context)
with torch.no_grad():
 full=model.rollout(state,context,100);first=model.rollout(state,context,40);buffer=io.BytesIO();torch.save(first['next_state'],buffer);buffer.seek(0);resumed=model.rollout(torch.load(buffer,weights_only=False),context,60);base=world.rollout(state['world'],context,100)
 joined_q=torch.cat((first['qpos'],resumed['qpos'][:,1:]),1);joined_e=torch.cat((first['event_probability'],resumed['event_probability']),1)
 result=dict(case=row['map']+'/'+row['case_dir'],world_model=contract['world_model'],event_model=str(a.run/'seed1/best.pt'),device='cuda',steps=[100,40,60],qpos_resume_max=float((full['qpos']-joined_q).abs().max()),event_resume_max=float((full['event_probability']-joined_e).abs().max()),physical_change_max=float((full['qpos']-base['qpos']).abs().max()))
 assert result['qpos_resume_max']==0 and result['event_resume_max']==0 and result['physical_change_max']==0
(a.run/'resume_verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
