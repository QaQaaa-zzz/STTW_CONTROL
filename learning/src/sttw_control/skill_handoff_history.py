"""Apply the frozen execution's controller reset to predicted handoff states."""
from types import SimpleNamespace
from pathlib import Path
import json,math
import torch
from .torch_controller import TorchController
from .torch_middle_control import TorchMiddleControl
from .torch_path_reference import TorchPathReference
from .playground_prediction import euler


class HandoffHistoryReset:
    def __init__(self,config,controller_schema,model_schema,bank,route_ids,device):
        hand=config['ground_handoff'];self.schema=controller_schema;self.model_schema=model_schema
        hcfg=json.loads((Path(hand['bundle'])/'config.json').read_text())
        hcfg['follower'].update(hand.get('follower_overrides',{}))
        if 'nominal_steer_slew_rad_s' in hand:hcfg['follower']['nominal_steer_slew_rad_s']=hand['nominal_steer_slew_rad_s']
        hcfg['follower']['lookahead_scale']=hand['handoff_variants'][0].get('lookahead_scale',1.)
        self.route=TorchPathReference(dict(bank),hcfg,device,route_ids=route_ids)
        ccfg=json.loads(Path(json.loads(Path(hand['settings']).read_text())['controller_config']).read_text())
        self.controller=TorchController(ccfg,len(route_ids),device)

    def __call__(self,qpos,qvel,history,speed_request):
        state={};offset=0
        for name,shape in self.schema.items():
            size=math.prod(shape);state[name]=history[:,offset:offset+size].reshape(len(history),*shape);offset+=size
        angles=euler(qpos[:,3:7]);s=self.model_schema
        measurement=torch.stack((-angles[:,0],-qvel[:,3],qpos[:,s['steer_qpos']],qvel[:,s['steer_dof']],qvel[:,5],qvel[:,s['rear_dof']],qvel[:,s['front_dof']]),1)
        pose=torch.stack((qpos[:,0],qpos[:,1],angles[:,2]),1)
        speed=qvel[:,0]*angles[:,2].cos()+qvel[:,1]*angles[:,2].sin()
        m=SimpleNamespace(**state)
        m.policy=SimpleNamespace(offsets=state['policy_offsets'],filtered=state['policy_filtered'])
        m.route=self.route;m.v_user=speed_request
        m.projection={k.removeprefix('projection_'):v for k,v in state.items() if k.startswith('projection_')}
        m.follower={k.removeprefix('follower_'):v for k,v in state.items() if k.startswith('follower_')}
        m.fault=m.follower['numerical_fault'];m.domain_exit=m.follower['invalid']
        for key in ['eso','filtered_roll_rate','disturbance','gains']:setattr(self.controller,key,state[key])
        mask=torch.ones(len(qpos),device=qpos.device,dtype=torch.bool)
        TorchMiddleControl.reset_mask(m,mask,measurement,pose,speed,self.controller)
        for name in state:
            if name.startswith('projection_'):state[name]=m.projection[name.removeprefix('projection_')]
            elif name.startswith('follower_'):state[name]=m.follower[name.removeprefix('follower_')]
            elif name.startswith('policy_'):state[name]=getattr(m.policy,name.removeprefix('policy_'))
            elif name in ['eso','filtered_roll_rate','disturbance','gains']:state[name]=getattr(self.controller,name)
            else:state[name]=getattr(m,name)
        state['recovering']=torch.ones_like(state['recovering'])
        state['lock']=qpos[:,[s['qpos_names'].index('hip_joint'),s['qpos_names'].index('knee_joint')]]
        return torch.cat([value.reshape(len(qpos),-1) for value in state.values()],-1)
