"""Two-interface conditional prediction, plus a ground-route outcome branch."""
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .ditch_event_model import KnownHeightMap
from .handoff_model import HandoffContinuationModel


def network(inputs, outputs):
    return nn.Sequential(nn.Linear(inputs,128),nn.SiLU(),nn.Linear(128,64),nn.SiLU(),nn.Linear(64,outputs))


def inputs(root, observation, history, time_s, parameters, crop):
    return np.concatenate((root,observation,history,np.asarray(time_s).reshape(-1,1),parameters,crop),axis=1).astype(np.float32)


class InterfaceWorldModel:
    def __init__(self, directory, task_config=None):
        self.directory=Path(directory)
        self.meta=json.loads((self.directory/'model.json').read_text())
        self.map=KnownHeightMap(np.load(self.meta['terrain_npz'])['vertices'],self.meta['terrain_spec'])
        self.downstream=HandoffContinuationModel(self.meta['downstream_directory'])
        if self.downstream.target!='goal':raise ValueError('Full-task composition requires conditional goal predictor')
        self.task_config=task_config
        if task_config is not None:
            self.task_downstream=HandoffContinuationModel(task_config['model'],feature_set='task_context')
            declaration=json.loads((Path(task_config['model'])/'declaration.json').read_text());self.task_mount=declaration['mount_wxyz'];self.task_representation=declaration.get('context_representation','world_x')
            bank=np.load(task_config['routes'])
            self.task_routes=[{key:bank[key][i] for key in ('xy','heading','curvature')} for i in task_config['route_ids']]
        self.ground=[]
        if 'ground_directory' in self.meta:
            for path in sorted(Path(self.meta['ground_directory']).glob('ground_*.pt')):
                checkpoint=torch.load(path,map_location='cpu',weights_only=False)
                member=network(len(checkpoint['input_mean']),1)
                member.load_state_dict(checkpoint['state_dict']);member.eval()
                self.ground.append((member,checkpoint['input_mean'],checkpoint['input_std']))
            if len(self.ground)!=3:raise ValueError('Expected three independent ground members')
        self.members=[]
        for path in sorted(self.directory.glob('upstream_*.pt')):
            c=torch.load(path,map_location='cpu',weights_only=False);m=(ActionInjectedNetwork if 'compact_transform' in c else network)(len(c['input_mean']),80)
            m.load_state_dict(c['state_dict']);m.eval();self.members.append(m);self.c=c
        if len(self.members)!=3:raise ValueError('Expected three upstream members')
        self.direct=[]
        for path in sorted(self.directory.glob('direct_*.pt')):
            c=torch.load(path,map_location='cpu',weights_only=False);m=network(len(c['input_mean']),1)
            m.load_state_dict(c['state_dict']);m.eval();self.direct.append(m);self.direct_c=c
        if len(self.direct)!=3:raise ValueError('Expected three direct comparator members')

    @torch.no_grad()
    def score_direct(self,root,observation,history,time_s,parameters,particles=32):
        raw=inputs(root,observation,history,time_s,parameters,self.map.crop(root))
        xx=torch.tensor((raw-self.direct_c['input_mean'])/self.direct_c['input_std'])
        value=np.mean([torch.sigmoid(m(xx)[:,0]).numpy() for m in self.direct],axis=0)
        return dict(score=value.tolist(),predicted_completion=value.tolist(),scope='Direct full-goal comparator; same query input and grouped training data')

    @torch.no_grad()
    def score(self,root,observation,history,time_s,parameters,particles=32,seed=71101,task_conditioned=False):
        raw=inputs(root,observation,history,time_s,parameters,self.map.crop(root))
        transform=self.c.get('compact_transform');encoded=compact_inputs(raw,transform) if transform is not None else raw
        baseline=handoff_baseline(raw,transform) if transform is not None else np.zeros((len(raw),39),np.float32)
        xx=torch.tensor((encoded-self.c['input_mean'])/self.c['input_std'])
        route_values=[];route_conditionals=[]
        rng=np.random.default_rng(seed);jump=parameters[:,-1]>.5;values=[];handoffs=[];means=[];within=[]
        for member in self.members:
            out=member(xx).numpy();mu=out[:,:39];logvar=out[:,39:78].clip(-4,3)
            # Common particles across candidates reduce arbitrary sampling differences.
            noise=rng.standard_normal((particles,39)).astype(np.float32)
            sampled=(mu[:,None]+np.exp(.5*logvar[:,None])*noise[None])*self.c['target_std']+self.c['target_mean']+baseline[:,None]
            q=sampled[:,:,3:7];sampled[:,:,3:7]=q/np.maximum(np.linalg.norm(q,axis=-1,keepdims=True),1e-8)
            sampled[:,:,38]=np.maximum(sampled[:,:,38],np.asarray(time_s)[:,None])
            sampled[:,:,37]=1. # Latest measured-frame validity flag.
            flat=sampled.reshape(-1,39);obs=np.zeros((len(flat),80),np.float32);obs[:,50:75]=flat[:,13:38]
            if task_conditioned:
                from .handoff_task_context import route_context
                if not np.all(jump):raise ValueError('Task-conditioned extension currently covers jump candidates only')
                per_route=[]
                for route in self.task_routes:
                    context=route_context(flat[:,:13],route,self.task_config['half_width_m'],self.task_mount,getattr(self,'task_representation','world_x'))
                    probability=self.task_downstream.predict(flat[:,:13],obs,flat[:,38],context)['conditional_goal_probability']
                    per_route.append(probability.reshape(len(root),particles).mean(1))
                conditional_routes=np.stack(per_route,axis=1)
                downstream=conditional_routes.max(1)
            else:
                downstream=self.downstream.predict(flat[:,:13],obs,flat[:,38])['conditional_goal_probability'].reshape(len(root),particles).mean(1)
            ph=1/(1+np.exp(-out[:,78].clip(-30,30)));pg=1/(1+np.exp(-out[:,79].clip(-30,30)))
            if task_conditioned:
                route_values.append(ph[:,None]*conditional_routes);route_conditionals.append(conditional_routes)
            values.append(np.where(jump,ph*downstream,pg));handoffs.append(ph);within.append(downstream)
            means.append(mu*self.c['target_std']+self.c['target_mean']+baseline)
        if self.ground:
            ground_probability=np.mean([torch.sigmoid(member(torch.tensor((raw-mean)/std))[:,0]).numpy()
                                        for member,mean,std in self.ground],axis=0)
            values=[np.where(jump,value,ground_probability) for value in values]
        result=dict(score=np.mean(values,axis=0).tolist(),predicted_completion=np.mean(values,axis=0).tolist(),
                    valid_handoff_probability=np.mean(handoffs,axis=0).tolist(),
                    conditional_goal_probability=np.mean(within,axis=0).tolist(),
                    predicted_handoff_mean=np.mean(means,axis=0).tolist(),
                    member_score_std=np.std(values,axis=0).tolist(),
                    scope='Fixed map/source jump/cold H75; composed goal probability estimate, not certified feasibility. Ground routes use separate direct branch.')

        if task_conditioned:
            # Choose a single route after averaging all particles and members.
            # E[max_route outcome] would grant unavailable sample-specific foresight.
            joint=np.mean(route_values,axis=0);selected=joint.argmax(1)
            result.update(score=joint.max(1).tolist(),predicted_completion=joint.max(1).tolist(),
                          downstream_route_scores=joint.tolist(),
                          conditional_goal_probability=np.mean(route_conditionals,axis=0)[np.arange(len(root)),selected].tolist(),
                          member_score_std=np.std(route_values,axis=0)[np.arange(len(root)),selected].tolist(),
                          selected_downstream_route=np.asarray(self.task_config['route_ids'])[selected].tolist(),
                          scope='Frozen upstream distribution composed with task-conditioned downstream outcomes; one route selected per approach before execution; no sample-wise hindsight')
        return result


class ActionInjectedNetwork(nn.Module):
    """Compact normalized state encoder with an explicit action skip connection."""
    def __init__(self, inputs, outputs):
        super().__init__()
        self.encoder=nn.Sequential(nn.Linear(inputs,128),nn.SiLU(),nn.Linear(128,64),nn.SiLU())
        self.readout=nn.Linear(64+8,outputs)
    def forward(self,x):return self.readout(torch.cat((self.encoder(x),x[:,-8:]),dim=-1))


def compact_inputs(raw,transform):
    history=(raw[:,13:445]-transform['history_mean'])/transform['history_std']
    principal=history@transform['history_components'].T
    # Eight pooled heights and eight pooled support-mask values.
    terrain=raw[:,454:582].reshape(-1,2,2,4,4,2).mean(axis=(3,5)).reshape(-1,16)
    return np.concatenate((raw[:,:13],principal,transform['jump_start_x_m']-raw[:,:1],terrain,raw[:,445:446],raw[:,446:454]),1).astype(np.float32)


def handoff_baseline(raw,transform):
    from .route_horizon import remaining_route_length
    parameters=raw[:,446:454];bank=transform['roads']
    # Route geometry repeats over speed/alpha. Select by the stored geometric encoding.
    geom=transform['parameters'][:,:5]
    selected=((parameters[:,None,:5]-geom[None])**2).sum(-1).argmin(1)
    lengths=remaining_route_length(raw[:,:13],bank,dict(x_min_m=transform['jump_start_x_m'],x_max_m=30.,abs_y_max_m=100.))[np.arange(len(raw)),selected]
    baseline=np.zeros((len(raw),39),np.float32);baseline[:,7]=parameters[:,5] if transform.get('velocity_baseline','requested')=='requested' else 0.
    baseline[:,38]=raw[:,445]+lengths/parameters[:,5]
    return baseline
