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
    def __init__(self, directory):
        self.directory=Path(directory)
        self.meta=json.loads((self.directory/'model.json').read_text())
        self.map=KnownHeightMap(np.load(self.meta['terrain_npz'])['vertices'],self.meta['terrain_spec'])
        self.downstream=HandoffContinuationModel(self.meta['downstream_directory'])
        if self.downstream.target!='goal':raise ValueError('Full-task composition requires conditional goal predictor')
        self.members=[]
        for path in sorted(self.directory.glob('upstream_*.pt')):
            c=torch.load(path,map_location='cpu',weights_only=False);m=network(len(c['input_mean']),80)
            m.load_state_dict(c['state_dict']);m.eval();self.members.append(m);self.c=c
        if len(self.members)!=3:raise ValueError('Expected three upstream members')
        self.direct=[]
        for path in sorted(self.directory.glob('direct_*.pt')):
            c=torch.load(path,map_location='cpu',weights_only=False);m=network(len(c['input_mean']),1)
            m.load_state_dict(c['state_dict']);m.eval();self.direct.append(m)
        if len(self.direct)!=3:raise ValueError('Expected three direct comparator members')

    @torch.no_grad()
    def score_direct(self,root,observation,history,time_s,parameters,particles=32):
        raw=inputs(root,observation,history,time_s,parameters,self.map.crop(root))
        xx=torch.tensor((raw-self.c['input_mean'])/self.c['input_std'])
        value=np.mean([torch.sigmoid(m(xx)[:,0]).numpy() for m in self.direct],axis=0)
        return dict(score=value.tolist(),predicted_completion=value.tolist(),scope='Direct full-goal comparator; same query input and grouped training data')

    @torch.no_grad()
    def score(self,root,observation,history,time_s,parameters,particles=32,seed=71101):
        raw=inputs(root,observation,history,time_s,parameters,self.map.crop(root))
        xx=torch.tensor((raw-self.c['input_mean'])/self.c['input_std'])
        rng=np.random.default_rng(seed);jump=parameters[:,-1]>.5;values=[];handoffs=[];means=[];within=[]
        for member in self.members:
            out=member(xx).numpy();mu=out[:,:39];logvar=out[:,39:78].clip(-4,3)
            # Common particles across candidates reduce arbitrary sampling differences.
            noise=rng.standard_normal((particles,39)).astype(np.float32)
            sampled=(mu[:,None]+np.exp(.5*logvar[:,None])*noise[None])*self.c['target_std']+self.c['target_mean']
            q=sampled[:,:,3:7];sampled[:,:,3:7]=q/np.maximum(np.linalg.norm(q,axis=-1,keepdims=True),1e-8)
            sampled[:,:,38]=np.maximum(sampled[:,:,38],np.asarray(time_s)[:,None])
            sampled[:,:,37]=1. # Latest measured-frame validity flag.
            flat=sampled.reshape(-1,39);obs=np.zeros((len(flat),80),np.float32);obs[:,50:75]=flat[:,13:38]
            downstream=self.downstream.predict(flat[:,:13],obs,flat[:,38])['conditional_goal_probability'].reshape(len(root),particles).mean(1)
            ph=1/(1+np.exp(-out[:,78].clip(-30,30)));pg=1/(1+np.exp(-out[:,79].clip(-30,30)))
            values.append(np.where(jump,ph*downstream,pg));handoffs.append(ph);within.append(downstream)
            means.append(mu*self.c['target_std']+self.c['target_mean'])
        return dict(score=np.mean(values,axis=0).tolist(),predicted_completion=np.mean(values,axis=0).tolist(),
                    valid_handoff_probability=np.mean(handoffs,axis=0).tolist(),
                    conditional_goal_probability=np.mean(within,axis=0).tolist(),
                    predicted_handoff_mean=np.mean(means,axis=0).tolist(),
                    member_score_std=np.std(values,axis=0).tolist(),
                    scope='Fixed map/source jump/cold H75; composed goal probability estimate, not certified feasibility. Ground routes use separate direct branch.')
