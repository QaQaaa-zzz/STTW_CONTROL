"""Direct completion probability baseline with the event model's input codec."""
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .ditch_event_model import EventPlanner


def completion_network(dimension):
    return nn.Sequential(nn.Linear(dimension,64),nn.SiLU(),nn.Linear(64,32),nn.SiLU(),nn.Linear(32,1))


class CompletionPlanner(EventPlanner):
    def __init__(self,directory):
        directory=Path(directory);metadata=json.loads((directory/'model.json').read_text())
        super().__init__(metadata['event_model'])
        weights=torch.load(directory/'model.pt',map_location='cpu',weights_only=False)
        self.mean=weights['mean'];self.std=weights['std'];self.model=completion_network(len(self.mean))
        self.model.load_state_dict(weights['state_dict']);self.model.eval()
    @torch.no_grad()
    def score(self,state,parameters,particles=16,seed=20261110):
        features=np.concatenate((state,self.plan_features(state,parameters),self.map.crop(state[:,:13])),-1)
        probability=self.model(torch.from_numpy((features-self.mean)/self.std)).sigmoid().flatten().numpy()
        return dict(score=probability.tolist(),predicted_completion=probability.tolist(),scope='Direct completion classifier; no event prediction or physical feasibility certificate')
