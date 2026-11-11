"""Frozen, fixed-context downstream continuation ensemble."""
from pathlib import Path
import numpy as np
import torch
from torch import nn


class HandoffContinuationModel:
    """Predict observed two-second continuation; not full-goal feasibility.

    Scope: original cold ground reset,1.5m/s,alpha0.5,straight road and
    the declared source terrain/controller. Future terrain/initializer changes
    require explicit conditioning and new validation.
    """
    def __init__(self, directory, feature_set='state_history'):
        directory=Path(directory);self.models=[]
        for path in sorted(directory.glob(f'{feature_set}_*.pt')):
            checkpoint=torch.load(path,map_location='cpu',weights_only=False)
            columns=checkpoint['columns']
            model=nn.Sequential(nn.Linear(len(columns),64),nn.SiLU(),nn.Linear(64,32),nn.SiLU(),nn.Linear(32,1))
            model.load_state_dict(checkpoint['state_dict']);model.eval()
            self.models.append((model,columns))
            self.mean=checkpoint['mean'];self.std=checkpoint['std']
        if len(self.models)!=3:raise ValueError('Expected three frozen ensemble members')

    @torch.no_grad()
    def predict(self, root, current_observation, physical_time_s):
        features=np.concatenate((root,current_observation[:,50:75],np.asarray(physical_time_s).reshape(-1,1)),axis=1).astype(np.float32)
        normalized=torch.from_numpy((features-self.mean)/self.std)
        samples=np.stack([torch.sigmoid(model(normalized[:,columns]).squeeze(-1)).numpy() for model,columns in self.models])
        return dict(continuation_probability=samples.mean(0),ensemble_std=samples.std(0))
