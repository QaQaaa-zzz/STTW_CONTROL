"""Baseline cache identity contract, without vehicle physics."""
import json
from pathlib import Path
import numpy as np
import pytest
from sttw_control.path_command_cache import reuse_zero_upper
from sttw_control.path_command_selection import cases
from sttw_control.geometric_path import build_path
CFG=json.loads((Path(__file__).parents[1]/'configs/STTW_Path_Feedback_V1.json').read_text())

def test_cached_baseline_requires_same_identity_path_and_commands(tmp_path):
 cfg=json.loads(json.dumps(CFG));cfg['training_future_phase_C']['episode_s']=.02
 parent=tmp_path/'parent';(parent/'zero_upper').mkdir(parents=True);source={'bank':'same','lower':'fixed300'}
 (parent/'manifest.json').write_text(json.dumps({'config':cfg,'source':source}))
 paths={route:build_path(route,np.zeros(3),cfg) for route in cfg['stage_C_evaluation']['route_ids']}
 for name,route,speed in cases(cfg):
  d=dict(time=np.arange(4)*.005,v_user=np.full(4,speed),offsets=np.zeros((4,2)),filtered_offset=np.zeros((4,2)),target_offset=np.zeros((4,2)))
  d.update({k:np.zeros(4,bool) for k in ['physical_failure','domain_exit','policy_fault','lower_fault']})
  np.savez(parent/'zero_upper'/f'{name}.npz',**d)
  np.savez(parent/'zero_upper'/f'{name}_path.npz',**{k:np.asarray(getattr(paths[route],k)) for k in ['s','xy','heading','curvature','goal','turn_end']})
 assert len(reuse_zero_upper(parent,tmp_path/'new',cfg,source,paths))==6
 with pytest.raises(ValueError):reuse_zero_upper(parent,tmp_path/'wrong',cfg,{'lower':'400'},paths)
 file=parent/'zero_upper'/'straight_2.0.npz';d=dict(np.load(file));d['v_user'][:]=2.6;np.savez(file,**d)
 with pytest.raises(AssertionError):reuse_zero_upper(parent,tmp_path/'changed',cfg,source,paths)
