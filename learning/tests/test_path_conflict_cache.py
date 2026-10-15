"""Real immutable baseline sources, CPU-only cache identity checks."""
import json
from pathlib import Path
import shutil
import numpy as np
import pytest
from sttw_control.path_command_training import load_plan
from sttw_control.path_command_cache import reuse_conflict_baseline
from sttw_control.path_command_scenarios import explicit_case_declarations
from sttw_control.geometric_path import build_path
from sttw_control.preference_best import atomic_json
ROOT=Path(__file__).parents[2]


def test_conflict_baseline_reuses_old_cases_but_rejects_script_or_geometry(tmp_path):
 _,cfg,source=load_plan(ROOT/'learning/configs/path_conflict_exposure_run.json')
 old,_,_=load_plan(ROOT/'learning/configs/path_unified_scratch_run.json');old_parent=Path(old['baseline_parent'])
 raw=dict(np.load(old_parent/'zero_upper/straight_2.0_path.npz'));pose=np.r_[raw['xy'][0],raw['heading'][0]]
 paths={d['route_id']:build_path(d['route_id'],pose,cfg) for d in explicit_case_declarations(cfg)}
 initial=tmp_path/'initial';initial.mkdir()
 names=reuse_conflict_baseline(old_parent,initial,cfg,source,paths);assert len(names)==6
 parent=tmp_path/'parent';(parent/'zero_upper').mkdir(parents=True);atomic_json(parent/'manifest.json',dict(config=cfg,source=source))
 for name in ('straight_2.0.npz','straight_2.0_path.npz'):shutil.copyfile(initial/'zero_upper'/name,parent/'zero_upper'/name)
 target=tmp_path/'target';target.mkdir();assert reuse_conflict_baseline(parent,target,cfg,source,paths)=={'straight_2.0'}
 file=parent/'zero_upper/straight_2.0.npz';d=dict(np.load(file));d['speed_script_switch_times']=np.array([1.,5.]);np.savez_compressed(file,**d)
 with pytest.raises(AssertionError):reuse_conflict_baseline(parent,tmp_path/'bad_script',cfg,source,paths)
 shutil.copyfile(initial/'zero_upper/straight_2.0.npz',file)
 pathfile=parent/'zero_upper/straight_2.0_path.npz';d=dict(np.load(pathfile));d['xy'][5,1]+=.1;np.savez_compressed(pathfile,**d)
 with pytest.raises(AssertionError):reuse_conflict_baseline(parent,tmp_path/'bad_path',cfg,source,paths)
