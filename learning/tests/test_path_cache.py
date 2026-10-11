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


def _fork_cache_fixture(tmp_path,alpha=0):
 import copy,shutil,torch
 from sttw_control.preference_best import config_hash
 from sttw_control.path_command_fork import validate_fork_contract
 from test_path_fork import fork_inputs
 plan,cfg,source=fork_inputs();receipt=validate_fork_contract(plan,cfg,source)
 record=receipt['identities'][f'alpha{alpha}'];identity=copy.deepcopy(receipt['original_identities'][f'alpha{alpha}'])
 identity.update(config_sha256=config_hash(cfg),sampler_sha256=config_hash(plan['sampler_resolution']),
                 parent_source_update=record['source_update'],parent_checkpoint_sha256=record['checkpoint_sha256'])
 out=tmp_path/f'alpha{alpha}';folder=out/'checkpoints';folder.mkdir(parents=True)
 checkpoint=folder/'update_0000.pt';saved=torch.load(record['checkpoint'],map_location='cpu',weights_only=False)
 saved.update(identity=identity,update=0);torch.save(saved,checkpoint)
 shutil.copyfile(record['actor'],folder/'actor_0000.pkl')
 return receipt,out,cfg,identity,checkpoint


@pytest.mark.parametrize('alpha',[0,1])
def test_fork_parent_reuse_registers_new_identity_and_selected_dev_without_physics(tmp_path,alpha):
 from sttw_control.path_command_cache import reuse_fork_parent,reuse_selected_validation
 from sttw_control.preference_best import digest,config_hash
 receipt,out,cfg,identity,checkpoint=_fork_cache_fixture(tmp_path,alpha)
 traces=reuse_fork_parent(receipt,out,alpha,cfg,identity,checkpoint)
 assert len(traces)==6
 for pointer in ('best_model.json','best_task_model.json'):
  best=json.loads((out/pointer).read_text())
  assert best['source_update']==0
  assert best['config_sha256']==config_hash(cfg)
  assert best['checkpoint_sha256']==digest(checkpoint)
  assert best['checkpoint_sha256']!=receipt['identities'][f'alpha{alpha}']['checkpoint_sha256']
 before={p.relative_to(out):digest(p) for p in out.rglob('*') if p.is_file()}
 again=reuse_fork_parent(receipt,out,alpha,cfg,identity,'unused_when_read_only',register=False)
 assert set(again)==set(traces)
 assert before=={p.relative_to(out):digest(p) for p in out.rglob('*') if p.is_file()}
 best=json.loads((out/'best_task_model.json').read_text())
 assert reuse_selected_validation(out,best,cfg)==set(traces)
 for name in traces:
  for suffix in ('','_path'):
   filename=name+suffix+'.npz'
   assert digest(out/'final_best'/filename)==digest(out/'validation'/'update_0000'/filename)
 bad=dict(best,actor_sha256='wrong')
 with pytest.raises(ValueError,match='source mismatch'):reuse_selected_validation(out,bad,cfg)
 # An incomplete selected panel must not be silently re-evaluated or reused.
 (out/'validation'/'update_0000'/'straight_2.0.npz').unlink()
 with pytest.raises(ValueError,match='missing complete cached case'):reuse_selected_validation(out,best,cfg)


def test_fork_cache_rejects_wrong_parent_receipt_and_wrong_initial_actor(tmp_path):
 import copy,pickle
 from sttw_control.path_command_cache import reuse_fork_parent
 receipt,out,cfg,identity,checkpoint=_fork_cache_fixture(tmp_path)
 changed=copy.deepcopy(receipt);changed['identities']['alpha0']['source_update']+=1
 with pytest.raises(ValueError,match='parent final panel source'):reuse_fork_parent(changed,out,0,cfg,identity,checkpoint)
 actor=checkpoint.parent/'actor_0000.pkl'
 with actor.open('rb') as f:params=pickle.load(f)
 params['params']['Dense_0']['kernel'][0,0]+=1
 with actor.open('wb') as f:pickle.dump(params,f)
 with pytest.raises(ValueError,match='actor'):reuse_fork_parent(receipt,out,0,cfg,identity,checkpoint)
 assert not (out/'best_model.json').exists()
 assert not (out/'validation').exists()
