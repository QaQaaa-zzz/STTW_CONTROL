from pathlib import Path
import copy,json
import numpy as np
import pytest
import jax,jax.numpy as j
from sttw_control.path_command_curriculum_run import declaration,validate
from sttw_control.path_command_curriculum import route_for_stage,promotion,panel_summary
from sttw_control.path_command_distillation import inheritance_gate
ROOT=Path(__file__).parents[2]
def config():return json.loads((ROOT/'learning/configs/STTW_Path_Distilled_Curriculum_V3.json').read_text())

def test_400_budget_counts_separate_supervision_and_ppo():
 c=config();validate(c);d=declaration(c)
 assert d['ppo_updates']==400 and d['ppo_policy_steps']==26214400
 assert d['bc_updates_max']==12000 and d['total_lower_tick_cap']==106882400
 assert d['lower_tick_caps']['training']==104857600


def test_stage_reset_ranges_fresh_previous_examples_and_shape():
 c=config();ids=j.arange(102,306,dtype=j.int32)
 f=jax.jit(jax.vmap(lambda eid,s:route_for_stage(88,eid,j.int32(0),j.zeros(3),c,s),in_axes=(0,None)))
 for stage in (0,1,2,3):
  r=f(ids,j.int32(stage));s=np.asarray(r.speed);rad=np.asarray(r.radius)
  assert r.path.s.shape==(204,5001)
  lo=min(c['curriculum']['stages'][max(stage-1,0)]['speed_m_s']);hi=max(c['curriculum']['stages'][stage]['speed_m_s'])
  assert np.all((s>=lo)&(s<=hi))
  if stage>0:assert np.any(s<c['curriculum']['stages'][stage]['speed_m_s'][0])
  assert np.all(np.isfinite(rad))


def test_promotion_requires_two_passes_minimum_exposure_and_no_fake_gap():
 c=config()['curriculum'];ok=dict(common_qualified=True,endpoint_primary_qualified=True);bad=dict(common_qualified=False,endpoint_primary_qualified=True)
 assert promotion(0,25,0,ok,c)==(0,1)
 assert promotion(0,50,1,ok,c)==(1,0)
 assert promotion(1,50,1,bad,c)==(1,0)
 assert promotion(3,100,1,ok,c)==(3,2)
 assert promotion(0,50,1,dict(common_qualified=True,endpoint_primary_qualified=False),c)==(1,0)
 assert promotion(1,50,1,dict(common_qualified=True,endpoint_primary_qualified=False),c)==(1,0)


def test_inheritance_gate_accepts_original_teachers_as_students():
 base=Path(config()['teacher_run'])
 if not base.exists():pytest.skip('local frozen teacher trajectories unavailable')
 cfg=json.loads((base/'manifest.json').read_text())['config']
 panels={f'alpha_{a}':{p.stem:dict(np.load(p)) for p in (base/f'alpha{a}/final_best').glob('*.npz') if not p.stem.endswith('_path')} for a in (0,1)}
 gate=inheritance_gate(panels,base,cfg)
 assert gate['passed'] and len(gate['cases'])==12
 bad=copy.deepcopy(panels);case=next(iter(bad['alpha_0']));bad['alpha_0'][case]['physical_failure'][-1]=True
 # A real failure at the last tick is a completed, failed observation, never inherited success.
 assert not inheritance_gate(bad,base,cfg)['passed']
 cfg['training_future_phase_C']['episode_s']=10.
 common=inheritance_gate(panels,base,cfg)
 assert common['passed'] and all(x['metrics']['comparison_seconds']==10 for x in common['cases'])
 assert all(len(d['time'])==4000 for group in panels.values() for d in group.values())


def test_validation_ids_do_not_leak_when_collection_width_changes():
 from sttw_control.path_command_distillation import validation_env
 teacher=np.r_[np.arange(12),np.arange(102,126),np.arange(306,334)]
 student=np.r_[np.arange(8),np.arange(102,114),np.arange(306,318)]
 teacher_val=set(teacher[validation_env(teacher)]);teacher_train=set(teacher[~validation_env(teacher)])
 student_val=set(student[validation_env(student)]);student_train=set(student[~validation_env(student)])
 assert not (teacher_val&student_train or teacher_train&student_val)
 assert len(student_val)>0 and len(student_train)>0
