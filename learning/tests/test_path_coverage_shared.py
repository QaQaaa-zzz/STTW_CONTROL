"""User-approved original Coverage reward with a fresh shared alpha policy."""
import copy,json
from pathlib import Path
import jax.numpy as j
import pytest
from sttw_control.path_command_training import load_plan,prepare
from sttw_control.path_command_conditioning import conditional_identity,validate_conditional_plan
from sttw_control.path_command_reward import costs,reward_version
ROOT=Path(__file__).parents[2]
PLAN=ROOT/'learning/configs/path_coverage_shared_run.json'

def test_original_reward_is_independent_of_modulated_architecture(tmp_path):
 p,c,s=load_plan(PLAN)
 original=json.loads((ROOT/'learning/configs/STTW_Path_Coverage_V2.json').read_text())
 assert c['path_reward']==original['path_reward'] and c['training_sampler']==original['training_sampler']
 assert 'reward_version' not in c['path_reward'] and reward_version(c) is None
 assert c['network']['architecture']=='alpha_modulated_v1'
 assert c['alpha_conditioning']['endpoint_probability_each']==.4 and c['alpha_conditioning']['uniform_probability']==.2
 identity=conditional_identity(p,c,s)
 assert identity['reward_version'] is None and identity['initialization_mode']=='scratch'
 assert not c['dynamic_preference_checks']['enabled'] and not c.get('preference_selection_schema')
 report=prepare(PLAN,tmp_path/'declaration')
 assert report['training_lower_ticks_total']==52428800
 assert report['maximum_validation_lower_ticks']==600000
 assert report['maximum_dynamic_lower_ticks']==0 and report['complete_experiment_lower_tick_budget']==53029312
 for alpha,expected in [(0,0),(.5,2.5),(1,5)]:
  out=costs(alpha=alpha,chi=1.,ev=.1,ey=.1,eh=0.,peak_roll=.3,roll_rate=0.,speed=2.,offsets=j.zeros(2),offset_rate=j.zeros(2),config=c)
  assert float(out['raw']['primary'])==pytest.approx(expected,abs=1e-5)
  assert all(float(v)==0 for v in out['diagnostic_subcosts'].values())
 bad=copy.deepcopy(c);bad['path_reward']['reward_version']=None
 with pytest.raises(ValueError,match='reward version'):validate_conditional_plan(p,bad)

def test_original_reward_does_not_require_other_endpoint_to_worsen():
 from test_path_conditional_reporting import panels
 from sttw_control.path_command_selection import summarize_conditional
 _,c,_=load_plan(PLAN)
 result=summarize_conditional(panels(),c)
 assert result['corridor_role']=='diagnostic'
 assert result['qualified_preference'] is None
 assert 'corridor_failures' not in result['ranking_fields']
 assert result['qualified_task']
