"""Bounded source-copy tests, without environment construction or physical steps."""
import copy
import json
import pickle
from pathlib import Path
from types import SimpleNamespace
import jax
import numpy as np
import pytest
from sttw_control.native_goal_run import (five_alpha_label_inventory,import_five_alpha_sources,
                                          force_pulses,planning_chunk_steps)


def protocol():
    return json.loads((Path(__file__).parents[1]/'configs/h75_five_alpha_tolerance_labels.json').read_text())


def test_inventory_two_clocks_five_candidates_exact_budget_and_pulse_origin():
    p=protocol();rows=five_alpha_label_inventory(p)
    assert len(rows)==len({r['case_id'] for r in rows})==280
    assert {r['alpha'] for r in rows}=={.5}
    assert {r['query_alpha'] for r in rows}=={0.,.25,.5,.75,1.}
    assert sum((10-r['source_task_s'])*200 for r in rows)==364000
    for t,blocks in [(1.,90),(6.,40)]:
        rr=five_alpha_label_inventory(p,task_s=t)
        assert len(rr)==140 and planning_chunk_steps(10.,10.-t)==[5]*blocks
        pulse=np.asarray(force_pulses(rr,3.5))
        np.testing.assert_allclose(pulse[:,0],3.5+t+np.asarray([r['preparation_s'] for r in rr]))
        np.testing.assert_allclose(pulse[:,1]-pulse[:,0],.15)
        assert all(r['primary_window_end_s']<=10 for r in rr)


@pytest.mark.parametrize('mutation',['alpha','clock','duration','budget','host','lane','route','block'])
def test_reject_silent_contract_or_source_changes(mutation):
    p=protocol()
    if mutation=='alpha':p['candidate_alphas']=[0.,.5,1.]
    elif mutation=='clock':p['planning_frequency_hz']=1.
    elif mutation=='duration':p['duration_s']=.1
    elif mutation=='budget':p['maximum_suffix_lower_ticks']+=1
    elif mutation=='host':p['source_groups'][12]['expected_replanned']=False
    elif mutation=='lane':p['source_groups'][0]['source_lane']=3
    elif mutation=='route':p['source_groups'][0]['route']='unrelated'
    else:p['source_groups'][7]['source_block']=5
    with pytest.raises(ValueError):five_alpha_label_inventory(p)


def test_source_selection_rejects_metadata_substitution():
    p=protocol();cases=[{} for _ in range(336)]
    for g in p['source_groups'][:7]:
        cases[g['source_lane']]=dict(case_id=g['source_case_id'],mode='H75',alpha=.5,
            force_N=0.,route=g['route'],speed=2.8 if 'S_' in g['route'] else 2.6,bank_index=6)
    assert len(five_alpha_label_inventory(p,cases))==280
    cases[2]['mode']='zero'
    with pytest.raises(ValueError,match='source lane'):five_alpha_label_inventory(p,cases)


# A registered CPU fixture with extra hidden fields demonstrates entire-tree
# import rather than reconstruction of visible posture alone.
@jax.tree_util.register_pytree_node_class
class Fixture:
    def __init__(self,state,hidden):self.state=state;self.hidden=hidden
    def tree_flatten(self):return (self.state,self.hidden),None
    @classmethod
    def tree_unflatten(cls,aux,children):return cls(*children)


@jax.tree_util.register_pytree_node_class
class Fields:
    def __init__(self,**fields):self.__dict__.update(fields)
    def tree_flatten(self):return tuple(self.__dict__.values()),tuple(self.__dict__)
    @classmethod
    def tree_unflatten(cls,keys,values):return cls(**dict(zip(keys,values)))


def test_import_preserves_hidden_carry_paths_host_and_correlated_copies(tmp_path):
    n=3;g=protocol()['source_groups'][12];g={**g,'source_lane':1}
    physical=Fields(physical_tick=np.full(n,1900,np.int32),failed=np.zeros(n,bool),eso=np.arange(12,dtype=np.int32).reshape(n,4))
    saved=Fixture(Fields(tick=np.full(n,1200,np.int32),alpha=np.full(n,.5,np.float32),fault=np.zeros(n,bool),physical=physical),np.arange(21,dtype=np.float32).reshape(n,7))
    path=np.arange(15).reshape(n,5)
    (tmp_path/'evaluation').mkdir()
    for key,value in [('complete_carry',saved),('executed_paths',path)]:
        with (tmp_path/g[key]).open('wb') as f:pickle.dump(value,f)
    np.savez(tmp_path/g['host_state'],replanned=[False,True,False],ever_arrived=[True,False,True],original_goals_xy=np.arange(6).reshape(n,2))
    carry,paths,host=import_five_alpha_sources(tmp_path,[g,g])
    np.testing.assert_array_equal(carry.hidden,np.stack([saved.hidden[1]]*2))
    np.testing.assert_array_equal(carry.state.physical.eso,np.stack([physical.eso[1]]*2))
    np.testing.assert_array_equal(paths,np.stack([path[1]]*2))
    assert host['replanned'].tolist()==[True,True] and host['ever_arrived'].tolist()==[False,False]
    with pytest.raises(ValueError,match='host flags'):
        import_five_alpha_sources(tmp_path,[dict(g,expected_ever_arrived=True)])
    with pytest.raises(ValueError,match='clock/user/failure'):
        import_five_alpha_sources(tmp_path,[dict(g,expected_task_tick=200)])
