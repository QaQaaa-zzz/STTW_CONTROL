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


@pytest.mark.parametrize('mutation',['alpha','clock','duration','budget','lane','route','block'])
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
            force_N=0.,route=g['route'],path_key=g['path_key'],speed=2.8 if 'S_' in g['route'] else 2.6,bank_index=6)
    assert len(five_alpha_label_inventory(p,cases))==280
    cases[2]['mode']='zero'
    with pytest.raises(ValueError,match='source'):five_alpha_label_inventory(p,cases)


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


def test_declarative_task_times_use_cumulative_saved_chunk_boundaries():
    from sttw_control.native_goal_run import snapshot_block_for_task
    p=protocol()
    for g in p['source_groups']:
        t=2. if g['source_index']<7 else 4.;b=19 if t==2. else 39
        g.update(source_task_s=t,source_block=b,expected_task_tick=round(t*200),expected_physical_tick=700+round(t*200),complete_carry=f'evaluation/complete_carry_{b:02d}.pkl',executed_paths=f'evaluation/executed_paths_{b:02d}.pkl',host_state=f'evaluation/host_state_{b:02d}.npz',expected_replanned=False,expected_ever_arrived=t==4. and g['source_lane'] in [50,98])
    p['maximum_suffix_lower_ticks']=392000
    rows=five_alpha_label_inventory(p)
    assert len(rows)==280 and sum((10-r['source_task_s'])*200 for r in rows)==392000
    assert len({r['source_group_identity'] for r in rows})==14
    old=five_alpha_label_inventory(protocol())
    assert not ({r['source_group_identity'] for r in old}&{r['source_group_identity'] for r in rows})
    assert snapshot_block_for_task({'chunk_upper_steps':[10,5,10]},.3)==1
    with pytest.raises(ValueError,match='saved chunk'):snapshot_block_for_task({'chunk_upper_steps':[10,5,10]},.2+.02)
    p['source_groups'][0]['source_block']=9
    with pytest.raises(ValueError,match='metadata'):five_alpha_label_inventory(p)


def test_authenticate_declared_host_and_physical_origin(tmp_path):
    from sttw_control.native_goal_run import validate_five_alpha_source_hosts
    (tmp_path/'evaluation').mkdir()
    (tmp_path/'evaluation/manifest.json').write_text(json.dumps(dict(physical_start_s=3.5)))
    group=dict(protocol()['source_groups'][0],source_lane=0)
    np.savez(tmp_path/group['host_state'],replanned=[False],ever_arrived=[False])
    validate_five_alpha_source_hosts(tmp_path,[group])
    with pytest.raises(ValueError,match='host declaration'):
        validate_five_alpha_source_hosts(tmp_path,[dict(group,expected_ever_arrived=True)])
    with pytest.raises(ValueError,match='physical origin'):
        validate_five_alpha_source_hosts(tmp_path,[dict(group,expected_physical_tick=901)])


def test_bounded_collection_preserves_h10_and_exact_primary_windows():
    from sttw_control.native_goal_run import label_collection_steps
    p=json.loads((Path(__file__).parents[1]/'configs/h75_friction08_tolerance_labels.json').read_text())
    rows=five_alpha_label_inventory(p)
    assert len(rows)==140 and p['horizon_s']==10.
    assert label_collection_steps(p,2.)==[5]*37+[3]
    assert sum(label_collection_steps(p,2.))*4*len(rows)==105280
    np.testing.assert_allclose(sorted({r['primary_window_end_s'] for r in rows}),[5.35,5.75])
    assert all(not r['expected_ever_arrived'] for r in rows)
    for key,value in [('collection_end_s',5.74),('collection_end_s',5.761),('require_ongoing_sources','yes')]:
        changed=copy.deepcopy(p);changed[key]=value
        with pytest.raises(ValueError):five_alpha_label_inventory(changed)
    p['source_groups'][0]['expected_ever_arrived']=True
    with pytest.raises(ValueError,match='ongoing'):five_alpha_label_inventory(p)


def test_relative_collection_and_explicit_route_subsets():
    from sttw_control.native_goal_run import label_collection_steps
    totals=[]
    for part,count,train,dev in [('nonS',200,120,80),('extendedS',80,40,40)]:
        p=json.loads((Path(__file__).parents[1]/f'configs/h75_friction08_wide_{part}_labels.json').read_text())
        rows=five_alpha_label_inventory(p)
        assert len(rows)==count
        assert {r['route'] for r in rows}==set(p['source_routes'])
        assert p['maximum_suffix_lower_ticks']==count*752
        assert (p['fitting_split']['expected_training_rows'],p['fitting_split']['expected_development_rows'])==(train,dev)
        for t in (1.,3.):assert label_collection_steps(p,t)==[5]*37+[3]
        bad=copy.deepcopy(p);bad['source_routes'].append('undeclared')
        with pytest.raises(ValueError,match='source routes'):five_alpha_label_inventory(bad)
        bad=copy.deepcopy(p);bad['source_routes'].pop()
        with pytest.raises(ValueError,match='source lanes'):five_alpha_label_inventory(bad)
        bad=copy.deepcopy(p);bad['collection_end_s']=5.76
        with pytest.raises(ValueError,match='relative or absolute'):five_alpha_label_inventory(bad)
        totals.append(p['maximum_suffix_lower_ticks'])
    assert sum(totals)==210560


def test_new_base_inventory_uses_source_identity_and_exact_budget():
    p=json.loads((Path(__file__).parents[1]/'configs/u170_lower200_tolerance_labels.json').read_text())
    rows=five_alpha_label_inventory(p)
    assert len(rows)==160 and len({r['source_index'] for r in rows})==8
    assert p['maximum_suffix_lower_ticks']==120320
    bad=copy.deepcopy(p);bad['model_id']='STTW_MIDDLE_FULLTAIL_PPO75_U77'
    with pytest.raises(ValueError,match='model'):five_alpha_label_inventory(bad)
