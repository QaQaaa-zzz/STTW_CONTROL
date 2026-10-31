"""New-physics capability inventory, no preparation/evaluation integration."""
import json
from pathlib import Path
import pytest
from sttw_control.native_goal_run import fixed_h75_capability_inventory,planning_chunk_steps


def config():return json.loads((Path(__file__).parents[1]/'configs/h75_friction08_capability.json').read_text())


def test_lower_transfer_parent_chain_requires_real_artifacts(tmp_path):
    from sttw_control.native_goal_run import authenticate_lower_transfer_parent,digest
    base={'checkpoint_sha256':'root','normalizer':{'dim':210}}
    assert authenticate_lower_transfer_parent({'parent_checkpoint_sha256':'root'},base,{})=={}
    candidate={'normalizer':base['normalizer']}
    for key,hash_key in [('actor','actor_sha256'),('checkpoint','checkpoint_sha256'),('config','config_file_sha256')]:
        file=tmp_path/key;file.write_text(key);candidate[key]=str(file);candidate[hash_key]=digest(file)
    manifest=tmp_path/'transfer.json';manifest.write_text(json.dumps({'parent_checkpoint_sha256':'root','candidate':candidate}))
    transfer={'parent_manifest':str(manifest),'parent_manifest_sha256':digest(manifest)}
    record={'parent_checkpoint_sha256':candidate['checkpoint_sha256']}
    assert len(authenticate_lower_transfer_parent(record,base,transfer))==4
    with pytest.raises(ValueError,match='lineage'):authenticate_lower_transfer_parent({'parent_checkpoint_sha256':'different'},base,transfer)
    (tmp_path/'actor').write_text('changed')
    with pytest.raises(ValueError,match='artifact'):authenticate_lower_transfer_parent(record,base,transfer)


def test_frozen_only_h75_thirtyfive_nominal_and_original_routes():
    p=config();rows=fixed_h75_capability_inventory(p)
    assert len(rows)==len({r['case_id'] for r in rows})==35
    assert {r['mode'] for r in rows}=={'H75'}
    assert {r['alpha'] for r in rows}=={0.,.25,.5,.75,1.}
    assert all(r['alpha']==r['query_alpha']==r['user_alpha'] and r['force_N']==0 for r in rows)
    assert sum(planning_chunk_steps(10.,10.))*4*len(rows)==70000
    reference=json.loads((Path(__file__).parents[1]/'configs/middle_selection.json').read_text())
    assert p['route_contexts']==reference['route_contexts']
    assert all(len([r for r in rows if r['route']==route])==5 for route in {r['route'] for r in rows})


@pytest.mark.parametrize('key,value',[('model_id','other'),('alphas',[0,.5,1]),('expected_cases',34),('maximum_lower_ticks',70001),('forces_N',[-64,64]),('planning_frequency_hz',1),('bank_indices',[3])])
def test_changed_capability_contract_rejected(key,value):
    p=config();p[key]=value
    with pytest.raises(ValueError):fixed_h75_capability_inventory(p)


def test_s_goal_extension_changes_only_goal_on_authenticated_saved_exit():
    import copy
    import numpy as np
    from sttw_control.native_goal_run import validate_goal_extension_source
    p=json.loads((Path(__file__).parents[1]/'configs/h75_friction08_s_goal_extension.json').read_text())
    rows=fixed_h75_capability_inventory(p)
    assert len(rows)==10 and p['maximum_lower_ticks']==2000*len(rows)
    old={c['route']:c for c in config()['route_contexts']}
    for c in p['route_contexts']:
        assert {k:v for k,v in c.items() if k!='goal_distance_m'}==old[c['route']]
        r=p['goal_source']['paths'][c['route']]
        assert r['new_goal_s']-r['old_goal_s']==3.
        with np.load(r['file']) as z:
            assert np.all(z['curvature'][(z['s']>=r['old_goal_s'])&(z['s']<=r['new_goal_s'])]==0)
    source=json.loads((Path(p['goal_source']['run'])/'plan.json').read_text())['initial_source']
    validate_goal_extension_source(p,source)
    bad=copy.deepcopy(p);bad['goal_source']['paths'][p['route_contexts'][0]['route']]['new_goal_s']+=1.
    with pytest.raises(ValueError):validate_goal_extension_source(bad,source)


def test_low_speed_tight_turn_has_four_routes_five_alphas_frozen_budget():
    p=json.loads((Path(__file__).parents[1]/'configs/h75_friction08_low_speed_tight_turn.json').read_text())
    rows=fixed_h75_capability_inventory(p)
    assert len(rows)==len({r['case_id'] for r in rows})==20
    assert {r['alpha'] for r in rows}=={0.,.25,.5,.75,1.}
    assert {(c['speed'],c['radius_m']) for c in p['route_contexts']}=={(1.6,.9),(1.6,1.3),(2.,.9),(2.,1.3)}
    assert all(c['family']==1 and c['direction']==1. and c['angle_deg']==90. and c['goal_offset_after_turn_m']==4. for c in p['route_contexts'])
    assert sum(planning_chunk_steps(10.,10.))*4*len(rows)==40000
    assert all(r['force_N']==0. and r['bank_index']==6 for r in rows)
    assert not any('goal_extension_m' in r for r in rows)
    p['route_contexts'][0]['speed']=1.2
    with pytest.raises(ValueError):fixed_h75_capability_inventory(p)


def test_explicit_candidate_inventory_preserves_low_speed_contract():
    p=json.loads((Path(__file__).parents[1]/'configs/h75_friction08_low_speed_tight_turn.json').read_text())
    p['candidate_model']={'registry_id':'candidate_U90'}
    rows=fixed_h75_capability_inventory(p)
    assert len(rows)==20 and {row['model_id'] for row in rows}=={'candidate_U90'}
    assert {row['mode'] for row in rows}=={'H75'}  # Existing runtime dispatch key only.
    bad=config();bad['candidate_model']=p['candidate_model']
    with pytest.raises(ValueError,match='only for declared low-speed'):fixed_h75_capability_inventory(bad)


@pytest.fixture
def capability_candidate(tmp_path):
    import pickle,hashlib
    import numpy as np
    import torch
    from sttw_control.native_goal_run import digest
    cfg={key:{} for key in ('network','timing','residual_interface','path_reward','follower')}
    cfg['network']={'architecture':'alpha_modulated_v1','actor_dim':351}
    cfg['finetuning']={'parent':{'checkpoint_sha256':'frozen_checkpoint'}}
    source={'candidate':{'actor_sha256':'lower'},'model':{'friction':.8}}
    identity={'config_sha256':hashlib.sha256(json.dumps(cfg,sort_keys=True,separators=(',',':')).encode()).hexdigest(),'parent_checkpoint_sha256':'frozen_checkpoint','lower_actor_sha256':'lower'}
    weights={};values={};widths=[351,128,128,64,2]
    for i,index in enumerate((0,2,4,6)):
        kernel=np.zeros((widths[i],widths[i+1]),np.float32);bias=np.zeros(widths[i+1],np.float32)
        weights[f'actor.{index}.weight']=torch.from_numpy(kernel.T.copy());weights[f'actor.{index}.bias']=torch.from_numpy(bias)
        values[f'Dense_{i}']={'kernel':kernel,'bias':bias}
    for key,width in (('gain_2',128),('shift_2',128),('gain_3',64),('shift_3',64)):
        values[key]=np.zeros(width,np.float32);weights['actor.'+key]=torch.from_numpy(values[key])
    actor=tmp_path/'actor.pkl';actor.write_bytes(pickle.dumps({'params':values}))
    checkpoint=tmp_path/'checkpoint.pt';torch.save(dict(update=90,identity=identity,policy=weights),checkpoint)
    manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps(dict(config=cfg,source=source,identity=identity)))
    candidate=dict(registry_id='candidate_U90',actor=str(actor),actor_sha256=digest(actor),checkpoint=str(checkpoint),checkpoint_sha256=digest(checkpoint),training_manifest=str(manifest),training_manifest_sha256=digest(manifest),source_update=90)
    return candidate,{'checkpoint_sha256':'frozen_checkpoint','actor':str(actor)},source,cfg


def test_capability_candidate_exact_export_and_provenance(capability_candidate):
    from sttw_control.native_goal_run import authenticate_capability_candidate
    candidate,frozen,source,cfg=capability_candidate
    receipt=authenticate_capability_candidate(candidate,frozen,source,cfg)
    assert receipt['registry_id']=='candidate_U90'
    assert receipt['role']=='explicit_training_candidate_not_frozen_H75'
    candidate=dict(candidate,source_update=91)
    with pytest.raises(ValueError,match='update differs'):authenticate_capability_candidate(candidate,frozen,source,cfg)


def test_capability_candidate_rejects_rehashed_actor_mismatch(capability_candidate):
    import pickle
    from sttw_control.native_goal_run import authenticate_capability_candidate,digest
    candidate,frozen,source,cfg=capability_candidate
    file=Path(candidate['actor']);params=pickle.loads(file.read_bytes());params['params']['gain_2'][0]=1
    file.write_bytes(pickle.dumps(params));candidate=dict(candidate,actor_sha256=digest(file))
    with pytest.raises(AssertionError):authenticate_capability_candidate(candidate,frozen,source,cfg)
