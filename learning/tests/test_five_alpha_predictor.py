"""No fitting or dynamics; frozen schema and authenticated reader checks."""
import copy
import json
from pathlib import Path
import numpy as np
import pytest
from sttw_control import tolerance_predictor_pilot as m
ROOT=Path(__file__).resolve().parents[2]


def test_fixed_budget_config():
    cfg=json.loads((ROOT/'learning/configs/h75_tolerance_predictor_five_alpha.json').read_text())
    m.validate_five_alpha_fit_config(cfg)
    for field,value in [('updates_per_model_seed',1001),('fixed_last',False),('learning_rate',.01)]:
        other=copy.deepcopy(cfg);other['training'][field]=value
        with pytest.raises(ValueError):m.validate_five_alpha_fit_config(other)


def test_grouped_metrics_keep_singleclass_uncertainty():
    meta=[dict(source_group_id='source7',preparation_s=.2,force_N=-64.,alpha=a,lifecycle_role='post_arrival_survival') for a in [0.,.25,.5,.75,1.]]
    report=m.five_alpha_grouped_metrics(np.zeros(5),np.full(5,.3),meta)
    assert report['overall']['roc_auc'] is None
    assert len(report['groups']['alpha'])==5
    assert report['groups']['lifecycle_role']['post_arrival_survival']['n']==5
    assert len(m.discordant_pairs(np.zeros(5),meta))==0


@pytest.fixture
def dataset(tmp_path):
    protocol=json.loads((ROOT/'learning/configs/h75_five_alpha_tolerance_labels.json').read_text())
    cfg=json.loads((ROOT/'learning/configs/h75_tolerance_predictor_five_alpha.json').read_text())
    def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v))
    model={'registry_id':'STTW_MIDDLE_FULLTAIL_PPO75_U77'}
    for key in ('actor','checkpoint'):
        f=tmp_path/key;f.write_text(key);model[key]=str(f);model[key+'_sha256']=m.digest(f)
    live=tmp_path/'live_module.py';live.write_text('frozen implementation')
    common=dict(replanning=dict(continuous_replanning=True,period_s=.1),mode='five_alpha_tolerance_labels',protocol=protocol,models={'H75':model},config={'residual_interface':dict(speed_positive_scale_m_s=1.,speed_negative_scale_m_s=1.,steer_scale_rad=.2)},input_sha256={},source_sha256={str(live):m.digest(live)})
    all_rows=[];all_cases=[];features=[];roles=[];children=[]
    for t in [1.,6.]:
        child=tmp_path/f'task{t:g}';(child/'source').mkdir(parents=True);(child/'source/live_module.py').write_bytes(live.read_bytes());children.append(str(child));rows=[];cases=[];states=[];source_records=[]
        for g in [g for g in protocol['source_groups'] if g['source_task_s']==t]:
            start=len(cases)
            for a in protocol['candidate_alphas']:
                for prep in [.2,.6]:
                    for force in [-64.,64.]:
                        cid=f"source{g['source_index']}/a{a}/p{prep}/f{force}"
                        c=dict(g,case_id=cid,alpha=.5,user_alpha=.5,query_alpha=a,candidate_alpha=a,preparation_s=prep,force_N=force,duration_s=.15,onset_s=t+prep,primary_window_start_s=t,primary_window_end_s=t+prep+.15+3)
                        source=np.zeros(351,np.float32);source[336]=.5;query=source.copy();query[336]=a;latent=np.zeros(2,np.float32)
                        raw=dict(time=np.array([t]),active_tick=np.ones(1,bool),reference_target_permitted=np.ones(1,bool),query_observation=query[None],composed_latent=latent[None],pre_upper_yaw=np.zeros(1))
                        for k in ['policy_fault','lower_fault','nonfinite','continuation_geometry_fault','physical_failure']:raw[k]=np.zeros(1,bool)
                        raw['contact_failure']=np.ones(1,bool)
                        f=child/'evaluation'/f'raw{len(cases)}.npz';f.parent.mkdir(parents=True,exist_ok=True);np.savez(f,**raw)
                        role='training' if c['route'] in protocol['fitting_split']['training_routes'] else 'development'
                        r=dict(c,raw_file=str(f),raw_sha256=m.digest(f),fitting_role=role,primary=dict(label_status='fall',no_fall_through_window=False))
                        x,_=m.feature_row(source,query,latent,0.,c,common['config']['residual_interface'],cfg,expected_source_alpha=.5)
                        cases.append(c);rows.append(r);states.append(source);features.append(x);roles.append(role)
            source_records.append(dict(pre_boundary_fingerprint=m.hashlib.sha256(f'{t}/{g["source_index"]}'.encode()).hexdigest(),case_indices=list(range(start,len(cases))),all_twenty_siblings_exact=True))
        save(child/'inventory.json',cases);save(child/'analysis/results.json',rows)
        np.savez(child/'evaluation/source_features.npz',native_observation=np.array(states),source_world_yaw=np.zeros(140))
        save(child/'evaluation/source_records.json',source_records)
        save(child/'plan.json',common);sha=m.digest(child/'plan.json');save(child/'seal.json',dict(plan_sha256=sha));save(child/'analysis/receipt.json',dict(plan_sha256=sha))
        all_rows+=rows;all_cases+=cases
    (tmp_path/'source').mkdir();(tmp_path/'source/live_module.py').write_bytes(live.read_bytes())
    parent=dict(common,label_parent=True,children=children);save(tmp_path/'plan.json',parent);sha=m.digest(tmp_path/'plan.json');save(tmp_path/'seal.json',dict(plan_sha256=sha));save(tmp_path/'analysis/receipt.json',dict(plan_sha256=sha));save(tmp_path/'inventory.json',all_cases);save(tmp_path/'analysis/results.json',all_rows)
    np.savez(tmp_path/'analysis/features_labels.npz',features=np.array(features),labels=np.zeros(280),case_ids=np.array([r['case_id'] for r in all_rows]),fitting_role=np.array(roles),valid=np.ones(280,bool))
    return tmp_path/'analysis',cfg


def test_authenticated_reader_and_raw_tamper(dataset):
    analysis,cfg=dataset;data=m.load_five_alpha_dataset(analysis,cfg)
    assert data['training']['full'].shape==(160,41)
    assert data['development']['full'].shape==(120,41)
    assert data['receipt']['source_groups']==14
    row=json.loads((analysis/'results.json').read_text())[0]
    Path(row['raw_file']).write_bytes(b'changed')
    with pytest.raises(ValueError,match='raw changed'):m.load_five_alpha_dataset(analysis,cfg)


def test_feature_cache_tamper_rejected(dataset):
    analysis,cfg=dataset;file=analysis/'features_labels.npz'
    with np.load(file) as z:cache=dict(z)
    cache['features'][0,0]=1.;np.savez(file,**cache)
    with pytest.raises(AssertionError):m.load_five_alpha_dataset(analysis,cfg)


def test_archived_source_survives_authorized_live_change(dataset):
    analysis,cfg=dataset
    (analysis.parent/'live_module.py').write_text('authorized new implementation')
    assert len(m.load_five_alpha_dataset(analysis,cfg)['training']['y'])==160
    (analysis.parent/'source/live_module.py').write_text('corrupt archive')
    with pytest.raises(ValueError,match='archived source changed'):m.load_five_alpha_dataset(analysis,cfg)


def test_only_ten_hz_data(dataset):
    analysis,cfg=dataset;root=analysis.parent
    p=json.loads((root/'plan.json').read_text());p['protocol']['planning_frequency_hz']=1.
    (root/'plan.json').write_text(json.dumps(p));sha=m.digest(root/'plan.json')
    (root/'seal.json').write_text(json.dumps(dict(plan_sha256=sha)));(analysis/'receipt.json').write_text(json.dumps(dict(plan_sha256=sha)))
    with pytest.raises(ValueError,match='only native10Hz'):m.load_five_alpha_dataset(analysis,cfg)


def make_second_batch(analysis):
    """Synthetic authenticated2/4 fixture, retaining the exact same H75 files."""
    import shutil
    original=analysis.parent;root=original.parent/(original.name+'_second');shutil.copytree(original,root)
    parent=json.loads((root/'plan.json').read_text());parent['children']=[]
    for old_child in json.loads((original/'plan.json').read_text())['children']:
        child=root/Path(old_child).name;parent['children'].append(str(child))
        cp=json.loads((child/'plan.json').read_text())
        for g in cp['protocol']['source_groups']:g['source_task_s']={1.:2.,6.:4.}[g['source_task_s']]
        records=json.loads((child/'analysis/results.json').read_text());cases=json.loads((child/'inventory.json').read_text())
        for record,c in zip(records,cases):
            t={1.:2.,6.:4.}[c['source_task_s']]
            for obj in (record,c):
                obj.update(source_task_s=t,primary_window_start_s=t,onset_s=t+c['preparation_s'],primary_window_end_s=t+c['preparation_s']+.15+3.)
            raw=child/'evaluation'/Path(record['raw_file']).name
            with np.load(raw) as z:data=dict(z)
            data['time']=np.array([t]);np.savez(raw,**data);record['raw_file']=str(raw);record['raw_sha256']=m.digest(raw)
        src=child/'evaluation/source_records.json';sr=json.loads(src.read_text())
        for i,r in enumerate(sr):r['pre_boundary_fingerprint']=m.hashlib.sha256(f'second/{child.name}/{i}'.encode()).hexdigest()
        src.write_text(json.dumps(sr))
        (child/'inventory.json').write_text(json.dumps(cases));(child/'analysis/results.json').write_text(json.dumps(records));(child/'plan.json').write_text(json.dumps(cp));sha=m.digest(child/'plan.json')
        (child/'seal.json').write_text(json.dumps(dict(plan_sha256=sha)));(child/'analysis/receipt.json').write_text(json.dumps(dict(plan_sha256=sha)))
    for g in parent['protocol']['source_groups']:g['source_task_s']={1.:2.,6.:4.}[g['source_task_s']]
    records=[];cases=[]
    for child in parent['children']:
        records+=json.loads((Path(child)/'analysis/results.json').read_text());cases+=json.loads((Path(child)/'inventory.json').read_text())
    (root/'analysis/results.json').write_text(json.dumps(records));(root/'inventory.json').write_text(json.dumps(cases));(root/'plan.json').write_text(json.dumps(parent));sha=m.digest(root/'plan.json')
    (root/'seal.json').write_text(json.dumps(dict(plan_sha256=sha)));(root/'analysis/receipt.json').write_text(json.dumps(dict(plan_sha256=sha)))
    return root/'analysis'


def test_multi_analysis_combines_routes_and_namespaces(dataset):
    analysis,cfg=dataset;second=make_second_batch(analysis)
    cfg=copy.deepcopy(cfg);cfg['dataset'].pop('analysis');cfg['dataset'].update(analyses=[str(analysis),str(second)],expected_cases=560,expected_training_rows=320,expected_development_rows=240)
    m.validate_five_alpha_fit_config(cfg);data=m.load_five_alpha_dataset(None,cfg)
    assert data['training']['full'].shape==(320,41)
    assert data['development']['full'].shape==(240,41)
    train={r['source_group_id'] for r in data['training']['meta']};dev={r['source_group_id'] for r in data['development']['meta']}
    assert len(train)==16 and len(dev)==12 and not train&dev
    assert data['receipt']['source_groups']==28
    assert len({r['case_id'] for role in ['training','development'] for r in data[role]['meta']})==560


def test_multi_analysis_rejects_same_source_semantics(dataset,tmp_path):
    import shutil
    analysis,cfg=dataset;alias=tmp_path/'copy';shutil.copytree(analysis.parent,alias)
    # Different directory is insufficient: copied metadata still describes the same source task states.
    cfg=copy.deepcopy(cfg);cfg['dataset'].pop('analysis');cfg['dataset'].update(analyses=[str(analysis),str(alias/'analysis')],expected_cases=560,expected_training_rows=320,expected_development_rows=240)
    with pytest.raises(ValueError,match='duplicate semantic source'):m.load_five_alpha_dataset(None,cfg)


def test_short_fit_budget_matches_three_fixed_seeds():
    cfg=json.loads((ROOT/'learning/configs/h75_tolerance_predictor_short_fit.json').read_text())
    m.validate_five_alpha_fit_config(cfg)
    assert cfg['training']['updates_per_model_seed']==200
    for steps in (0,True,200.5,1001):
        bad=copy.deepcopy(cfg);bad['training']['updates_per_model_seed']=steps
        with pytest.raises(ValueError):m.validate_five_alpha_fit_config(bad)
    bad=copy.deepcopy(cfg);bad['budget']['maximum_optimizer_updates']=3000
    with pytest.raises(ValueError):m.validate_five_alpha_fit_config(bad)


def make_training_only_batch(analysis,second):
    import shutil
    root=analysis.parent.parent/'bank4';root.mkdir();base=json.loads((analysis.parent/'plan.json').read_text());protocol=base['protocol'];groups=[];rows=[];cases=[];children=[];features=[]
    for batch_index,src in enumerate([analysis,second]):
        pp=json.loads((src.parent/'plan.json').read_text())
        for g in pp['protocol']['source_groups']:
            g['source_index']+=batch_index*14;g['bank_index']=4;g['source_case_id']=g['source_case_id'].replace('bank6/','bank4/');groups.append(g)
        for old in pp['children']:
            child=root/(f'b{batch_index}_'+Path(old).name);shutil.copytree(old,child);children.append(str(child));rr=json.loads((child/'analysis/results.json').read_text());cc=json.loads((child/'inventory.json').read_text())
            for r,c in zip(rr,cc):
                for obj in [r,c]:obj['source_index']+=batch_index*14;obj['bank_index']=4;obj['case_id']='bank4/'+str(obj['source_index'])+'/'+obj['case_id'];obj['source_case_id']=obj['source_case_id'].replace('bank6/','bank4/')
                r['fitting_role']='training'
                f=child/'evaluation'/Path(r['raw_file']).name;r['raw_file']=str(f);r['raw_sha256']=m.digest(f)
            (child/'analysis/results.json').write_text(json.dumps(rr));(child/'inventory.json').write_text(json.dumps(cc));rows+=rr;cases+=cc
            sr=json.loads((child/'evaluation/source_records.json').read_text())
            for i,r in enumerate(sr):r['pre_boundary_fingerprint']=m.hashlib.sha256(f'bank4/{child.name}/{i}'.encode()).hexdigest()
            (child/'evaluation/source_records.json').write_text(json.dumps(sr))
        with np.load(src/'features_labels.npz') as z:features.append(z['features'])
    protocol.update(source_groups=groups,data_partition='training_only',expected_cases=560,source_plan_sha256='bank4source',expected_source_states=28)
    protocol['fitting_split'].update(training_routes=sorted({g['route'] for g in groups}),development_routes=[],expected_training_rows=560,expected_development_rows=0)
    for child in map(Path,children):
        cp=json.loads((child/'plan.json').read_text());cp['protocol']=protocol;(child/'plan.json').write_text(json.dumps(cp));sha=m.digest(child/'plan.json');(child/'seal.json').write_text(json.dumps(dict(plan_sha256=sha)));(child/'analysis/receipt.json').write_text(json.dumps(dict(plan_sha256=sha)))
    (root/'source').mkdir();shutil.copy(analysis.parent/'source/live_module.py',root/'source/live_module.py');base.update(protocol=protocol,children=children);(root/'plan.json').write_text(json.dumps(base));sha=m.digest(root/'plan.json');(root/'seal.json').write_text(json.dumps(dict(plan_sha256=sha)));(root/'inventory.json').write_text(json.dumps(cases));(root/'analysis').mkdir();(root/'analysis/receipt.json').write_text(json.dumps(dict(plan_sha256=sha)));(root/'analysis/results.json').write_text(json.dumps(rows))
    np.savez(root/'analysis/features_labels.npz',features=np.concatenate(features),labels=np.zeros(560),valid=np.ones(560,bool),case_ids=np.array([r['case_id'] for r in rows]),fitting_role=np.full(560,'training'))
    return root/'analysis'


def test_training_only_combines_without_changing_frozen_dev(dataset):
    analysis,cfg=dataset;second=make_second_batch(analysis);training=make_training_only_batch(analysis,second)
    cfg=copy.deepcopy(cfg);cfg['dataset'].pop('analysis');cfg['dataset'].update(analyses=[str(analysis),str(second)],expected_cases=560,expected_training_rows=320,expected_development_rows=240)
    old=m.load_five_alpha_dataset(None,cfg);freeze=analysis.parent/'freeze.json'
    payload=dict(schema='sttw_development_freeze_v1',rows=[dict(case_id=r['case_id'],features=r['features'],label=r['label'],meta={}) for r in old['receipt']['development_identity_rows']],inputs=old['receipt']['inputs']);freeze.write_text(json.dumps(payload))
    cfg['dataset'].update(analyses=[str(analysis),str(second),str(training)],expected_cases=1120,expected_training_rows=880,expected_development_rows=240,development_freeze=dict(file=str(freeze),sha256=m.digest(freeze)))
    m.validate_five_alpha_fit_config(cfg);data=m.load_five_alpha_dataset(None,cfg)
    assert data['training']['full'].shape==(880,41)
    assert data['development']['full'].shape==(240,41)
    assert data['receipt']['development_freeze_verified']
    np.testing.assert_array_equal(data['development']['full'],old['development']['full'])
    assert data['receipt']['source_groups']==56
    bad=copy.deepcopy(data);bad['receipt']['development_identity_rows'][0]['label']=1
    with pytest.raises(ValueError,match='frozen development'):m.verify_five_alpha_development_freeze(bad,cfg)


def test_physical_source_collision_rejected_across_different_semantics(dataset):
    analysis,cfg=dataset;second=make_second_batch(analysis)
    first_plan=json.loads((analysis.parent/'plan.json').read_text());second_plan=json.loads((second.parent/'plan.json').read_text())
    a=Path(first_plan['children'][0])/'evaluation/source_records.json';b=Path(second_plan['children'][0])/'evaluation/source_records.json'
    aa=json.loads(a.read_text());bb=json.loads(b.read_text());bb[0]['pre_boundary_fingerprint']=aa[0]['pre_boundary_fingerprint'];b.write_text(json.dumps(bb))
    cfg=copy.deepcopy(cfg);cfg['dataset'].pop('analysis');cfg['dataset'].update(analyses=[str(analysis),str(second)],expected_cases=560,expected_training_rows=320,expected_development_rows=240)
    with pytest.raises(ValueError,match='physical complete-state source collision'):m.load_five_alpha_dataset(None,cfg)


def test_limited_collection_explicit_fit_contract():
    cfg=json.loads((ROOT/'learning/configs/h75_friction08_tolerance_fit.json').read_text())
    m.validate_five_alpha_fit_config(cfg)
    protocol=json.loads((ROOT/'learning/configs/h75_friction08_tolerance_labels.json').read_text())
    part=m.five_alpha_partition(protocol)
    assert (part['kind'],part['cases'],part['training_rows'],part['development_rows'])==('limited_collection',140,80,60)
    bad=copy.deepcopy(cfg);del bad['dataset']['physics_identity']
    with pytest.raises(ValueError,match='physics'):m.validate_five_alpha_fit_config(bad)
    protocol['source_groups'][0]['expected_ever_arrived']=True
    with pytest.raises(ValueError,match='ongoing'):m.five_alpha_partition(protocol)


@pytest.mark.parametrize('identity',['physics_identity','analysis_plan_sha256'])
def test_limited_collection_loader_rejects_wrong_identity(identity):
    cfg=json.loads((ROOT/'learning/configs/h75_friction08_tolerance_fit.json').read_text())
    cfg['dataset'][identity]={} if identity=='physics_identity' else '0'*64
    with pytest.raises(ValueError,match='physics/plan identity'):
        m.load_five_alpha_dataset(None,cfg)


def test_stage_subset_partition_and_explicit_two_plan_fit():
    cfg=json.loads((ROOT/'learning/configs/h75_friction08_stage_coverage_fit.json').read_text())
    m.validate_five_alpha_fit_config(cfg)
    sizes=[]
    for name in ('nonS','extendedS'):
        p=json.loads((ROOT/f'learning/configs/h75_friction08_wide_{name}_labels.json').read_text())
        part=m.five_alpha_partition(p);sizes.append((part['cases'],part['training_rows'],part['development_rows']))
        bad=copy.deepcopy(p);bad['source_groups'][0]['source_task_s']=bad['source_groups'][-1]['source_task_s']
        with pytest.raises(ValueError,match='route/time'):m.five_alpha_partition(bad)
    assert tuple(map(sum,zip(*sizes)))==(280,160,120)
    bad=copy.deepcopy(cfg);bad['dataset']['analysis_plan_sha256s'].pop(next(iter(bad['dataset']['analysis_plan_sha256s'])))
    with pytest.raises(ValueError,match='two explicit'):m.validate_five_alpha_fit_config(bad)


def test_stage_subset_cannot_mix_physics_or_historical_batches(monkeypatch):
    cfg=json.loads((ROOT/'learning/configs/h75_friction08_stage_coverage_fit.json').read_text())
    def batch(kind,physics,n,nt,nd):
        return dict(receipt=dict(data_partition=kind,physics_identity=physics,declared_training_rows=nt,declared_development_rows=nd,source_route_times=[(f'r{i}',t) for i in range(n) for t in (1.,3.)]))
    a=batch('limited_route_subset',{'xml':'new'},5,120,80);b=batch('mixed',{'xml':'new'},2,40,40)
    batches=iter([a,b]);monkeypatch.setattr(m,'_load_five_alpha_single',lambda *args:next(batches))
    with pytest.raises(ValueError,match='cannot mix historical'):m.load_five_alpha_dataset(None,cfg)
    b=batch('limited_route_subset',{'xml':'old'},2,40,40)
    # Distinct route names give complete semantic coverage, isolating physics check.
    b['receipt']['source_route_times']=[(f'r{i}',t) for i in (5,6) for t in (1.,3.)]
    cfg['dataset']['source_route_times']=a['receipt']['source_route_times']+b['receipt']['source_route_times']
    batches=iter([a,b])
    with pytest.raises(ValueError,match='mixed physics'):m.load_five_alpha_dataset(None,cfg)
