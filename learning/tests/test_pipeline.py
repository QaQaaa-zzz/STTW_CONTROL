import pytest
from sttw_control.pipeline import evaluation_alphas, select_endpoint


def test_conditioned_evaluation_requires_explicit_unique_alphas():
    assert evaluation_alphas({}, {})==[None]
    assert evaluation_alphas({'priority':{}},{'priority_alphas':[0,.5,1]})==[0.,.5,1.]
    for panel in ({},{'priority_alphas':[]},{'priority_alphas':[0,0]},{'priority_alphas':[float('nan')]},{'priority_alphas':[-1]}):
        with pytest.raises(ValueError):evaluation_alphas({'priority':{}},panel)
    with pytest.raises(ValueError):evaluation_alphas({}, {'priority_alphas':[.5]})


def test_conditioned_endpoint_is_fixed_even_when_an_earlier_checkpoint_is_best():
    r={'best_checkpoint':'earlier','last_checkpoint':'final'}
    assert select_endpoint(r,conditioned=True)=='final'
    assert select_endpoint(r,conditioned=False)=='earlier'
    assert select_endpoint({**r,'best_checkpoint':None},conditioned=False)=='final'


def test_pipeline_evaluates_all_alphas_at_same_final_checkpoint(tmp_path,monkeypatch):
    import json
    from pathlib import Path
    import sttw_control.pipeline as pipeline
    import sttw_control.disturbance as disturbance
    task=tmp_path/'task.json';task.write_text(Path('learning/configs/priority_conditioned_learning.json').read_text())
    config=tmp_path/'training.json';config.write_text(json.dumps({'num_envs':1,'rollout_steps':1,'updates':1,'minibatch_size':1,'seed':1,'validation_seeds':[2]}))
    panel=tmp_path/'panel.json';panel.write_text(json.dumps({'priority_alphas':[0.,.5,1.],'evaluation_seeds':[3,4],'start_seconds':6.,'duration_seconds':1.}))
    calls=[]
    def invoke(args,**kwargs):
        get=lambda flag:args[args.index(flag)+1]
        out=Path(get('--output'));out.mkdir(parents=True)
        if args[1].endswith('/train.py'):
            (out/'status.json').write_text(json.dumps({'complete':True,'best_checkpoint':'earlier','last_checkpoint':'final'}))
        elif args[1].endswith('/disturbance.py'):
            calls.append((float(get('--priority-alpha')),int(get('--seed')),get('--checkpoint')))
            assert kwargs['env']['JAX_PLATFORMS']=='cpu'
            (out/'results.jsonl').write_text(json.dumps({'scenario':'nominal','policy':'baseline'})+'\n')
        else:(out/'INDEX.md').write_text('media')
    monkeypatch.setattr(pipeline.subprocess,'run',invoke)
    monkeypatch.setattr(disturbance,'plot_panel',lambda _:None)
    out=tmp_path/'run';pipeline.run(task,config,panel,out)
    assert calls==[(a,s,'final') for a in (0.,.5,1.) for s in (3,4)]
    assert json.loads((out/'pipeline_status.json').read_text())['phase']=='complete'
    assert len(json.loads((out/'standard_results.json').read_text()))==6
    assert 'alpha=1.0' in (out/'complete_media/INDEX.md').read_text()


def test_resume_skips_completed_training_and_evaluations(tmp_path,monkeypatch):
    import json
    from pathlib import Path
    import sttw_control.pipeline as pipeline
    import sttw_control.disturbance as disturbance
    task=tmp_path/'task.json';task.write_text(Path('learning/configs/priority_conditioned_learning.json').read_text())
    train=tmp_path/'train.json';train.write_text(json.dumps({'num_envs':1,'rollout_steps':1,'updates':1,'minibatch_size':1,'seed':1,'validation_seeds':[2]}))
    panel=tmp_path/'panel.json';panel.write_text(json.dumps({'priority_alphas':[0.,1.],'evaluation_seeds':[3],'start_seconds':6.,'duration_seconds':1.}))
    import hashlib
    checkpoint=tmp_path/'checkpoint';checkpoint.mkdir();(checkpoint/'identity.json').write_text('{}')
    sidecar=hashlib.sha256((checkpoint/'identity.json').read_bytes()).hexdigest()
    calls={'train':0,'eval':0,'media':0}
    def invoke(args,**kwargs):
        get=lambda key:args[args.index(key)+1]
        out=Path(get('--output'))
        if args[1].endswith('/train.py'):
            calls['train']+=1;out.mkdir()
            (out/'status.json').write_text(json.dumps({'complete':True,'control_transitions':1,'last_checkpoint':str(checkpoint),'best_checkpoint':None}))
        elif args[1].endswith('/disturbance.py'):
            calls['eval']+=1;out.mkdir(parents=True)
            (out/'status.json').write_text(json.dumps({'complete':True,'episodes':2}))
            (out/'declaration.json').write_text(json.dumps({'priority_alpha_override':float(get('--priority-alpha')),'panel':{'seed':3},'checkpoint':str(checkpoint),'policy':{'checkpoint_sidecar_sha256':sidecar},'scenarios':{'nominal':{}}}))
            (out/'results.jsonl').write_text('\n'.join(json.dumps({'scenario':'nominal','policy':p}) for p in ('baseline','residual')))
        else:
            calls['media']+=1
            if calls['media']==1:raise RuntimeError('render failure')
            out.mkdir(parents=True);(out/'INDEX.md').write_text('media')
    monkeypatch.setattr(pipeline.subprocess,'run',invoke);monkeypatch.setattr(disturbance,'plot_panel',lambda _:None)
    output=tmp_path/'run'
    with pytest.raises(RuntimeError,match='render failure'):pipeline.run(task,train,panel,output)
    pipeline.run(task,train,panel,output,resume=True)
    assert calls=={'train':1,'eval':2,'media':3}
    assert json.loads((output/'pipeline_status.json').read_text())['phase']=='complete'
    record=json.loads((output/'resumptions/0001.json').read_text())
    assert record['previous_status']['phase']=='error'
    (checkpoint/'identity.json').write_text('{"replaced":true}')
    with pytest.raises(ValueError,match='checkpoint hash mismatch'):
        pipeline.run(task,train,panel,output,resume=True)
    assert calls['train']==1 and calls['eval']==2
