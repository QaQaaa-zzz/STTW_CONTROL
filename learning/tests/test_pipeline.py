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
