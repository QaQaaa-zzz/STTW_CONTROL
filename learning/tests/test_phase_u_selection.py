from sttw_control.phase_u_selection import select_adapted_skill


def row(update,nominal,success,time):
    return dict(update=update,checkpoint=str(update),nominal={'n':{'full_window':nominal,'strict_support':nominal}},
                hard={'h':dict(success=success,time_s=time)})


def test_rejects_task_gain_that_loses_original_nominal_condition():
    result=select_adapted_skill([row(0,True,False,None),row(5,False,True,10.)])
    assert result['selected_update']==0
    assert len(result['decisions'][1]['lost_nominal_conditions'])==2


def test_selects_earliest_best_qualified_time_not_last_update():
    result=select_adapted_skill([row(0,True,False,None),row(5,True,True,10.),row(10,True,True,9.),row(15,True,True,9.),row(20,True,True,11.)])
    assert result['selected_update']==10


def test_no_nominal_success_is_reported_as_uninformative():
    result=select_adapted_skill([row(0,False,False,None),row(5,False,True,10.)])
    assert result['selected_update']==5
    assert result['nominal_baseline_positive_conditions']==0


def test_shortlist_includes_explicit_post_update_checkpoints(tmp_path):
    import json,subprocess,sys
    run=tmp_path/'run';(run/'training').mkdir(parents=True)
    (run/'status.json').write_text(json.dumps(dict(stage='training_complete',completed_updates=150)))
    (run/'best_step_model.json').write_text(json.dumps(dict(update=75)))
    (run/'training_rewards.jsonl').write_text('\n'.join(json.dumps(dict(update=u,mean_step_reward=r)) for u,r in [(0,0.),(25,3.),(75,5.)]))
    for name in ('model_initial.pt','model_update_00025.pt','model_update_00075.pt','model_step_best.pt','model_150.pt'):(run/'training'/name).touch()
    out=tmp_path/'selection.json'
    subprocess.run([sys.executable,'learning/cli/select_phase_u_candidates.py','--run',str(run),'--output',str(out),'--top','2'],check=True,capture_output=True)
    result=json.loads(out.read_text())
    assert [r['policy_update'] for r in result['candidates']]==[75,25,150]
    stored=next(r for r in result['all_stored'] if r['checkpoint'].endswith('model_update_00075.pt'))
    assert stored['score']['mean_step_reward']==5.
