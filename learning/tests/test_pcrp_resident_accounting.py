import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'cli'))
from summarize_pcrp_decision import e1_execution_timing, resident_probe_accounting


def save(path, data):
    path.parent.mkdir(exist_ok=True,parents=True)
    path.write_text(json.dumps(data))


def test_charge_only_worker_entered_slots_and_preserve_partial_failure(tmp_path):
    upgrade=tmp_path/'resident_upgrade'
    assert resident_probe_accounting(tmp_path)==[]
    for name in ['probe_a','probe_b']:
        folder=upgrade/name
        save(upgrade/(name+'_launch.json'),dict(output=str(folder),reserved_diagnostic=9))
        for i in range(1,4):
            save(folder/f'replay_{i}'/'cases.json',[{}]*3)
    batch=upgrade/'probe_a/replay_1/batch'
    save(batch/'status.json',dict(stage='complete',policy_steps=500,physics_steps=150000,valid_steps=[500,139,107]))
    (upgrade/'probe_a/replay_2/batch').mkdir()
    save(upgrade/'probe_a/summary.json',dict(passed=False,wall_s_including_queue=177.))
    a,b=resident_probe_accounting(tmp_path)
    assert (a['reserved'],a['entered'],a['complete'],a['incomplete'],a['unstarted'])==(9,6,3,3,3)
    assert a['batches']==['complete','entered_setup','unstarted'] and a['passed'] is False
    assert (b['reserved'],b['entered'],b['unstarted'],b['passed'])==(9,0,9,None)


def test_split_cold_from_resident_and_count_one_queue_call_for_each_lease(tmp_path):
    common=dict(lanes=3,sampling_s=10.,export_s=1.,scoring_and_archive_s=0.5)
    rows=[dict(**common,wall_s=17,measured_wall_includes_queue=True,batch='warm/w00')]
    for directory,pid,times in [('g01',111,[23.,11.5]),('g02',222,[21.])]:
        for i,wall in enumerate(times):
            body=dict(candidate=i,process_id=pid,scene_source=directory+'/c00/batch',
                      app_startup_s=2. if i==0 else 0.,candidate_initialization_s=1.,
                      plant_setup_or_reset_s=.75,sampling_s=10.,export_s=1.)
            folder=f'{directory}/c{i:02d}'
            save(tmp_path/folder/'batch/resident_timing.json',body)
            rows.append(dict(**common,wall_s=wall,batch=folder,measured_wall_includes_queue=i==0))
    report=e1_execution_timing(tmp_path,dict(batches=rows,analysis_and_plot_s=2.))
    assert report['wall_s']==72.5 and report['sampling_s']==40
    assert report['execution_modes']['cold']['wall_s']==17
    resident=report['execution_modes']['resident']
    assert resident['measured_queue_inclusive_calls']==2 and resident['batches']==3
    assert resident['wall_s']==55.5 and len(report['resident_leases'])==2
    assert report['resident_leases']['g01/c00/batch']['wall_s']==34.5
