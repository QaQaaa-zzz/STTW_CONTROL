"""Paired, fixed-slot DEV evaluation of a nominal envelope-triggered speed override."""
import argparse
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from pcrp_review_batches import LAB, load, write

PRESELECT = {3.9:0., 4.2:.5, 4.5:1.}
GUARD = dict(enter_margin_rad=.07,release_margin_rad=.14,horizon_s=.25,grid_s=.001)


def prepare(root):
    disk = root.parent
    e2 = disk/'pcrp_decision_20261122/e2/matched'
    bends = disk/'pcrp_local300_p0_20261121/full_ground'
    cases = []
    for ds in (-4,-2,-1,0,1,2,4):
        source = e2/f'ds{ds:+d}cm_a0'
        old = load(source/'cases.json')
        for i,c in enumerate(old):
            if c['seed'] > 404:
                continue
            center = list(PRESELECT)[i//8]
            terrain = Path(c['terrain_path'])
            geometry = load(terrain/'identity.json')
            assert abs(geometry['center_s_m']-center-ds/100)<1e-7
            condition = f'ridge{center}_ds{ds:+d}cm'
            cases.append((condition,center,ds,c))
    assert len(cases)==84
    s = load(e2/'ds+0cm_a0/settings.json')
    s['plant_options'].pop('replay_clone_origins')
    # The 84-slot scene uses a new placement. All candidate methods reuse that exact scene.
    bundle = Path('runs/h75_selected_local300_flat/bundle').resolve()
    pulsed = [c for c in load(bends/'both_bends/cases.json') if 's_bend_pulse' in c['output']]
    assert len(pulsed)==5
    bend_raw = load(Path(pulsed[0]['output'])/'contract.json')
    bend_settings = load(bends/'settings.json')
    root.mkdir(parents=True,exist_ok=False)
    blocks, lanes = [], []
    for scene,b,common,series,methods in (
            ('ridges',bundle,s,cases,['fixed','fixed_guard','preselected','preselected_guard','Zero']),
            ('pulsed',Path(bend_raw['bundle']),bend_settings,[('s_bend_pulse',None,0,c) for c in pulsed],['fixed','fixed_guard','Zero'])):
        jobs = []
        for method in methods:
            folder = root/scene/method
            folder.mkdir(parents=True)
            settings = copy.deepcopy(common)
            if method.endswith('_guard'):
                settings['roll_envelope_brake']=GUARD
            settings['evaluation_role']='Matched exploratory online-brake DEV'
            settings['omit_diagnostic_fields']=['observation','policy_measurement']
            rows = []
            for slot,(cond,center,ds,old) in enumerate(series):
                a = PRESELECT[center] if method.startswith('preselected') else (0. if method=='Zero' else .5)
                raw=folder/f'{cond}_g{old["seed"]}'
                c = {k:old[k] for k in ('terrain_path','randomization','seed','user_speed_m_s')}
                if 'evaluation_pulse' in old:
                    c['evaluation_pulse'] = old['evaluation_pulse']
                c.update(alphas=[a],zero_middle=[method=='Zero'],case_labels=[method],output=str(raw))
                rows.append(c)
                lanes.append(dict(condition=cond,center_m=center,ds_cm=ds,group=c['seed'],slot=slot,
                                  method=method,alpha=a,guard=method.endswith('_guard'),raw=str(raw)))
            for key in ('zero_reference_channels','case_labels'):
                settings.pop(key,None)
            write(folder/'settings.json',settings)
            write(folder/'cases.json',rows)
            jobs.append(dict(config=str(folder/'settings.json'),case_manifest=str(folder/'cases.json'),
                             terrain=rows[0]['terrain_path'],output=str(folder/'batch')))
        write(root/scene/'jobs.json',jobs)
        blocks.append(dict(scene=scene,bundle=str(b),methods=methods,slots=len(series)))
    with (root/'episodes.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(lanes[0]));writer.writeheader();writer.writerows(lanes)
    assert len(lanes)==435
    manifests=sorted([*root.glob('*/jobs.json'),*root.glob('*/*/cases.json'),*root.glob('*/*/settings.json'),root/'episodes.csv'])
    hashes={str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest() for f in manifests}
    write(root/'protocol.json',dict(role='Exploratory paired DEV; zero training updates',episodes=435,groups=89,
         blocks=blocks,guard=GUARD,identities=hashes,seconds_per_episode=10,
         inputs='50 Hz current local IMU tilt, causal backward tilt difference, steer encoder and measured rear axle speed times 0.1 m; wheel-speed proxy may slip. No ground-truth forward speed, contact load, location or future samples enter the intervention.',
         treatment='Apply the maximum negative speed offset through the existing 200 Hz bounded shaper. Steering and reward preference stay with the chosen PCRP policy.',
         pairing='84 newly arranged ridge slots and five pulsed-bend slots. Reuse identical slot/car/terrain/clock after fresh PhysX reset across methods per scene. No bitwise equivalence claim to historical E2. Preselection only sees nominal ridge center, blind to shifts. Both preselection labels on pulsed bend reuse the fixed .5 lanes. At center 4.2, fixed/preselected are technical repeats, never pooled as independent groups.',
         endpoint='Counts of full 10 s, hold, tasks A/B/C, roll/other termination, first trigger and total trigger duration, paired rescues/losses, XY and reward time series. Survival gains are paired observations. Persistent failures do not prove physical uncontrollability.',
         stop='Stop after 435 trajectories or first process/output error. No selection or retraining.'))


def run(root):
    plan=load(root/'protocol.json')
    for f,h in plan['identities'].items():
        assert hashlib.sha256((root/f).read_bytes()).hexdigest()==h
    def notify(tag,message):
        receipt=root/(tag+'.json')
        result=subprocess.run([sys.executable,'learning/scripts/notify_sttw_job.py','--message',message,'--receipt',str(receipt)])
        if result.returncode:
            write(receipt,dict(status='failed',exit_code=result.returncode,message=message))
    for block in plan['blocks']:
        scene=block['scene']
        completed=root/scene/'complete.json'
        if completed.exists():
            continue
        jobs=root/scene/'jobs.json'
        write(root/'status.json',dict(stage='running',scene=scene))
        notify(scene+'_start',f'PCRP WP2 在线包络干预实验开始 {scene}，每批 {block["slots"]} 环境')
        cmd=[LAB,'-u','learning/cli/evaluate_isaac_gpu.py','--backend','lab','--bundle',block['bundle'],
             '--resident-jobs',str(jobs),'--seconds','10']
        with (root/scene/'simulation.log').open('a') as log:
            rc=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONPATH=str(Path('learning/src').resolve())+os.pathsep+os.getenv('PYTHONPATH',''))).returncode
        jobs_data=load(jobs)
        if rc or any(not (Path(j['output'])/'status.json').exists() or load(Path(j['output'])/'status.json')['policy_steps']!=500
                or any(not (Path(c['output'])/'trajectory.npz').is_file() for c in load(Path(j['case_manifest']))) for j in jobs_data):
            write(root/'status.json',dict(stage='error',scene=scene,exit_code=rc))
            notify(scene+'_error',f'PCRP WP2 {scene} 异常，请查阅 {root/scene}/simulation.log')
            raise SystemExit(rc or 1)
        write(completed,block)
        notify(scene+'_complete',f'PCRP WP2 {scene} 轨迹已保存至 {root/scene}')
    write(root/'status.json',dict(stage='physics_complete',episodes=plan['episodes']))
    cpu = '/home/qy/miniconda3/envs/mjp/bin/python'
    analyze_cmd = [cpu,'docs/paper/english/analyze_online_brake.py',str(root)]
    result = subprocess.run(analyze_cmd,env=dict(os.environ,PYTHONPATH=str(Path('learning/src').resolve())+os.pathsep+os.getenv('PYTHONPATH','')))
    if result.returncode:
        notify('analysis_error',f'PCRP WP2 轨迹已保存，分析失败 {root}')
        raise SystemExit(result.returncode)
    notify('report_complete',f'PCRP WP2 配对分析已完成 {root}/analysis/README.md')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','run'])
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    (prepare if args.action=='prepare' else run)(args.output.resolve())
