"""Freeze and run the paired PCRP frontier and analytical-governor review cases."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

LAB = "/home/qy/miniconda3/envs/env_isaaclab/bin/python"
ALPHAS = [0, .25, .5, .75, 1]
EXTRA = [.1, .2, .3, .4, .6, .7, .8, .9]
TUNED = "cap_L3_a2"
ORIGINAL = "cap_L6_a3.5"


def load(path):
    return json.loads(path.read_text())


def write(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def rule(length, acc, gain=0):
    name = f"cap_L{length:g}_a{acc:g}" if not gain else f"bias_L{length:g}_a{acc:g}_g{gain:g}"
    return dict(id=name, preview_m=length, lateral_acceleration_limit_m_s2=acc, curvature_bias_gain=gain)


def prepare(root):
    root.mkdir(parents=True, exist_ok=False)
    disk = root.parent
    p0 = disk / "pcrp_local300_p0_20261121/full_ground"
    original = disk / "pcrp_curvature_governor_20261121"
    tuned = disk / "pcrp_p1_20261121/matched"
    layouts = disk / "h75_theory_iteration/layout_test"
    rules = [rule(l, a) for l in (3, 4.5, 6) for a in (1.5, 2, 2.5, 3, 3.5, 4)]
    rules += [rule(l, 3.5, g) for l in (3, 4.5, 6) for g in (-.5, -.25, .25, .5)]
    nominal_missing = [r for r in rules if r['curvature_bias_gain'] == 0 and
                       r['lateral_acceleration_limit_m_s2'] >= 2 and r['id'] not in (TUNED, ORIGINAL)]
    stages, index = {}, []

    def register(stage, condition, seed, method, cohort, alpha, raw, lane, reference, ref_lane):
        index.append(dict(stage=stage, condition=condition, group=seed, method=method, cohort=cohort,
                          alpha=alpha, raw=str(raw), lane=lane, h75_reference=str(reference), h75_lane=ref_lane))

    def add_batch(name, source, output, labels, alphas, lane_rules):
        contract = load(source / "contract.json")
        s = contract['settings']
        fields = ['asset', 'controller_config', 'lower_actor', 'lower_sha256', 'plant_options',
                  'initial_state', 'selected_lower']
        config = {k: s[k] for k in fields}
        config['asset'] = str(Path(config['asset']).resolve())
        config.update(episode_s=10, auto_reset_horizon=False, diagnostics=True, reward_components=True,
                      record_controller_terms=True, paired_reward_diagnostics=False, compact_results=True,
                      omit_diagnostic_fields=['observation', 'policy_measurement'],
                      evaluation_role="Development comparison with archived matched physical groups")
        stage = stages.setdefault(name, dict(cases=[], config=config, bundle=contract['bundle']))
        assert config == stage['config'] and contract['bundle'] == stage['bundle'], (name, source)
        case = dict(seed=s['seed'], randomization=s['randomization'], user_speed_m_s=s['user_speed_m_s'],
                    terrain_path=s['terrains']['evaluation'], output=str(output), case_labels=labels,
                    alphas=alphas, zero_middle=[False]*len(labels), traditional_rules=lane_rules)
        if 'evaluation_pulse' in s:
            case['evaluation_pulse'] = s['evaluation_pulse']
        assert (Path(case['terrain_path']) / 'terrain.npz').is_file()
        stage['cases'].append(case)

    for cond in ('s_bend_pulse', 'right_bend', 'flat', 'oblique_2cm', 'oblique_4cm'):
        for seed in range(101, 106):
            bend = cond in ('s_bend_pulse', 'right_bend')
            src = p0 / cond / f'flat_{seed}' if bend else (Path('runs/h75_tradeoff_repeats') /
                        f'{cond}_v2p6_seed{seed}' / 'evaluation').resolve()
            old = load(src / 'contract.json')['settings']
            assert old['alphas'][:5] == ALPHAS and old['zero_middle'][5]
            assert (src / 'trajectory.npz').is_file()
            name = cond if bend else "error_plane"
            out = root / name / f'{cond}_g{seed}'
            labels = [r['id'] for r in rules] + [f'H75_{a:g}' for a in EXTRA]
            add_batch(name, src, out, labels, [.5]*len(rules)+EXTRA, rules+[None]*len(EXTRA))
            for i, a in enumerate(ALPHAS):
                register('T3A', cond, seed, f'H75_{a:g}', 'h75', a, src, i, src, i)
            register('T3A', cond, seed, 'Zero', 'zero', 0, src, 5, src, 5)
            for i, r in enumerate(rules):
                register('T3A', cond, seed, r['id'], 'governor', .5, out, i, src, 2)
            for i, a in enumerate(EXTRA, len(rules)):
                register('T3A', cond, seed, f'H75_{a:g}', 'h75', a, out, i, src, 2)

    sources = load(original / 'source_cases.json')
    for key, raw in sources.items():
        src = Path(raw)
        cond, seed_str = key.rsplit('_seed', 1)
        seed = int(seed_str)
        out = root / 'governors' / ('nominal_' + key)
        labels = [r['id'] for r in nominal_missing]
        add_batch('governors', src, out, labels, [.5]*len(labels), nominal_missing)
        for i,a in enumerate(ALPHAS):
            register('T2_nominal',cond,seed,f'H75_{a:g}','h75',a,src,i,src,i)
        register('T2_nominal',cond,seed,'Zero','zero',0,src,5,src,5)
        for label, case in [(ORIGINAL,original/key),(TUNED,tuned/key)]:
            assert (case/'trajectory.npz').is_file()
            register('T2_nominal',cond,seed,label,'governor',.5,case,0,src,2)
        for i, label in enumerate(labels):
            register('T2_nominal',cond,seed,label,'governor',.5,out,i,src,2)

    for job in load(layouts / 'jobs.json'):
        key = job['name']
        cond, seed_str = key.rsplit('_seed', 1)
        seed = int(seed_str)
        src = layouts/key
        s = load(src/'contract.json')['settings']
        for j, label in enumerate(s['case_labels'][:9]):
            cohort = 'fixed' if j < 5 else ('zero' if j == 5 else 'selector')
            register('T2_selector',cond,seed,label,cohort,s['alphas'][j],src,j,src,j)
        if key in sources:
            assert Path(sources[key]).resolve() == src.resolve()
            out = tuned/key
        else:
            out = root/'governors'/('selector_'+key)
            add_batch('governors', src, out, [TUNED], [.5], [rule(3,2)])
        register('T2_selector',cond,seed,TUNED,'governor',.5,out,0,src,2)

    plan = {}
    for name, stage in stages.items():
        folder = root/name
        folder.mkdir()
        groups = {}
        for c in stage['cases']:
            assert c['randomization'] == groups.setdefault(c['seed'], c['randomization'])
        write(folder/'cases.json',stage['cases'])
        write(folder/'settings.json',stage['config'])
        plan[name] = dict(bundle=stage['bundle'], lanes=sum(len(c['alphas']) for c in stage['cases']),
                          cases=len(stage['cases']), manifest_sha256=hashlib.sha256((folder/'cases.json').read_bytes()).hexdigest())
    assert [s['lanes'] for s in plan.values()] == [190,190,570,688]
    with (root/'episodes_index.csv').open('w',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(index[0]))
        writer.writeheader()
        writer.writerows(index)
    write(root/'protocol.json',dict(batches=plan,horizon_s=10,dt=[.0002,.005,.02],training_updates=0,
          role='Paired previously examined development cases', archive_geometry='Full-ground corrected bends',
          lanes=1638, maximum_physics_lane_steps=81900000, new_technical_repeats=0,
          stop='All four batches complete or first execution error. No automatic reruns.',
          comparison='Common observation prefix across all declared methods per condition/group. Equal average of the five physical-group RMSEs. Aggregate full 10 s and final hold separately. Reference for minimizing 2D hypervolume is (0.80 m cross-track, 0.30 m/s speed), normalized by product. Report strict dominance counts and all points.',
          pairing='Same local initial state, fixed physical parameters, original route, task, pulse, stack, and clock. Independent batch/clone slots; bitwise replay not established.',
          command='Run from sttw-gravel with PYTHONPATH=learning/src. Queue wraps this CLI run.' ))


def run(root):
    protocol = load(root/'protocol.json')
    def notice(tag, body):
        receipt = root/(tag+'.json')
        command = [sys.executable,'learning/scripts/notify_sttw_job.py','--message',body,'--receipt',str(receipt)]
        result = subprocess.run(command, check=False)
        if result.returncode:
            write(receipt,dict(status='failed',exit_code=result.returncode,message=body))
    for name, stage in protocol['batches'].items():
        folder = root/name
        cases = load(folder/'cases.json')
        assert hashlib.sha256((folder/'cases.json').read_bytes()).hexdigest()==stage['manifest_sha256']
        finished = folder/'batch/status.json'
        if finished.is_file() and load(finished).get('stage')=='complete':
            assert all((Path(c['output'])/'trajectory.npz').is_file() for c in cases)
            continue
        notice(name+'_start',f"PCRP 第一章 {name} 仿真开始，共 {stage['lanes']} 回合")
        write(root/'status.json',dict(stage='running',batch=name,utc=datetime.now(timezone.utc).isoformat()))
        cmd = [LAB,'-u','learning/cli/evaluate_isaac_gpu.py','--backend','lab','--config',str(folder/'settings.json'),
               '--case-manifest',str(folder/'cases.json'),'--bundle',stage['bundle'],'--terrain',cases[0]['terrain_path'],
               '--output',str(folder/'batch'),'--seconds','10']
        with (folder/'simulation.log').open('a') as log:
            result = subprocess.run(cmd, stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONPATH=str(Path('learning/src').resolve())+os.pathsep+os.getenv('PYTHONPATH','')))
        if result.returncode or not finished.is_file() or load(finished).get('policy_steps')!=500:
            write(root/'status.json',dict(stage='error',batch=name,exit_code=result.returncode))
            notice(name+'_error',f"PCRP 第一章 {name} 仿真异常，查看 {folder}/simulation.log")
            raise SystemExit(result.returncode or 1)
        notice(name+'_complete',f"PCRP 第一章 {name} 仿真完成，轨迹位于 {folder}")
    write(root/'status.json',dict(stage='physics_complete',utc=datetime.now(timezone.utc).isoformat()))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','run'))
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    (prepare if args.action=='prepare' else run)(args.output.resolve())
