"""Prepare six alpha0 nominal-program counterfactual suffixes, without integration."""
import argparse
import copy
import json
import subprocess
from pathlib import Path
from prepare_preference_history_factorial import sha,write


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',required=True,type=Path)
    parser.add_argument('--audit',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    a=parser.parse_args();base=Path.cwd();source=a.source.resolve();audit=a.audit.resolve()
    original=json.loads((source/'plan.json').read_text());eligible=json.loads((audit/'review.json').read_text())
    states={r['state_id']:r for r in eligible['records'] if r['eligible']}
    assert len(states)==2 and all(r['original_user_alpha']==0 for r in states.values())
    model=original['actors']['speed_support'];assert sha(model['path'])==model['sha256']
    registry=json.loads((base/'learning/configs/frozen_middle_registry.json').read_text())
    assert registry['records']['STTW_MIDDLE_FULLTAIL_PPO75_U77']['actor_sha256']==model['sha256']
    cases=[]
    for row in original['cases']:
        if row['state_id'] not in states or row['baseline_role']!='speed_support':continue
        program=states[row['state_id']];assert sha(program['program_file'])==program['program_sha256']
        new=copy.deepcopy(row);new.update(policy='H75',baseline_role='nominal_program',gate_mode=0,
            nominal_program_file=program['program_file'],nominal_program_sha256=program['program_sha256'])
        assert new['entry_alpha']==new['future_alpha']==new['alpha']==0
        cases.append(new)
    assert len(cases)==6 and {r['force_N'] for r in cases}=={0.,-32.,32.}
    raw=json.loads((source/'analysis/review.json').read_text())['records']
    cached=[r for r in raw if r['state_id'] in states and r['role'] in ('future0','zero_latent','speed_support')]
    assert len(cached)==18
    for r in cached:assert sha(r['raw_file'])==r['raw_sha256']
    output=a.output.resolve();output.mkdir(exist_ok=False)
    files=list((base/'learning/src/sttw_control').glob('*.py'))+[Path(__file__).resolve(),
        base/'learning/cli/probe_sensor_gate_domain_hold.py',base/'learning/cli/prepare_preference_history_factorial.py',
        base/'learning/cli/audit_nominal_latent_programs.py',Path(original['source_declaration']),Path(model['path']),
        source/'plan.json',source/'analysis/review.json',audit/'review.json']
    files += [Path(r['snapshot']) for r in cases]+[Path(r['program_file']) for r in states.values()]
    files += [Path(r['raw_file']) for r in cached]+[Path(r['raw_file']) for r in states.values()]
    sources={str(p.resolve()):sha(p) for p in files}
    archive=output/'source_at_launch'
    for p in files:
        if p.suffix=='.py' and p.is_relative_to(base):
            target=archive/p.relative_to(base);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(p.read_bytes())
    plan={k:copy.deepcopy(original[k]) for k in ['project','source_declaration','path_evidence_directory',
        'sensor_contract','pulse_duration_s','observation_s','direction_contract','termination','primary_outcome']}
    plan.update(experiment_id='nominal_program_feedback_v1',
        question='Does alpha0 live neural feedback differ from its complete nominal time program under the same lower loop and domain permissions?',
        actors={'H75':model},cases=cases,source_sha256=sources,source_run=str(source),
        source_audit=str(audit/'review.json'),cached_cases=cached,declared_cases=6,
        maximum_lower_ticks=4800,maximum_allocated_lower_slots=5400,new_prefix_ticks=0,training_steps=0,
        code_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        code_identity='actual current .py files SHA archived; dirty source is not an old-code exact replay claim',
        original_alpha=0.,gate_config=original['gate_config'],
        action_contract='at relative tasktick//4 publish frozen nominal bare latent; actual follower/filter/lower/domain/held carry retained; live Actor is logged shadow only',
        scope='DEV alpha0 two eligible origins only; alpha1 missing nominal tail excluded before new outcomes; not independent TEST or deployment policy',
        stop_rule='One six-case diagnostic; engineering error stops without retry; no duration/force/program/alpha expansion regardless of result')
    write(output/'plan.json',plan);write(output/'status.json',dict(phase='prepared',declared_cases=6,
        completed_cases=0,active_lower_ticks=0,training_steps=0))
    print(json.dumps(dict(output=str(output),new_cases=6,cached_cases=18,physics_ticks=0)))


if __name__=='__main__':main()
