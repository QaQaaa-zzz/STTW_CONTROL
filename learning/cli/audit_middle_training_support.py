"""Audit actual reference-task bank coverage separately from template ranges."""
import argparse
import hashlib
import json
import pickle
from collections import Counter
from pathlib import Path
import numpy as np


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''):h.update(block)
    return h.hexdigest()


def interface(config):
    c=config['residual_interface']
    return {k:c[k] for k in ['speed_reference_bounds_m_s','steer_reference_bound_rad',
        'speed_negative_scale_m_s','speed_positive_scale_m_s','steer_scale_rad','filter_tau_s']}


def bank_summary(directory):
    directory=Path(directory);receipt=json.loads((directory/'receipt.json').read_text())
    assert digest(directory/'bank.pkl')==receipt['bank_sha256']
    bank=pickle.loads((directory/'bank.pkl').read_bytes());valid=np.asarray(bank['valid'],bool)
    routes=bank['routes'];speed=np.asarray(routes.speed)[valid];radius=np.asarray(routes.radius)[valid]
    family=np.asarray(routes.family)[valid];alpha=np.asarray(bank['states'].alpha)[valid]
    groups={}
    for f in sorted(set(family.tolist())):
        selected=family==f
        groups[str(f)]=dict(n=int(selected.sum()),requested_speed_m_s=[float(speed[selected].min()),float(speed[selected].max())],
            radius_parameter_m=[float(radius[selected].min()),float(radius[selected].max())])
    return dict(valid_contexts=int(valid.sum()),all_contexts=len(valid),
        requested_speed_m_s=[float(speed.min()),float(speed.max())],
        radius_parameter_m=[float(radius.min()),float(radius.max())],
        alpha_counts={str(float(k)):int(v) for k,v in zip(*np.unique(alpha,return_counts=True))},
        by_family=groups,bank_sha256=receipt['bank_sha256'],
        limitation='Requested route parameters only; not observed physical speed, all 351D input support, or realized visitation probabilities')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    base=Path.cwd();frozen=Path('/home/qy/STTW_CONTROL/runs/model_registry/middle_lineage_20261024_v1')
    manifest=json.loads((frozen/'manifest.json').read_text());node=next(r for r in manifest['nodes'] if r['id']=='STTW_MIDDLE_FULLTAIL_PPO75_U77')
    inputs={str(frozen/'manifest.json'):digest(frozen/'manifest.json')}
    for record in [node['actor'],node['checkpoint'],node['manifest'],*node['evidence']]:
        obj=frozen/'objects'/record['object'];assert digest(obj)==record['sha256'];inputs[str(obj)]=record['sha256']
    config=json.loads((frozen/node['resolved_config']['path']).read_text());assert digest(frozen/node['resolved_config']['path'])==node['resolved_config']['sha256']
    hplan=json.loads((base/'runs/preference_full_tail_length75_v1/uniform/plan.json').read_text())
    bank_dir=Path(hplan['bank_dir']);hbank=bank_summary(bank_dir)
    declaration=json.loads((base/'runs/preference_full_tail_length75_v1/training_declaration.json').read_text())
    assert digest(bank_dir/'manifest.json')==declaration['bank_manifest_sha256']
    assert digest(bank_dir/'receipt.json')==declaration['bank_receipt_sha256']
    low=base/'runs/reserve_feature_expansion_v1/uniform_stage2';lp=json.loads((low/'plan.json').read_text())
    lbank=bank_summary(lp['bank_dir']);lc=json.loads(Path(lp['config']).read_text())
    done=json.loads((low/'last_completed.json').read_text());assert done['update']==77
    assert digest(done['checkpoint'])==done['checkpoint_sha256']
    online=base/'runs/online_preference_survival_v1_attempt3';op=json.loads((online/'plan.json').read_text());oc=json.loads((online/'effective_config.json').read_text())
    status=json.loads((online/'status.json').read_text());assert status['phase']=='completed' and status['completed_updates']==64
    cases=op['cases'];contexts=[]
    for route,speed,alpha in sorted({(r['route'],r['speed'],r['alpha']) for r in cases}):
        group=[r for r in cases if (r['route'],r['speed'],r['alpha'])==(route,speed,alpha)]
        contexts.append(dict(route=route,requested_speed_m_s=speed,alpha=alpha,
            force_N=sorted({r['force_N'] for r in group}),onset_task_ticks=sorted({r['pulse_onset_task_tick'] for r in group}),lanes=len(group)))
    recent=base/'runs/nominal_program_feedback_v1/plan.json';rp=json.loads(recent.read_text());rc=json.loads(Path(rp['source_declaration']).read_text())['config']
    for directory in [bank_dir,Path(lp['bank_dir'])]:
        for name in ['receipt.json','manifest.json']:inputs[str(directory/name)]=digest(directory/name)
    for path in [base/'runs/preference_full_tail_length75_v1/uniform/plan.json',low/'plan.json',low/'last_completed.json',
                 Path(lp['config']),online/'plan.json',online/'effective_config.json',online/'effective_training_contract.json',online/'status.json',recent,Path(rp['source_declaration'])]:
        inputs[str(path)]=digest(path)
    result=dict(H75=dict(actor_sha256=node['actor']['sha256'],bank=hbank,interface=interface(config),
        template_speed_range_m_s=config['training_future_phase_C']['speed_range_m_s'],
        initializer='u325 parent plus bank recovery PPO; this bank is not the full lifetime observation distribution'),
        low_speed_75=dict(bank=lbank,interface=interface(lc),completed_update=77,checkpoint_sha256=done['checkpoint_sha256']),
        online64=dict(completed_updates=64,contexts=contexts,interface=interface(oc),
            initialization=op['snapshot'],training_case_count=len(cases),scope='Two task contexts, fixed planned onsets, stochastic actions, original 10s task and held physical tail'),
        recent_mechanism=dict(requested_speed_m_s=1.5,radius_m=1.1,interface=interface(rc),suffix_s=4.,user_alpha=0.),
        interpretation=['Recent H75 mechanism results are transferred request/geometry/reference-authority conditions.',
            'Within-experiment matched control comparisons remain valid; transfer is not a reason to erase negative results.',
            'No causal decomposition of distribution vs authority vs reward vs training length was performed.',
            'Online64 already trained low-speed tight-turn expanded-interface cases; do not claim this project never covered them.',
            'Two contexts/fixed onset/directional exposures are not full coverage, but missing coverage does not establish that more training improves strong baselines.'],
        input_sha256=inputs,new_physics_ticks=0,new_training_steps=0)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='input_sha256'},indent=2))


if __name__=='__main__':main()
