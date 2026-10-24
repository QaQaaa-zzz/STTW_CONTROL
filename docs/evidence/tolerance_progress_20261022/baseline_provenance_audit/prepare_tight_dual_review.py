"""Check historical endpoint specialists in the identical new task contract."""
import argparse
import copy
import json
import pickle
from pathlib import Path


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--nominal-run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    from sttw_control.preference_best import atomic_json,digest
    from sttw_control.path_command_policy import validate_actor_parameters
    source=args.nominal_run.resolve();root=args.output.resolve()
    root.mkdir(parents=True,exist_ok=False)
    registry_file=Path('learning/configs/frozen_middle_registry.json').resolve()
    registry=json.loads(registry_file.read_text());frozen=Path(registry['root'])
    if digest(frozen/'manifest.json')!=registry['manifest_sha256']:
        raise ValueError('frozen lineage manifest changed')
    parent_plan=json.loads((source/'plan.json').read_text());panels=[];checks=[]
    for panel in parent_plan['panels']:
        parent=json.loads(Path(panel['declaration']).read_text())
        for alpha in (0,1):
            key=f'STTW_MIDDLE_DUAL_COVERAGE_U200_A{alpha}'
            record=registry['records'][key]
            for field in ('actor','checkpoint'):
                if digest(record[field])!=record[field+'_sha256']:
                    raise ValueError('frozen specialist identity changed')
            teacher=json.loads((frozen/(key+'_resolved_config.json')).read_text())
            for field in ('context_fields','history_frames','frame_fields','actor_dim','norm','hidden','activation'):
                if teacher['network'][field]!=parent['config']['network'][field]:
                    raise ValueError('teacher observation/network contract differs: '+field)
            if teacher['timing']!=parent['config']['timing']:
                raise ValueError('teacher timing differs')
            # Native legacy architecture, without padding or migrating weights.
            d=copy.deepcopy(parent);d['config']['network']['architecture']='legacy'
            params=pickle.loads(Path(record['actor']).read_bytes())
            validate_actor_parameters(params,d['config'])
            name=panel['name']+f'_A{alpha}';folder=root/name;folder.mkdir()
            d.update(experiment_id='low_tight_dual_nominal_review_v1_'+name,
                     question='Does this tight-scene failure depend on the H75 branch? Revisit frozen dual endpoints.',
                     policies={'dual':{k:record[k] for k in ('actor','actor_sha256','checkpoint','checkpoint_sha256')}},
                     alphas=[float(alpha)],maximum_lower_ticks=8000,
                     data_role='same new-contract development scene, historical specialist review',
                     stop_rule='four cases once; no radius/cap tuning, force or optimization')
            d['source_sha256'][str(Path(__file__).resolve())]=digest(__file__)
            atomic_json(folder/'declaration.json',d);atomic_json(folder/'effective_config.json',d['config'])
            panels.append(dict(name=name,radius_m=panel['radius_m'],alpha=alpha,
                declaration=str(folder/'declaration.json'),output=str(folder/'evaluation'),cases=4,maximum_lower_ticks=8000))
            checks.append(dict(node=key,actor_sha256=record['actor_sha256'],native_architecture='legacy',
                matching_observation_fields=True,matching_timing=True,
                other_config_identical_to=str(panel['declaration']),only_architecture_metadata_changed=True))
    atomic_json(root/'plan.json',dict(experiment='low_tight_dual_nominal_review_v1',panels=panels,
        total_cases=16,maximum_lower_ticks=32000,training_steps=0,checks=checks,
        source_nominal=str(source),source_nominal_plan_sha256=digest(source/'plan.json'),
        baseline_reuse='Use authenticated H75/zero raw trajectories from source_nominal; never rerun them',
        frozen_registry_sha256=digest(registry_file),force_scan='not authorized by this plan'))
    print(json.dumps(dict(cases=16,maximum_lower_ticks=32000,weight_migration=False)))


if __name__=='__main__':main()
