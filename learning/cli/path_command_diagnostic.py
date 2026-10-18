#!/usr/bin/env python3
"""Read-only preparation by default; --execute runs only eight fixed-Actor probes."""
import argparse,json,os,sys
from pathlib import Path
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request',type=Path,default=Path(__file__).resolve().parents[1]/'configs/path_safety_repair_diagnostic.json')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    from sttw_control.path_command_diagnostic import prepare_request,make_transform,compare_prefix,diagnostic_groups
    from sttw_control.preference_best import atomic_json
    from sttw_control.path_command_policy import config_key
    request=json.loads(args.request.read_text());receipt=prepare_request(request)
    if not args.execute:
        print(json.dumps({k:v for k,v in receipt.items() if k not in ('manifest','request','files','execution_manifests')},indent=2));return
    if args.output is None:parser.error('--execute requires immutable --output')
    args.output.mkdir(parents=True,exist_ok=False)
    atomic_json(args.output/'PREPARATION.json',receipt)
    import pickle,jax,jax.numpy as j,numpy as np
    from sttw_control.path_command_training import load_environment
    from sttw_control.path_command_evaluation import BatchedPanelEvaluator
    from sttw_control.path_command_cache import diagnostic_cache_identity
    from sttw_control.runtime import configure_compilation_cache
    configure_compilation_cache(Path('/home/qy/STTW_CONTROL/runs/.jax_compilation_cache'))
    ticks=0
    def account(n):
        nonlocal ticks
        ticks+=n
        atomic_json(args.output/'ACCOUNTING.json',dict(active_lower_ticks=ticks,declared_budget=32000,training_updates=0,state='running'))
        if ticks>32000:raise RuntimeError('diagnostic budget exceeded')
    evaluators={}
    try:
        for group_index,rows in enumerate(diagnostic_groups(request)):
            row=rows[0]
            record=request['models'][row['model']]
            manifest=receipt['execution_manifests'][row['model']];cfg=manifest['config']
            if row['model'] not in evaluators:
                env,snapshot=load_environment(cfg,manifest['source'],'terminal_contract_v2')
                evaluators[row['model']]=BatchedPanelEvaluator(env,snapshot,cfg,account=account)
            evaluator=evaluators[row['model']]
            with Path(record['actor']).open('rb') as f:params=jax.tree.map(j.asarray,pickle.load(f))
            # Existing exports are complete Flax variables, preserving exact Actor identity.
            context=dict(actor=record['actor'],checkpoint=record['checkpoint'],source_update=record.get('source_update'),case_ids=[item['case'] for item in rows],prepared_bank_sha256=manifest['source']['prepared_bank_sha256'],config_key=config_key(cfg))
            identity=dict(context=context,transform=row['transform'],cache_key=diagnostic_cache_identity(context,row['transform']))
            group_output=args.output/f'group_{group_index:02d}'
            traces=evaluator.evaluate(params,1,group_output,case_names=[item['case'] for item in rows],diagnostic_transform=make_transform(cfg,row['transform']),diagnostic_identity=identity)
            for row in rows:
                row_output=args.output/row['id'];row_output.mkdir(exist_ok=False)
                for suffix in ('','_path'):
                    (group_output/(row['case']+suffix+'.npz')).replace(row_output/(row['case']+suffix+'.npz'))
                atomic_json(row_output/'source.json',dict(model=record,diagnostic_identity=identity,batch_timings=str(group_output/'evaluation_timings.json')))
                if row['model']=='u200':
                    original_path=Path(request['cached_panels']['u200'])/(row['case']+'.npz')
                    with np.load(original_path,allow_pickle=False) as z:original={k:z[k] for k in z.files}
                    atomic_json(row_output/'prefix_comparison.json',compare_prefix(original,traces[row['case']]))
        atomic_json(args.output/'ACCOUNTING.json',dict(active_lower_ticks=ticks,declared_budget=32000,training_updates=0,state='completed'))
    except BaseException:
        atomic_json(args.output/'ACCOUNTING.json',dict(active_lower_ticks=ticks,declared_budget=32000,training_updates=0,state='error'));raise

if __name__=='__main__':main()
