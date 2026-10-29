"""Explicit frozen CPU predictor fit/eval; never runs control physics."""
import argparse
from pathlib import Path
from sttw_control.tolerance_predictor_pilot import fit,evaluate,ranking_fit,confirmation_evaluate,prepare_temporal_mix,temporal_mix_fit,prepare_user_mix,user_mix_fit,prepare_user_mix_reference,five_alpha_fit

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='stage',required=True)
    train=sub.add_parser('fit');train.add_argument('--config',type=Path,default=Path('learning/configs/h75_tolerance_predictor_pilot.json'));train.add_argument('--train-analysis',type=Path,required=True);train.add_argument('--dev-analysis',type=Path,required=True);train.add_argument('--output',type=Path,required=True)
    ranking=sub.add_parser('ranking-fit')
    ranking.add_argument('--config',type=Path,required=True);ranking.add_argument('--train-analysis',type=Path,required=True);ranking.add_argument('--dev-analysis',type=Path,required=True);ranking.add_argument('--output',type=Path,required=True)
    preparation=sub.add_parser('temporal-prepare');preparation.add_argument('--config',type=Path,required=True);preparation.add_argument('--train-analysis',type=Path,required=True);preparation.add_argument('--dev-analysis',type=Path,required=True);preparation.add_argument('--late-evaluation',type=Path,required=True);preparation.add_argument('--q-export',type=Path,required=True);preparation.add_argument('--output',type=Path,required=True)
    mixed=sub.add_parser('temporal-fit');mixed.add_argument('--config',type=Path,required=True);mixed.add_argument('--prepared',type=Path,required=True);mixed.add_argument('--output',type=Path,required=True)
    userprep=sub.add_parser('user-mix-prepare');userprep.add_argument('--config',type=Path,required=True);userprep.add_argument('--ancestor-prepared',type=Path,required=True);userprep.add_argument('--user-evaluation',type=Path,required=True);userprep.add_argument('--q-export',type=Path,required=True);userprep.add_argument('--output',type=Path,required=True)
    priorprep=sub.add_parser('user-mix-prior-prepare');priorprep.add_argument('--config',type=Path,required=True);priorprep.add_argument('--prepared-reference',type=Path,required=True);priorprep.add_argument('--output',type=Path,required=True)
    userfit=sub.add_parser('user-mix-fit');userfit.add_argument('--config',type=Path,required=True);userfit.add_argument('--prepared',type=Path,required=True);userfit.add_argument('--output',type=Path,required=True)
    five=sub.add_parser('five-alpha-fit');five.add_argument('--config',type=Path,required=True);five.add_argument('--analysis',type=Path,nargs='+',help='Defaults to dataset.analysis or dataset.analyses in frozen config');five.add_argument('--output',type=Path,required=True)
    confirm=sub.add_parser('confirmation-eval');confirm.add_argument('--manifest',type=Path,required=True);confirm.add_argument('--analysis',type=Path,required=True);confirm.add_argument('--output',type=Path,required=True)
    test=sub.add_parser('eval');test.add_argument('--output',type=Path,required=True);test.add_argument('--heldout-analysis',type=Path,required=True)
    args=parser.parse_args()
    if args.stage=='fit':fit(args.config,args.train_analysis,args.dev_analysis,args.output)
    elif args.stage=='ranking-fit':ranking_fit(args.config,args.train_analysis,args.dev_analysis,args.output)
    elif args.stage=='temporal-prepare':prepare_temporal_mix(args.config,args.train_analysis,args.dev_analysis,args.late_evaluation,args.q_export,args.output)
    elif args.stage=='temporal-fit':temporal_mix_fit(args.config,args.prepared,args.output)
    elif args.stage=='user-mix-prepare':prepare_user_mix(args.config,args.ancestor_prepared,args.user_evaluation,args.q_export,args.output)
    elif args.stage=='user-mix-prior-prepare':prepare_user_mix_reference(args.config,args.prepared_reference,args.output)
    elif args.stage=='user-mix-fit':
        from engineer_reference_bias import notify
        try:user_mix_fit(args.config,args.prepared,args.output,on_training_complete=lambda receipt:notify(args.output.parent,'STTW容扰覆盖训练阶段完成','三seed各1000步已结束，开始只读分析。'))
        except BaseException:
            notify(args.output.parent,'STTW容扰覆盖训练报错','保留实际更新数与日志，不自动重试。')
            raise
        notify(args.output.parent,'STTW容扰覆盖训练流水线完成','3000步训练与离线分析完成；未执行控制物理。')
    elif args.stage=='five-alpha-fit':
        from engineer_reference_bias import notify
        try:five_alpha_fit(args.config,args.analysis,args.output,on_training_complete=lambda receipt:notify(args.output.parent,'STTW五alpha容扰训练阶段完成',f"三seed固定预算已结束，共{receipt['actual_optimizer_updates']}次更新，开始离线分析。"))
        except BaseException:
            notify(args.output.parent,'STTW五alpha容扰训练报错','保留实际更新与日志，不自动重试。')
            raise
        notify(args.output.parent,'STTW五alpha容扰流水线完成','固定预算训练、分组指标及三checkpoint导出完成；未执行控制物理。')
    elif args.stage=='confirmation-eval':confirmation_evaluate(args.manifest,args.analysis,args.output)
    else:evaluate(args.output,args.heldout_analysis)
