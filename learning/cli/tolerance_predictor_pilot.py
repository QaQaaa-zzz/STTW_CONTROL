"""Explicit frozen CPU predictor fit/eval; never runs control physics."""
import argparse
from pathlib import Path
from sttw_control.tolerance_predictor_pilot import fit,evaluate,ranking_fit,confirmation_evaluate,prepare_temporal_mix,temporal_mix_fit

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='stage',required=True)
    train=sub.add_parser('fit');train.add_argument('--config',type=Path,default=Path('learning/configs/h75_tolerance_predictor_pilot.json'));train.add_argument('--train-analysis',type=Path,required=True);train.add_argument('--dev-analysis',type=Path,required=True);train.add_argument('--output',type=Path,required=True)
    ranking=sub.add_parser('ranking-fit')
    ranking.add_argument('--config',type=Path,required=True);ranking.add_argument('--train-analysis',type=Path,required=True);ranking.add_argument('--dev-analysis',type=Path,required=True);ranking.add_argument('--output',type=Path,required=True)
    preparation=sub.add_parser('temporal-prepare');preparation.add_argument('--config',type=Path,required=True);preparation.add_argument('--train-analysis',type=Path,required=True);preparation.add_argument('--dev-analysis',type=Path,required=True);preparation.add_argument('--late-evaluation',type=Path,required=True);preparation.add_argument('--q-export',type=Path,required=True);preparation.add_argument('--output',type=Path,required=True)
    mixed=sub.add_parser('temporal-fit');mixed.add_argument('--config',type=Path,required=True);mixed.add_argument('--prepared',type=Path,required=True);mixed.add_argument('--output',type=Path,required=True)
    confirm=sub.add_parser('confirmation-eval');confirm.add_argument('--manifest',type=Path,required=True);confirm.add_argument('--analysis',type=Path,required=True);confirm.add_argument('--output',type=Path,required=True)
    test=sub.add_parser('eval');test.add_argument('--output',type=Path,required=True);test.add_argument('--heldout-analysis',type=Path,required=True)
    args=parser.parse_args()
    if args.stage=='fit':fit(args.config,args.train_analysis,args.dev_analysis,args.output)
    elif args.stage=='ranking-fit':ranking_fit(args.config,args.train_analysis,args.dev_analysis,args.output)
    elif args.stage=='temporal-prepare':prepare_temporal_mix(args.config,args.train_analysis,args.dev_analysis,args.late_evaluation,args.q_export,args.output)
    elif args.stage=='temporal-fit':temporal_mix_fit(args.config,args.prepared,args.output)
    elif args.stage=='confirmation-eval':confirmation_evaluate(args.manifest,args.analysis,args.output)
    else:evaluate(args.output,args.heldout_analysis)
