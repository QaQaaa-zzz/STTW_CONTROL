"""Explicit frozen CPU predictor fit/eval; never runs control physics."""
import argparse
from pathlib import Path
from sttw_control.tolerance_predictor_pilot import fit,evaluate

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='stage',required=True)
    train=sub.add_parser('fit');train.add_argument('--config',type=Path,default=Path('learning/configs/h75_tolerance_predictor_pilot.json'));train.add_argument('--train-analysis',type=Path,required=True);train.add_argument('--dev-analysis',type=Path,required=True);train.add_argument('--output',type=Path,required=True)
    test=sub.add_parser('eval');test.add_argument('--output',type=Path,required=True);test.add_argument('--heldout-analysis',type=Path,required=True)
    args=parser.parse_args()
    if args.stage=='fit':fit(args.config,args.train_analysis,args.dev_analysis,args.output)
    else:evaluate(args.output,args.heldout_analysis)
