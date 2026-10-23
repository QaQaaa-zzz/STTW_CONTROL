"""Execute the predeclared continuation-length evaluation."""
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from sttw_control.preference_length_evaluation import evaluate
from prepare_short_recovery_campaign import register
p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--output',required=True)
a=p.parse_args();evaluate(a.plan,a.output,register)
