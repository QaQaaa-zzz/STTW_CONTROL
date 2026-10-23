"""Wait for the declared matched pilot and evaluate its fixed final actors."""
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from prepare_short_recovery_campaign import register
from sttw_control.preference_recovery_evaluation import finish

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--campaign',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();finish(a.campaign,a.output,register)
