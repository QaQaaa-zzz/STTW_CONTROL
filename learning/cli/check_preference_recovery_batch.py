"""Real512-lane conditional recovery engineering; no PPO."""
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from sttw_control.short_recovery import check_preference_batch

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();check_preference_batch(a.plan,a.output)
