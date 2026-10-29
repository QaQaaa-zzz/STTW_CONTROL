"""Explicit preparation or execution of the declared mixed-route continuation."""
import argparse
import os
import sys
from pathlib import Path
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from sttw_control.path_mixed_finetuning import prepare, train


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',default='learning/configs/h75_low_speed_mixed_finetune.json')
    p.add_argument('--output',required=True)
    p.add_argument('--execute',action='store_true')
    a=p.parse_args()
    result=(train if a.execute else prepare)(a.config,a.output)
    if not a.execute:print("prepared_not_run; policy transitions="+str(result["policy_transitions"])+"; physics=0")


if __name__=='__main__':main()
