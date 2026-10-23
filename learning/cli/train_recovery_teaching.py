import argparse
from sttw_control.recovery_teaching_training import train
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--declaration',required=True);p.add_argument('--output',required=True);a=p.parse_args();train(a.declaration,a.output)
