"""Freeze one checkpoint for native frozen-middle transfer evaluation."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from sttw_control.lower_transfer import export_lower_transfer
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--checkpoint',required=True);p.add_argument('--initial-source',required=True);p.add_argument('--output',required=True)
a=p.parse_args();base=json.loads(Path(a.initial_source).read_text())['candidate']
print(json.dumps(export_lower_transfer(a.checkpoint,base,a.output),indent=2))
