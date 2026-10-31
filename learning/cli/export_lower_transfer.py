"""Freeze one checkpoint for native frozen-middle transfer evaluation."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from sttw_control.lower_transfer import export_lower_transfer,select_scored_lower_checkpoint
p=argparse.ArgumentParser(description=__doc__)
source=p.add_mutually_exclusive_group(required=True)
source.add_argument('--checkpoint');source.add_argument('--training-run')
p.add_argument('--initial-source',required=True);p.add_argument('--output',required=True)
a=p.parse_args();base=json.loads(Path(a.initial_source).read_text())['candidate']
selection=None
if a.training_run:
    selection=select_scored_lower_checkpoint(a.training_run)
    if selection['selected'] is None:p.error('no saved continuation has its own score and actual positive/negative exposure')
    a.checkpoint=selection['selected']['checkpoint']
print(json.dumps(export_lower_transfer(a.checkpoint,base,a.output),indent=2))
if selection is not None:(Path(a.output)/'selection.json').write_text(json.dumps(selection,indent=2)+'\n')
