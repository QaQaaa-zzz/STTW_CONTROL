"""Audit saved wheel-contact traces without rerunning simulation or changing labels."""
import argparse,json
from pathlib import Path
import numpy as np
from sttw_control.jump_event_labels import extract_contact_events
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
p.add_argument('--far-edge-x',type=float,required=True);p.add_argument('--far-bank-half-width',type=float,required=True)
a=p.parse_args();d=np.load(a.input);names=d['body_names'].tolist();ids=[names.index(w) for w in ('frontwheel','rearwheel')]
reports=[extract_contact_events(d['time'],d['body_state'][:,lane,ids,:3],d['contact'][:,lane,ids],d['valid'][:,lane],a.far_edge_x,a.far_bank_half_width) for lane in range(d['valid'].shape[1])]
a.output.parent.mkdir(parents=True,exist_ok=True)
a.output.write_text(json.dumps(dict(source=str(a.input.resolve()),role='offline instrumentation audit, not new physical task evaluation',lanes=reports),indent=2)+'\n')
