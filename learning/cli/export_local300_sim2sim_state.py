"""Export the existing MuJoCo standard-case initial state; no physics stepping."""
import argparse,json,pickle
from pathlib import Path
import numpy as np
from sttw_control.model import load_model
from sttw_control.isaac_portable_state import export_physical_states
p=argparse.ArgumentParser();p.add_argument('--all-states',action='store_true');p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
s=json.loads(a.config.read_text());bank=pickle.load(open(s['initial_bank'],'rb'));bundle=load_model()
indices=range(len(bank.physical_tick)) if a.all_states else [s['initial_bank_index']]
state=export_physical_states(bank,bundle,indices)
if not a.all_states:state={key:value if key=='joint_names' else value[0] for key,value in state.items()}
a.output.parent.mkdir(parents=True,exist_ok=True);np.savez(a.output,**state);print('exported',a.output,'physical_tick',state['physical_tick'])
