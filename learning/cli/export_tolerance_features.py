"""Extract pre-pulse state features without loading any post-pulse trajectories."""
import sys,argparse,json,pickle,hashlib
from pathlib import Path
from dataclasses import asdict,replace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from sttw_control.tolerance_state_features import snapshot_features
from sttw_control.env import load_config
from sttw_control.teleop_commands import CONFIG_PATH

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():raise FileExistsError(a.output)
 manifest=json.loads((a.run/'manifest.json').read_text());config_path=CONFIG_PATH.parent/'fixed_endpoint_directional_alpha0.json';cc=asdict(replace(load_config(config_path).controller,fixed_roll_reference=None));records=[]
 for c in manifest['contexts']:
  for onset in manifest['pulse_times_s']:
   name=f"{c['route']}_v{c['speed']:g}_a{c['alpha']}_t{onset:g}_snapshot.pkl";path=a.run/name
   if not path.exists():continue
   with path.open('rb') as f:snapshot=pickle.load(f)
   record=dict(context=c,onset_s=onset,snapshot=str(path),snapshot_sha256=digest(path),features=snapshot_features(snapshot,speed_request=c['speed'],onset_s=onset,controller=cc,horizon_s=manifest['horizon_s']))
   records.append(record)
 output=dict(schema='tolerance_pre_pulse_features_v1',source_manifest_sha256=digest(a.run/'manifest.json'),controller_config=cc,controller_config_sha256=digest(config_path),feature_source_sha256=digest(Path(__file__).resolve().parents[1]/'src/sttw_control/tolerance_state_features.py'),snapshot_count=len(records),declared_snapshot_count=len(manifest['contexts'])*len(manifest['pulse_times_s']),records=records,scope='Inputs use only genuine pre-pulse state and original task. Model features are not force margins, full Markov state, or deployment-observable guarantees. No post-pulse result file is read.')
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(output,indent=2));print('snapshots',len(records),'features',len(records[0]['features']) if records else 0)
if __name__=='__main__':main()
