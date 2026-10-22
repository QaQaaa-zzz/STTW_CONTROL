"""Export identity-bound finite-grid labels from a completed pulse panel."""
import argparse,hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from sttw_control.empirical_tolerance import summarize_direction

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():raise FileExistsError(a.output)
 status=json.loads((a.run/'status.json').read_text());assert status['phase']=='completed'
 manifest=json.loads((a.run/'manifest.json').read_text());rows=json.loads((a.run/'results.json').read_text())['cases']
 assert len(rows)==status['declared_cases']==status['completed_cases']
 groups=[]
 for alpha,onset in sorted({(r['alpha'],r['onset_s']) for r in rows}):
  subset=[r for r in rows if r['alpha']==alpha and r['onset_s']==onset]
  snapshot=a.run/f'alpha{alpha}_t{onset:g}_snapshot.pkl'
  if not snapshot.exists():
   snapshot=Path(manifest['parent_snapshot']);assert digest(snapshot)==manifest['parent_snapshot_sha256']
  identity=dict(snapshot_sha256=digest(snapshot),endpoint=manifest['checkpoint_identities'][f'alpha{alpha}'],lower=manifest['lower'],physical=manifest['physical_identity'],route=manifest['route'],speed=manifest['v_user'],duration_s=manifest['duration_s'],onset_s=onset,horizon_s=manifest['horizon_s'],force_direction=manifest['force_direction'],force_point=manifest['force_point'],success_spec=manifest['terminal'])
  for sign in [-1,1]:
   selected=[r for r in subset if r['force_N']==0 or r['force_N']*sign>0]
   summary=summarize_direction([(abs(r['force_N']),r['success']) for r in selected])
   groups.append(dict(alpha=alpha,sign=sign,identity=identity,**summary))
 result=dict(source_run=str(a.run.resolve()),source_manifest_sha256=digest(a.run/'manifest.json'),scope='Finite-grid DEV labels only; no continuous lower bound or iid repeat assumption',groups=groups)
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2))
 print(json.dumps([dict(alpha=g['alpha'],onset=g['identity']['onset_s'],sign=g['sign'],status=g['status'],last_pass=g['last_prefix_grid_success_N'],first_failure=g['first_grid_failure_N']) for g in groups],indent=2))
if __name__=='__main__':main()
