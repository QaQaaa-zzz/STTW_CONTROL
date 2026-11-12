"""Build a compact evidence archive, preserving numeric results and derivations."""
import argparse,hashlib,json,subprocess,zipfile
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--runs-root',type=Path,default=Path('/media/qy/6A7E3DB07E3D75C3/STTW_runs'));a=p.parse_args()
root=Path(__file__).resolve().parents[2];rev=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip();tracked=subprocess.check_output(['git','ls-files'],cwd=root,text=True).splitlines();files={};omitted=[]
evidence=['h75_theory_iteration','h75_evidence_queue_20261110','h75_mathematical_verification','h75_theory_atlas','h75_state_alpha_scan']
for name in tracked:
 path=root/name
 if not path.is_file():continue
 in_evidence=any(name.startswith('docs/evidence/'+d+'/') for d in evidence)
 source=any(name.startswith(d) for d in ['learning/src/sttw_control/','learning/cli/','learning/tests/','learning/configs/','learning/scripts/'])
 if (in_evidence and path.suffix in ['.md','.json','.csv','.npz','.py','.txt','.svg']) or (source and path.suffix in ['.py','.json','.sh','.toml','.txt']):files[name]=path
 elif in_evidence:omitted.append(dict(path=name,bytes=path.stat().st_size,reason='Derived graphic or duplicate format; main figures included with manuscript'))
for name in ['MANUSCRIPT.md','MANUSCRIPT.pdf','README.md','build.py','requirements.txt','evidence_manifest.json','export_receipt.json']:
 files['docs/paper/'+name]=root/'docs/paper'/name
paper=json.loads((root/'docs/paper/evidence_manifest.json').read_text())
for name in paper['inputs']:
 if '/figures/' in name:files[name]=root/name
# E1 was stored in another worktree; preserve its result directory as evidence.
e1=Path(paper['evidence']['E1']).parent
for f in e1.rglob('*'):
 if f.is_file() and f.suffix in ['.md','.json','.csv','.npz']:files['external/E1/'+str(f.relative_to(e1))]=f
run=a.runs_root/'h75_theory_iteration'
for stage in ['S1','S2']:
 for f in (run/stage).rglob('*'):
  if f.is_file() and f.suffix in ['.json','.csv','.npz']:files['external/theory/'+str(f.relative_to(run))]=f
counts={}
for stage in ['calibration','mechanism_scan','layout_test']:
 folder=run/stage
 for name in ['analysis_summary.json','protocol.json','jobs.json','status.json','completion_notice.json']:
  f=folder/name
  if f.is_file():files[f'external/theory/{stage}/{name}']=f
 jobs=json.loads((folder/'jobs.json').read_text());counts[stage]=dict(cases=len(jobs),episodes=len(json.loads((folder/'analysis_summary.json').read_text())['rows']))
 for job in jobs:
  f=folder/job['name']/'contract.json'
  if f.is_file():files[f'external/theory/{stage}/{job["name"]}/contract.json']=f
  cfg=Path(job['config']);files[f'external/theory/{stage}/configs/{cfg.name}']=cfg
# Original archive indexes, model identity and references remain in contracts.
for name in ['learning/cli/package_h75_research.py','docs/evidence/h75_research_package/README.md']:
 files[name]=root/name
records=[dict(path=name,source=str(f),bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest()) for name,f in sorted(files.items())]
manifest=dict(source_revision=rev,scope='H75 manuscript E1-E9, S0-S4 formulas, derivations, results, executable analysis source and compact numerical evidence',files=records,counts=counts,excluded=omitted,raw_simulation_root=str(a.runs_root),limitations='Full per-step simulation NPZ, 5kHz contacts, videos, checkpoints and external Isaac/CUDA SDK are not bundled. Numeric summaries and derived analysis arrays are included; contracts retain exact identity and raw locations. This is an evidence/analysis archive, not a fresh-machine GPU replay package.')
a.output.parent.mkdir(parents=True,exist_ok=True)
with zipfile.ZipFile(a.output,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
 for name,f in sorted(files.items()):z.write(f,name)
 z.writestr('MANIFEST.json',json.dumps(manifest,ensure_ascii=False,indent=2))
 z.writestr('README.md',(root/'docs/evidence/h75_research_package/README.md').read_text())
with zipfile.ZipFile(a.output) as z:
 assert z.testzip() is None
 for item in records:assert hashlib.sha256(z.read(item['path'])).hexdigest()==item['sha256']
receipt=dict(archive=a.output.name,bytes=a.output.stat().st_size,sha256=hashlib.sha256(a.output.read_bytes()).hexdigest(),files=len(records),uncompressed_bytes=sum(x['bytes'] for x in records),counts=counts,verified='ZIP CRC and every payload SHA256 match source',source_revision=rev)
a.output.with_suffix('.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
