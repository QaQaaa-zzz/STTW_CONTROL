"""Compact Chapter 3 code, frozen models and current evidence delivery."""
import argparse,hashlib,json,subprocess,zipfile
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
repo=Path(__file__).resolve().parents[2];runs=Path('/media/qy/6A7E3DB07E3D75C3/STTW_runs');files={}
def add(path,name):
 path=Path(path)
 if not path.is_file():raise FileNotFoundError(path)
 files[name]=path

def directory(path,name,suffixes=None):
 for f in sorted(Path(path).rglob('*')):
  if f.is_file() and '__pycache__' not in f.parts and (suffixes is None or f.suffix in suffixes):add(f,name+'/'+str(f.relative_to(path)))

tracked=subprocess.check_output(['git','ls-files'],cwd=repo,text=True).splitlines()
for name in tracked:
 if name.startswith('learning/') and Path(name).suffix in {'.py','.json','.sh','.toml','.txt'}:add(repo/name,'code/'+name)
add(Path(__file__),'code/learning/cli/package_chapter3_progress.py')
add(repo/'docs/CHAPTER3_EVENT_WORLD_MODEL.md','progress/CHAPTER3_EVENT_WORLD_MODEL.md')
add(repo/'docs/evidence/chapter3_delivery/README.md','README.md')
add(repo/'docs/evidence/chapter3_delivery/model_load_check.json','model_load_check.json')
add('/home/qy/STTW_CONTROL/research-hub/CHAPTER3_PROJECT_STATE.md','progress/CHAPTER3_PROJECT_STATE_snapshot.md')
for name in ['chapter3_review_repairs','chapter3_repaired_joint_fast','chapter3_rb_candidate_handoff','chapter3_rb_feedback','chapter3_interface_diagnosis','chapter3_playground_full_chain','chapter3_playground_failure_videos']:
 directory(repo/'docs/evidence'/name,'evidence/'+name)
for name in ['chapter3_time_residual_online','chapter3_playground_full_chain','chapter3_playground_failure_videos']:
 folder=runs/name
 for f in folder.iterdir():
  if f.is_file() and f.suffix in {'.md','.json','.png','.mp4','.mjb','.xml','.npz'}:add(f,'evidence/'+name+'/'+f.name)
 for f in folder.rglob('config.json'):add(f,'inputs/'+name+'/'+str(f.relative_to(folder)))
# Frozen upstream/downstream predictors and the direct/state-only baselines.
for name in ['chapter3_compact_interface','chapter3_compact_time_residual/fit','chapter3_frenet_condition_model']:
 folder=runs/name
 for f in folder.iterdir():
  if f.is_file() and f.suffix in {'.pt','.json'}:add(f,'models/'+name+'/'+f.name)
# Actual H75 bundle used by the Chapter 3 handoff, not the separately trained H75 variants.
directory(repo/'runs/rb_h75_ecbc_ground/left90_R2/bundle','models/h75', {'.npz','.json'})
add(repo/'runs/rb_h75_ecbc_ground/left90_R2/config.json','inputs/h75_settings.json')
add(repo/'learning/configs/rb_sttw_rounded_controller.json','inputs/rb_sttw_rounded_controller.json')
add(repo/'runs/local300_torch_portable_parity/actor.npz','models/local300/actor.npz')
# Preserve full native checkpoints, normalizers, identities and training/runtime provenance.
registry=json.loads((repo/'learning/configs/playground_rb_skill_sources.json').read_text())
for name,record in registry['records'].items():
 policy=record['policy'];directory(Path(policy['checkpoint']),'models/native_rb/'+name)
 cfg=Path(policy['formal_config']);add(cfg,'models/native_rb/'+name+'/formal_config.json')
 raw=json.loads(cfg.read_text());up=Path(raw['inputs']['up_config_path']);add(up,'models/native_rb/'+name+'/up_config.json')
 expected=policy['payload_sha256'];actual=hashlib.sha256((Path(policy['checkpoint'])/'payload.pkl').read_bytes()).hexdigest()
 if actual!=expected:raise ValueError('Native checkpoint mismatch: '+name)
# Isaac RB candidates are distinct from the three original native RB runs.
for name in ['source','update301','update400','update500']:
 cfg=runs/'chapter3_rb_candidate_handoff'/name/'config.json';raw=json.loads(cfg.read_text());add(cfg,'inputs/isaac_candidates/'+name+'.json')
 checkpoint=raw['evaluation_checkpoint']
 if checkpoint:add(Path(checkpoint),'models/isaac_rb/'+name+'/'+Path(checkpoint).name)
 else:directory(repo/raw['bundle'],'models/isaac_rb/source',{'.npz','.json'})
for name in ['roads.npz','issued.npz']:add(runs/'chapter3_task_condition_collection'/name,'inputs/routes/'+name)
add(runs/'chapter3_long_bend_goal/task.json','inputs/task.json')
# Native runtime source is read-only copied; no changes to DVGC.
directory(Path('/home/qy/DVGC/JIT/src/jit_dvgc'),'third_party/jit_dvgc',{'.py'})
model=Path(registry['model']['path']);add(model,'assets/native/'+model.name)
import xml.etree.ElementTree as ET
for item in ET.parse(model).getroot().findall('.//asset/mesh'):
 f=model.parent/'meshes'/item.attrib['file'];add(f,'assets/native/meshes/'+item.attrib['file'])
records=[dict(path=name,source=str(f.resolve()),bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest()) for name,f in sorted(files.items())]
manifest=dict(source_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),files=records,scope='Chapter3 code, native and Isaac frozen skill candidates, interface predictors, compact actual results and input mapping',limitations='No Isaac/CUDA/MuJoCo SDK or full training datasets/optimizer history. Original absolute paths retained for provenance; source-to-archive mapping is explicit. Not a validated new-machine GPU reproduction. Native and Isaac RB weights are separate families.')
a.output.parent.mkdir(parents=True,exist_ok=True)
with zipfile.ZipFile(a.output,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for name,f in sorted(files.items()):z.write(f,name)
 z.writestr('MANIFEST.json',json.dumps(manifest,ensure_ascii=False,indent=2))
with zipfile.ZipFile(a.output) as z:
 assert z.testzip() is None
 for item in records:assert hashlib.sha256(z.read(item['path'])).hexdigest()==item['sha256']
receipt=dict(archive=a.output.name,bytes=a.output.stat().st_size,sha256=hashlib.sha256(a.output.read_bytes()).hexdigest(),files=len(records),model_files=[x for x in records if x['path'].startswith('models/')],validation='ZIP CRC and every file SHA256; native checkpoint hashes match original registry',source_revision=manifest['source_revision'])
a.output.with_suffix('.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:v for k,v in receipt.items() if k!='model_files'},indent=2))
