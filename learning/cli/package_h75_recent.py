"""Compact, hash-indexed delivery of frozen H75 continuation inputs and evidence."""
import argparse,hashlib,json,shutil,subprocess,os
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();root=Path.cwd();out=a.output;out.mkdir(parents=True,exist_ok=True);records=[]
def copy(src,dst):
 src=Path(src);target=out/dst;target.parent.mkdir(parents=True,exist_ok=True)
 if target.is_symlink():target.unlink()
 shutil.copy2(src,target);records.append(dict(path=str(dst),source=str(src.resolve()),bytes=target.stat().st_size,sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
def tree(src,dst,suffixes=None):
 for f in sorted(Path(src).rglob('*')):
  if f.is_file() and '__pycache__' not in f.parts and (suffixes is None or f.suffix in suffixes):copy(f,Path(dst)/f.relative_to(src))
s=json.loads((a.source/'common_start/evaluation_no_push/config.json').read_text())
for key in ['bundle','training_bank']:
 tree(Path(s[key]),Path('inputs')/key,{'.json','.npz','.pt'})
for key in ['initial_state','controller_config','lower_actor']:
 copy(s[key],Path('inputs')/key/Path(s[key]).name)
asset=Path(s['asset']);tree(asset.parent,'inputs/asset',{'.urdf','.obj','.json'})
tree(a.source/'runtime','inputs/runtime',{'.py','.md','.txt'})
tree(a.source/'train_assets','inputs/train_assets',{'.npz','.json'});tree(a.source/'test_assets','inputs/test_assets',{'.npz','.json'})
copy(a.source/'common_start/training_resume/checkpoint.pt','models/common_start_update150.pt')
copy(a.source/'training_resume/checkpoint.pt','models/deprecated_midroute_update76.pt')
for rel in ['training','training_resume','common_start/training','common_start/training_resume']:
 for name in ['config.json','history.json','status.json']:
  f=a.source/rel/name
  if f.exists():copy(f,Path('training_records')/rel/name)
for rel in ['analysis','common_start/evaluation','common_start/evaluation_no_push']:
 src=a.source/rel
 for f in sorted(src.rglob('*')):
  if f.is_file() and 'input' not in f.relative_to(src).parts and f.suffix in {'.md','.png','.json','.csv'}:copy(f,Path('results')/rel/f.relative_to(src))
 for f in src.glob('*/trajectory.npz'):copy(f,Path('results')/rel/f.relative_to(src))
 for f in src.glob('*/reference_routes.npz'):copy(f,Path('results')/rel/f.relative_to(src))
 for f in src.glob('*/initial_state.npz'):copy(f,Path('results')/rel/f.relative_to(src))
dense=a.source.parent/'h75_dense_corridor_ridge10/evaluation'
for f in sorted(dense.iterdir()):
 if f.is_file() and f.suffix in {'.md','.json','.csv','.png'}:copy(f,Path('results/dense_corridor_ridge10')/f.name)
tree(dense/'alpha_comparison','results/dense_corridor_ridge10/alpha_comparison',{'.png','.md','.json','.npz'})
# Relocatable evaluation config: launch from repository root, preserve original identities.
for key in ['bundle','training_bank']:s[key]=str(out/'inputs'/key)
for key in ['initial_state','controller_config','lower_actor']:s[key]=str(out/'inputs'/key/Path(s[key]).name)
s['asset']=str(out/'inputs/asset'/asset.name);s['initial_policy']=str(out/'inputs/training_bank/original_policy.pt');s['rsl_runtime']=str(out/'inputs/runtime');s.pop('go2_checkout',None)
s['terrains']={k:('plane' if v=='plane' else str(out/'inputs/test_assets'/Path(v).name)) for k,v in s['terrains'].items()}
(out/'evaluate_no_push.json').write_text(json.dumps(s,indent=2)+'\n')
seen={}
for row in records:
 path=out/row['path'];digest=row['sha256']
 if digest in seen and row['bytes']>10000:
  path.unlink();path.symlink_to(os.path.relpath(seen[digest],path.parent));row['deduplicated_to']=str(seen[digest].relative_to(out))
 else:seen[digest]=path
revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
(out/'manifest.json').write_text(json.dumps(dict(source_revision=revision,files=records,total_bytes=sum(x['bytes'] for x in records),external_dependencies=['Isaac Sim 5.1 / compatible Isaac Lab','CUDA GPU and driver','PyTorch, tensordict, tensorboard, numpy','MuJoCo and ffmpeg only for saved-pose video rendering']),indent=2)+'\n')
print(len(records),'files',sum(x['bytes'] for x in records),'bytes')
