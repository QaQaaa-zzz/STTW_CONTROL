"""Validate and reuse already-paid fixed zero-upper trajectories."""
from pathlib import Path
import json,shutil
import numpy as np
from .preference_best import config_hash,digest,atomic_json
from .path_command_selection import cases

def reuse_zero_upper(parent,root,config,source,expected_paths):
    parent=Path(parent);root=Path(root);manifest=json.loads((parent/'manifest.json').read_text())
    if config_hash(manifest['config'])!=config_hash(config) or config_hash(manifest['source'])!=config_hash(source):
        raise ValueError('baseline config/model/bank/timing identity mismatch')
    out=root/'zero_upper';out.mkdir(parents=True,exist_ok=True);receipt={}
    count=round(config['training_future_phase_C']['episode_s']/config['timing']['lower_dt_s'])
    prior_receipt=parent/'zero_upper_reuse.json'
    prior=json.loads(prior_receipt.read_text()).get('cases',{}) if prior_receipt.exists() else {}
    for name,route,speed in cases(config):
        file=parent/'zero_upper'/f'{name}.npz';pathfile=file.with_name(name+'_path.npz')
        if not file.exists() or not pathfile.exists():continue
        if name in prior and (digest(file)!=prior[name]['trace_sha256'] or digest(pathfile)!=prior[name]['path_sha256']):raise ValueError('baseline cache SHA changed '+name)
        d=dict(np.load(file));p=dict(np.load(pathfile));expected=expected_paths[route]
        for key in ['s','xy','heading','curvature','goal','turn_end']:
            np.testing.assert_allclose(p[key],np.asarray(getattr(expected,key)),rtol=0,atol=2e-6,err_msg='cached path changed '+name)
        if len(d['time'])!=count:raise ValueError('incomplete baseline '+name)
        np.testing.assert_allclose(d['time'],np.arange(count)*config['timing']['lower_dt_s'],atol=2e-6,rtol=0)
        np.testing.assert_allclose(d['v_user'],speed,atol=1e-6,rtol=0)
        for key in ['offsets','target_offset','filtered_offset']:np.testing.assert_array_equal(d[key],np.zeros_like(d[key]))
        if any(np.any(d[k]) for k in ['physical_failure','domain_exit','policy_fault','lower_fault']):raise ValueError('baseline requires terminal-contract audit '+name)
        if any(not np.isfinite(a).all() for a in d.values() if np.issubdtype(a.dtype,np.number)):raise ValueError('nonfinite cached baseline')
        for src in [file,pathfile]:shutil.copyfile(src,out/src.name)
        receipt[name]={'trace_sha256':digest(file),'path_sha256':digest(pathfile),'source':str(file.resolve()),'steps':count}
    atomic_json(root/'zero_upper_reuse.json',dict(parent=str(parent.resolve()),config_sha256=config_hash(config),source_sha256=config_hash(source),cases=receipt))
    return set(receipt)


def reuse_final_panel(parent,out,alpha,config,source,best,expected_paths):
    """Reuse only complete traces attested at the deliberate saved boundary."""
    parent=Path(parent);out=Path(out)
    receipt_file=parent/'optimization_boundary_interrupt.json'
    if not receipt_file.exists():return set()
    receipt=json.loads(receipt_file.read_text())
    old=receipt['final_panel_source']
    if old['alpha']!=alpha:return set()
    manifest=json.loads((parent/'manifest.json').read_text())
    for key,value in [('config',config),('source',source)]:
        if config_hash(manifest[key])!=config_hash(value):raise ValueError('final cache identity mismatch')
    for key in ['actor_sha256','checkpoint_sha256','source_update','schema','alpha']:
        if old[key]!=best[key]:raise ValueError('final cache selected model mismatch '+key)
    for key in ['actor','checkpoint']:
        if digest(best[key])!=best[key+'_sha256']:raise ValueError('selected weight changed')
    attested={Path(p).name:sha for p,sha in receipt['complete_final_files'].items()}
    count=round(config['training_future_phase_C']['episode_s']/config['timing']['lower_dt_s'])
    reused={}
    for name,route,speed in cases(config):
        files=[parent/f'alpha{alpha}'/'final_best'/f'{name}{suffix}.npz' for suffix in ['', '_path']]
        if not all(p.name in attested for p in files):continue
        if any(digest(p)!=attested[p.name] for p in files):raise ValueError('attested final trace changed')
        with np.load(files[0]) as archive:d={k:archive[k] for k in archive.files}
        with np.load(files[1]) as archive:p={k:archive[k] for k in archive.files}
        for key in ['s','xy','heading','curvature','goal','turn_end']:
            np.testing.assert_allclose(p[key],np.asarray(getattr(expected_paths[route],key)),rtol=0,atol=2e-6)
        size=len(d['time'])
        if not 0<size<=count:raise ValueError('invalid cached length')
        if size<count and not (d['physical_failure'][-1] or d['domain_exit'][-1]):raise ValueError('unfinished final case')
        if any(np.any(d[k]) for k in ['policy_fault','lower_fault']):raise ValueError('engineering failure in cache')
        if any(not np.isfinite(a).all() for a in d.values() if np.issubdtype(a.dtype,np.number)):raise ValueError('nonfinite cache')
        np.testing.assert_allclose(d['time'],np.arange(size)*config['timing']['lower_dt_s'],rtol=0,atol=2e-6)
        np.testing.assert_allclose(d['v_user'],speed,rtol=0,atol=1e-6)
        target=out/'final_best';target.mkdir(exist_ok=True)
        for file in files:shutil.copyfile(file,target/file.name)
        reused[name]={p.name:attested[p.name] for p in files}
    atomic_json(out/'final_panel_reuse.json',dict(parent=str(parent.resolve()),source=old,attestation=str(receipt_file.resolve()),cases=reused))
    return set(reused)


def inherit_numerical_history(parent,out,alpha):
    """Keep parent DEV and statistics available without any physical replay."""
    previous=Path(parent)/f'alpha{alpha}';out=Path(out);copied=[]
    for pattern in ['validation/update_*/*metrics.json','statistics/update_*.json']:
        for src in previous.glob(pattern):
            dst=out/'inherited_history'/src.relative_to(previous)
            if dst.exists():continue
            dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)
            copied.append(str(src.resolve()))
    atomic_json(out/'inherited_numerical_history.json',dict(sources=copied))
