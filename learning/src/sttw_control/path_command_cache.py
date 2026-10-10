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
    for name,route,speed in cases(config):
        file=parent/'zero_upper'/f'{name}.npz';pathfile=file.with_name(name+'_path.npz')
        if not file.exists() or not pathfile.exists():continue
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
