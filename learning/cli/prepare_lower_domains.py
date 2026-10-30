"""Generate a sealed finite bank of isolated physical domains; no simulation."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from sttw_control.model import export_domain_model,DEFAULT_XML


def prepare(config,output):
    cfg=json.loads(Path(config).read_text());out=Path(output)
    bounds=np.array([cfg['wheel_sliding_friction'],cfg['chassis_mass_scale']],float)
    if bounds.shape!=(2,2) or not np.isfinite(bounds).all() or np.any(bounds[:,0]<=0) or np.any(bounds[:,0]>=bounds[:,1]):
        raise ValueError('two positive ordered physical-domain ranges required')
    if len(set(cfg['seeds'].values()))!=2 or set(cfg['seeds'])!={'TRAIN','TEST'} or any(not isinstance(cfg['counts'][r],int) or cfg['counts'][r]<1 for r in ('TRAIN','TEST')):
        raise ValueError('distinct TRAIN/TEST seeds and positive counts required')
    out.mkdir(parents=True,exist_ok=False);records=[]
    parameters=dict(wheel_sliding_friction=.8,chassis_mass_scale=1.)
    records.append(dict(domain_id='nominal',data_role='REFERENCE',**export_domain_model(out/'nominal.xml',parameters)))
    for role in ('TRAIN','TEST'):
        rng=np.random.default_rng(cfg['seeds'][role]);count=cfg['counts'][role]
        samples=np.column_stack([(rng.permutation(count)+rng.uniform(size=count))/count for _ in range(2)])
        values=bounds[:,0]+samples*(bounds[:,1]-bounds[:,0])
        for i,(friction,mass) in enumerate(values):
            name=f'{role.lower()}_{i:02d}'
            parameters=dict(wheel_sliding_friction=float(friction),chassis_mass_scale=float(mass))
            records.append(dict(domain_id=name,data_role=role,seed=cfg['seeds'][role],**export_domain_model(out/f'{name}.xml',parameters)))
    manifest=dict(config=cfg,source_xml=str(DEFAULT_XML),domains=records,physical_steps=0,optimizer_updates=0,
        status='models_compiled_not_simulated',scope='finite stratified physical-domain bank, not yet episode-wise randomized training')
    file=out/'manifest.json';file.write_text(json.dumps(manifest,indent=2)+'\n')
    (out/'seal.json').write_text(json.dumps(dict(manifest_sha256=hashlib.sha256(file.read_bytes()).hexdigest()),indent=2)+'\n')
    print(json.dumps(dict(domains=len(records),physical_steps=0,output=str(out.resolve()))))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();prepare(a.config,a.output)
