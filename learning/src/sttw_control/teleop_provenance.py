"""One source/config/model identity per run; no plot hash sweeps."""
import hashlib,json,platform,subprocess
from pathlib import Path
import jax,mujoco,numpy

def provenance(root,config,env):
    root=Path(root);path=root/'source_manifest.json'
    identity=dict(base_commit='b446abae131f7a3698c884f1872f8049e9ba3c6e',
        commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        dirty=bool(subprocess.check_output(['git','status','--porcelain'],text=True).strip()),
        config_sha256=hashlib.sha256(Path(config).read_bytes()).hexdigest(),
        xml_sha256=env.bundle.identity['source_xml_sha256'],model_identity=env.bundle.identity,
        dependencies=dict(jax=jax.__version__,mujoco=mujoco.__version__,numpy=numpy.__version__),
        devices=[str(d) for d in jax.devices()],hardware=platform.platform(),
        backend='mjx jax',observation_mode=env.observation_mode)
    if path.exists():
        old=json.loads(path.read_text())
        for k in ['config_sha256','xml_sha256','dependencies']:
            if old[k]!=identity[k]:raise ValueError('run identity mismatch: '+k)
        return old
    path.write_text(json.dumps(identity,indent=2)+'\n');return identity
