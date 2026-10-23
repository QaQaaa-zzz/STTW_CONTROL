"""Authenticate completed matched-policy panels before full-tail comparison."""
from pathlib import Path
from itertools import product
import json
from .preference_best import digest, config_hash


def cached_policy_panel(run, protocol, record, source_policy, output_policy):
    run=Path(run)
    status=json.loads((run/'status.json').read_text())
    manifest=json.loads((run/'manifest.json').read_text())
    saved=manifest['declaration']
    rows=json.loads((run/'results.json').read_text())['cases']
    if status['phase']!='completed' or len(rows)!=status['completed_cases'] or len(rows)!=status['declared_cases']:
        raise ValueError('cached panel incomplete')
    if len({r['name'] for r in rows})!=len(rows):
        raise ValueError('duplicate cached case')
    for key in ('config','source','routes','speeds','alphas','forces_N','onsets_s','duration_s'):
        if config_hash(saved[key])!=config_hash(protocol[key]):
            raise ValueError('cached protocol differs: '+key)
    if saved.get('recovery_intervention')!=protocol.get('recovery_intervention'):
        raise ValueError('cached control intervention differs')
    for kind in ('actor','checkpoint'):
        if saved['policies'][source_policy][kind+'_sha256']!=record[kind+'_sha256'] or digest(record[kind])!=record[kind+'_sha256']:
            raise ValueError('cached model identity differs')
    selected=[r for r in rows if r['policy']==source_policy]
    keys=('route','speed','alpha','force_N','onset_s')
    expected=set(product([r['name'] for r in protocol['routes']],protocol['speeds'],
                         protocol['alphas'],protocol['forces_N'],protocol['onsets_s']))
    if len(selected)!=len(expected) or {tuple(r[k] for k in keys) for r in selected}!=expected:
        raise ValueError('cached contexts incomplete or duplicated')
    if any(not (run/(r['name']+'.npz')).is_file() for r in selected):
        raise ValueError('cached raw trajectory missing')
    return dict(rows=[dict(r,policy=output_policy) for r in selected],
                receipt=dict(run=str(run.resolve()),source_policy=source_policy,output_policy=output_policy,
                    cases=len(selected),manifest_sha256=digest(run/'manifest.json'),
                    results_sha256=digest(run/'results.json'),record=record,
                    interpretation='authenticated reuse; not a new rollout or independent replicate'))
