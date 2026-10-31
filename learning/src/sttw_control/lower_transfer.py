"""Export an immutable lower candidate for explicit frozen-middle transfer tests."""
from pathlib import Path
import hashlib,io,json,pickle
import numpy as np


def select_scored_lower_checkpoint(run):
    """Select one saved continuation by its own next-rollout score and exposure."""
    import torch
    run=Path(run)
    if json.loads((run/'status.json').read_text())['state']!='completed':
        raise ValueError('selection requires completed training')
    rows=[json.loads(line) for line in (run/'training/metrics.jsonl').read_text().splitlines()]
    if len({r['update'] for r in rows})!=len(rows):raise ValueError('duplicate scoring updates')
    scores={r['update']-1:r for r in rows}
    candidates=[]
    for path in sorted((run/'training/checkpoints').glob('*.pt')):
        saved=torch.load(path,map_location='cpu',weights_only=False)
        update=saved['update'];row=scores.get(update)
        reasons=[]
        if update==0:reasons.append('parent_weights')
        if row is None:reasons.append('no_next_rollout_score')
        else:
            groups=row['command_sampling']['disturbance_groups']
            exposure={direction:sum(v['exposed_ticks'] for k,v in groups.items() if k.startswith(direction+'_')) for direction in ('negative','positive')}
            nominal=sum(v['samples'] for k,v in groups.items() if k.startswith('nominal_'))
            if not all(exposure.values()) or not nominal:reasons.append('missing_actual_direction_or_nominal_coverage')
            if not np.isfinite(row['mean_step_reward']):reasons.append('nonfinite_reward')
        candidates.append(dict(checkpoint=str(path.resolve()),source_update=update,eligible=not reasons,reasons=reasons,
            scored_by_rollout=row['update'] if row else None,mean_step_reward=row['mean_step_reward'] if row else None,
            disturbance_groups=row['command_sampling']['disturbance_groups'] if row else None))
    eligible=[c for c in candidates if c['eligible']]
    selected=max(eligible,key=lambda c:(c['mean_step_reward'],-c['source_update'])) if eligible else None
    return dict(selected=selected,candidates=candidates,selection='Best eligible saved continuation by its own pre-update rollout reward; not fixed-scene qualification.',
        unrestricted_best=json.loads((run/'training/train_best_candidate.json').read_text()))


def export_lower_transfer(checkpoint,base,output):
    import torch
    data=Path(checkpoint).read_bytes();saved=torch.load(io.BytesIO(data),map_location='cpu',weights_only=False)
    cfg=saved['config'];parent=cfg['weights_only_parent']
    if parent['checkpoint_sha256']!=base['checkpoint_sha256']:
        raise ValueError('candidate is not a continuation of the declared frozen lower')
    base_bytes=Path(base['config']).read_bytes()
    if hashlib.sha256(base_bytes).hexdigest()!=base['config_file_sha256']:
        raise ValueError('parent lower config changed')
    previous=json.loads(base_bytes)
    fields=('actor_dim','critic_dim','hidden_sizes','history_frames','frame_fields','observation_scales','observation_clip_abs','activation','dt','limits')
    if any(cfg[k]!=previous[k] for k in fields):raise ValueError('lower observation, action or timing contract differs')
    if cfg['actor_dim']!=210 or cfg['hidden_sizes']!=[256,128] or cfg['activation']!='elu':
        raise ValueError('export requires the existing 210D frozen lower interface')
    weights=saved['policy'];expected={f'actor.{i}.{k}' for i in (0,2,4) for k in ('weight','bias')}
    if {k for k in weights if k.startswith('actor.')}!=expected:raise ValueError('actor parameter structure differs')
    params={}
    for layer,(index,shape) in enumerate(zip((0,2,4),((256,210),(128,256),(2,128)))):
        weight=weights[f'actor.{index}.weight'].detach().cpu().numpy();bias=weights[f'actor.{index}.bias'].detach().cpu().numpy()
        if weight.shape!=shape or bias.shape!=(shape[0],) or not np.isfinite(weight).all() or not np.isfinite(bias).all():
            raise ValueError('actor tensor shape/nonfinite mismatch')
        params[f'Dense_{layer}']=dict(kernel=weight.T,bias=bias)
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    (out/'checkpoint.pt').write_bytes(data)
    (out/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    with (out/'actor.pkl').open('wb') as f:pickle.dump({'params':params},f)
    candidate={k:base[k] for k in ('normalizer',)}
    candidate.update(role='explicit_lower_transfer_candidate',status='not_adopted',source_update=saved['update'])
    for key,name,hash_key in [('actor','actor.pkl','actor_sha256'),('checkpoint','checkpoint.pt','checkpoint_sha256'),('config','config.json','config_file_sha256')]:
        path=out/name;candidate[key]=str(path);candidate[hash_key]=hashlib.sha256(path.read_bytes()).hexdigest()
    record=dict(candidate=candidate,parent_checkpoint_sha256=base['checkpoint_sha256'],source_checkpoint=str(Path(checkpoint).resolve()),
                semantics='Frozen middle zero-shot transfer; original model, controller, actuator and prepared bank preserved. Candidate is not adopted.',
                optimizer_updates=0,physical_steps=0)
    (out/'transfer.json').write_text(json.dumps(record,indent=2)+'\n')
    return record
