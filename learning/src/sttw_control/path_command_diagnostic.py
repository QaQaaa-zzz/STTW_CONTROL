"""Explicit fixed-Actor causal probes. Never imported by training or selection."""
import jax
import jax.numpy as j
import numpy as np
from .path_command_policy import map_latent


def transform_latent(z,config,*,chi,time_s,spec):
    c=config['residual_interface'];target=map_latent(z,config)
    kind=spec.get('kind','scaled_target')
    beta=spec.get('beta',1.)
    backoff=spec.get('speed_backoff',0.)
    if kind=='late_fade':
        beta=j.clip(1-(time_s-spec['start_s'])/spec['duration_s'],0.,1.)
    elif kind!='scaled_target':raise ValueError('unknown diagnostic transform')
    speed_requested=target[0]-backoff*chi
    speed=j.clip(speed_requested,-c['speed_negative_scale_m_s'],c['speed_positive_scale_m_s'])
    steer=beta*target[1]
    scale=j.where(speed>=0,c['speed_positive_scale_m_s'],c['speed_negative_scale_m_s'])
    speed_z=z[0] if backoff==0 else j.arctanh(j.clip(speed/scale,-1+1e-6,1-1e-6))
    steer_z=j.where(beta==1,z[1],j.arctanh(j.clip(steer/c['steer_scale_rad'],-1+1e-6,1-1e-6)))
    applied=j.stack([speed_z,steer_z])
    # Preserve exact latent, including original nonfinite faults, on the no-op path.
    if kind=='scaled_target' and beta==1 and backoff==0:applied=z
    return applied,dict(original_latent=z,applied_latent=applied,original_target=target,
        applied_target=map_latent(applied,config),requested_target=j.stack([speed_requested,steer]),
        target_clip_count=j.asarray(speed!=speed_requested,j.int32))


def current_demand(state,path,speed,env):
    from .geometric_path import at
    from .direct_command_policy import q_ratio
    _,_,k=at(path,state.projection.progress)
    ff=j.arctan(env.cc.wheelbase*k/j.cos(env.cc.caster))
    return j.clip((j.abs(ff/q_ratio(speed,env.cc))-.18)/.12,0,1)


def make_transform(config,spec):
    def apply(z,state,path,speed,env):
        return transform_latent(z,config,chi=current_demand(state,path,speed,env),
            time_s=state.tick*config['timing']['lower_dt_s'],spec=spec)
    return apply


def compare_prefix(original,diagnostic,until_s=10.,atol=2e-6):
    """Check all original logged channels; real faults and endpoints stay visible."""
    masks=[np.asarray(d['time'])<until_s for d in (original,diagnostic)]
    if masks[0].sum()!=masks[1].sum():raise AssertionError('diagnostic prefix length differs')
    for key,value in original.items():
        if key not in diagnostic:raise AssertionError('missing original prefix channel '+key)
        a=np.asarray(value)[masks[0]];b=np.asarray(diagnostic[key])[masks[1]]
        if a.dtype.kind in 'bui':np.testing.assert_array_equal(a,b,err_msg=key)
        else:np.testing.assert_allclose(a,b,rtol=0,atol=atol,equal_nan=True,err_msg=key)
    return dict(compared_lower_ticks=int(masks[0].sum()),until_s=until_s,absolute_tolerance=atol)


def prepare_request(request):
    """Read-only audit of frozen source and all declared panel coordinates."""
    import json
    from pathlib import Path
    from .preference_best import digest,config_hash
    from .path_command_selection import cases
    if request['budget_lower_ticks']!=32000 or request['training_updates']!=0 or len(request['cases'])!=8:
        raise ValueError('exact eight-case 32000-tick diagnostic required')
    expected={(model,case,config_hash(spec)) for model,case,spec in
        [('parent1',case,dict(beta=beta,speed_backoff=backoff)) for beta,backoff in [(.75,0),(.5,0),(.5,.05)] for case in ['left90_R2_2.6','right90_R2_2.6']]+
        [('u200',case,dict(kind='late_fade',start_s=10.,duration_s=.5)) for case in ['straight_2.0','straight_2.6']]}
    actual={(r['model'],r['case'],config_hash(r['transform'])) for r in request['cases']}
    if actual!=expected or len({r['id'] for r in request['cases']})!=8:raise ValueError('diagnostic intervention declaration differs')
    manifest_path=Path(request['manifest']);manifest=json.loads(manifest_path.read_text());cfg=manifest['config'];source=manifest['source']
    if manifest['terminal_contract']!='terminal_contract_v2':raise ValueError('terminal contract changed')
    if cfg['training_future_phase_C']['episode_s']!=20.:raise ValueError('diagnostic horizon changed')
    files={str(manifest_path):digest(manifest_path)}
    def audit(path,sha):
        if digest(path)!=sha:raise ValueError('frozen SHA mismatch '+str(path))
        files[str(path)]=sha
    frozen_path=Path(request['frozen_identities']);frozen=json.loads(frozen_path.read_text())
    files[str(frozen_path)]=digest(frozen_path)
    if frozen['source']!=source:raise ValueError('frozen prepared/lower source changed')
    for label,role in request['frozen_roles'].items():
        anchor=frozen['records'][role];record=request['models'][label]
        for key in ('actor','checkpoint','actor_sha256','checkpoint_sha256'):
            if record[key]!=anchor[key]:raise ValueError('frozen model anchor changed '+label)
    for label,path in request['model_manifests'].items():
        files[str(path)]=digest(path)
        other=json.loads(Path(path).read_text())
        if other['source']!=source or other['terminal_contract']!='terminal_contract_v2':raise ValueError('diagnostic prepared-state identity mismatch')
    for record in request['models'].values():
        for key in ['checkpoint','actor']:audit(record[key],record[key+'_sha256'])
    for key in ['actor','checkpoint']:audit(source['candidate'][key],source['candidate'][key+'_sha256'])
    audit(source['candidate']['config'],source['candidate']['config_file_sha256']);audit(source['prepared_bank'],source['prepared_bank_sha256'])
    if source['candidate']['source_update']!=300:raise ValueError('local300 changed')
    for label,folder in request['cached_panels'].items():
        folder=Path(folder)
        for name,_,speed in cases(cfg):
            for suffix in ['', '_path']:
                path=folder/(name+suffix+'.npz');files[str(path)]=digest(path)
                with np.load(path,allow_pickle=False) as z:
                    if suffix=='':
                        if not len(z['time']) or len(z['time'])>4000:raise ValueError('invalid cached endpoint')
                        np.testing.assert_allclose(z['v_user'],speed,rtol=0,atol=1e-6)
                    elif not {'s','xy','heading','curvature','goal','turn_end'}<=set(z.files):raise ValueError('incomplete original path')
    return dict(schema=request['schema'],state='prepared_not_run',physical_ticks=0,training_updates=0,
        budget_lower_ticks=32000,prepared_state=dict(path=source['prepared_bank'],sha256=source['prepared_bank_sha256'],bank_index=source['bank_index'],restoration='complete load_environment snapshot'),
        files=files,request=request,manifest=manifest,manifest_sha256=digest(manifest_path))


def diagnostic_groups(request):
    """Pair same-shape fixed cases per Actor/intervention; no redundant panels."""
    from .preference_best import config_hash
    groups={}
    for row in request['cases']:
        groups.setdefault((row['model'],config_hash(row['transform'])),[]).append(row)
    return list(groups.values())
