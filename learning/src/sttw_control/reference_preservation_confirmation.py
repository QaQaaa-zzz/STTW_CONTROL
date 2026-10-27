"""Fixed new-time confirmation: authenticated sharing, no event-conditioned choice."""
import json
import os
import shutil
import time
import traceback
from pathlib import Path
import numpy as np
from .preference_best import atomic_json,digest

METHODS={'full','context_choice','geometry_alpha','alpha_only'}


def audit_pulse(raw,case):
    """Authenticate actual force/time with the existing1% integration allowance.

    Truncated exposure is reported, never relabeled unknown merely for falling.
    """
    a=np.asarray(raw['active_tick'],bool);clock=np.asarray(raw['physical_tick'])[a]*.005
    start=case['pulse_onset_task_tick']*.005;end=start+case['pulse_duration_s']
    overlap=np.maximum(0,np.minimum(clock+.005,end)-np.maximum(clock,start))
    force=case['force_N']*np.asarray(case['world_direction'])
    expected=overlap[:,None]*force;actual=np.asarray(raw['applied_force_impulse'])[a]
    if not np.isfinite(actual).all():raise ValueError('nonfinite physical pulse log')
    error=float(np.abs(actual-expected).sum());scale=float(np.abs(expected).sum())
    if error>max(1e-5,.01*scale):raise ValueError('actual force/time profile differs from frozen event')
    return dict(actual_world_impulse=actual.sum(axis=0).tolist(),expected_observed_world_impulse=expected.sum(axis=0).tolist(),profile_absolute_error=error,profile_relative_error=error/scale if scale else None)


def validate_inventory(logical,physical):
    if len(logical)!=384 or len(physical)!=240:raise ValueError('fixed384/240 only')
    clean=lambda c:{k:v for k,v in c.items() if k not in ('case_id','role')}
    identities={json.dumps(dict(case=clean(p['case']),vector=p['vector']),sort_keys=True) for p in physical}
    if len(identities)!=240:raise ValueError('physical inventory includes duplicated simulation')
    used=set();vectors={};events={}
    for row in logical:
        c=row['case'];index=row['physical_index'];p=physical[index]
        if c['role'] not in METHODS:raise ValueError('unknown method')
        v=np.asarray(row['vector'])
        if v.shape!=(6,) or not np.isfinite(v).all() or np.any(abs(v)>1):raise ValueError('invalid reference vector')
        if row['vector']!=p['vector']:raise ValueError('shared vector mismatch')
        if clean(c)!=clean(p['case']):raise ValueError('shared full case semantics mismatch')
        key=(c['context'],c['alpha'],c['role'])
        if key in vectors and vectors[key]!=row['vector']:raise ValueError('future event changed chosen vector')
        vectors[key]=row['vector'];events.setdefault(key,set()).add((c['pulse_onset_task_tick'],c['force_N']))
        used.add(index)
    if len(vectors)!=48 or any(len(e)!=8 for e in events.values()) or used!=set(range(240)):
        raise ValueError('alltask/method/event/shared identities required')
    return True


def read_inputs(parent):
    parent=Path(parent).resolve();out=parent/'preservation_timecheck_v1'
    seal=json.loads((out/'seal.json').read_text())
    for f,sha in seal.items():
        if digest(out/f)!=sha:raise ValueError('declared protocol or inventory changed')
    p=json.loads((out/'declared_protocol.json').read_text())
    for file,key in [(parent/'plan.json','source_parent_plan_sha256'),
                     (parent/'production/analysis/results.json','source_results_sha256'),
                     (parent/'production/selected_references.json','selected_references_sha256'),
                     (parent/'production/analysis/posthoc_coarse_selection_controls.json','mapping_sha256')]:
        if digest(file)!=p[key]:raise ValueError('old calibrated source changed')
    logical=json.loads((out/'logical_inventory.json').read_text());physical=json.loads((out/'physical_inventory.json').read_text())
    validate_inventory(logical,physical)
    mappings=json.loads((parent/'production/analysis/posthoc_coarse_selection_controls.json').read_text())['comparisons']
    selected=json.loads((parent/'production/selected_references.json').read_text())
    vectors={(r['context'],r['alpha'],r['method']):r['vector'] for r in selected}
    old_events=json.loads((parent/'evaluation_events.json').read_text())
    for row in logical:
        c=row['case'];method=c['role'];ctx=c['context'];a=c['alpha']
        key=ctx+'/'+str(a) if method=='context_choice' else ('left' if '__left' in ctx else 'right')+'/'+str(a) if method=='geometry_alpha' else str(a)
        label={'context_choice':'full_context_alpha','geometry_alpha':'geometry_alpha','alpha_only':'alpha_only'}
        role='goal_full_feedback' if method=='full' else mappings[label[method]]['mapping'][key]
        vector=[0.]*6 if role=='goal_full_feedback' else vectors[(ctx,a,role)]
        if row['source_role']!=role or row['vector']!=vector:raise ValueError('reference differs from frozen old mapping')
        phase=c['event_id'].split('_')[0]
        if phase not in ('early','late') or c['pulse_onset_task_tick']!=p['onsets'][ctx][phase+'_lower_tick']:
            raise ValueError('fixed new onset changed')
        source=next(e for e in old_events if e['context']==ctx and e['alpha']==a and e['force_N']==c['force_N'])
        keep=lambda e:{k:v for k,v in e.items() if k not in ('case_id','role','event_id','pulse_onset_task_tick')}
        if keep(c)!=keep(source):raise ValueError('physical/interface fields differ from original event')
    for c in json.loads((out/'cached_nominal.json').read_text()):
        if digest(c['raw_file'])!=c['raw_sha256']:raise ValueError('nominal cached raw changed')
    return out,p,logical,physical


def bind(parent):
    out,p,logical,physical=read_inputs(parent)
    shape=Path(parent)/'shape_batch64/status.json'
    if json.loads(shape.read_text())['phase']!='completed':raise ValueError('actual64 model shape required')
    if (out/'execution_binding.json').exists():raise ValueError('one explicit binding only')
    receipt=json.loads((shape.parent/'receipt.json').read_text())
    if receipt['batch']!=64 or receipt['physics_steps'] or receipt['source_plan_sha256']!=digest(Path(parent)/'plan.json'):raise ValueError('shape receipt identity mismatch')
    files=list(Path(__file__).parent.glob('*.py'))+[Path('learning/cli',f).resolve() for f in ['run_preservation_timecheck.py','analyze_preservation_timecheck.py','diagnose_reference_bias_campaign.py']]
    archive=out/'source';archive.mkdir()
    for i,f in enumerate(files):shutil.copy2(f,archive/f'{i:03d}_{f.name}')
    atomic_json(out/'execution_binding.json',dict(declaration_seal_sha256=digest(out/'seal.json'),
        source_sha256={str(f.resolve()):digest(f) for f in files},shape_receipt_sha256=digest(shape.parent/'receipt.json'),
        ready=True,real_batch_sizes=[64,64,64,16,16,16],scope='Implementation readiness only; immutable scientific protocol and old mappings unchanged.'))
    atomic_json(out/'status.json',dict(phase='prepared',execution_ready=True,active_lower_ticks=0,allocated_lower_slots=0,optimizer_updates=0))
    from prepare_short_recovery_campaign import register
    register(Path(parent)/'ACTIVE_PRESERVATION_TIMECHECK.json','STTW固定参考选择范围外起扰确认',out/'status.json')


def costs(out):
    records=[json.loads(p.read_text()) for p in (out/'physical').glob('batch_*/status.json')]
    return dict(active_lower_ticks=sum(s.get('active_lower_ticks',0) for s in records),
                allocated_lower_slots=sum(s.get('allocated_lower_slots',0) for s in records),
                physical_batches=len(records),optimizer_updates=0)


def execute(parent):
    from .reference_bias_runtime import BiasRuntime
    out,p,logical,physical=read_inputs(parent)
    if json.loads((out/'status.json').read_text())['phase']!='prepared':raise ValueError('fresh prepared run only')
    binding=json.loads((out/'execution_binding.json').read_text())
    if not binding['ready'] or digest(out/'seal.json')!=binding['declaration_seal_sha256']:raise ValueError('changed execution binding')
    for f,sha in binding['source_sha256'].items():
        if digest(f)!=sha:raise ValueError('execution source changed '+f)
    state=dict(phase='running',stage='initializing',execution_ready=True,pid=os.getpid(),completed_physical_cases=0,completed_logical_cases=0)
    start=time.monotonic()
    def update():
        state.update(costs(out),elapsed_s=time.monotonic()-start)
        atomic_json(out/'status.json',state)
        if max(state['active_lower_ticks'],state['allocated_lower_slots'])>480000:raise ValueError('fixed240 episode limit exceeded')
    update()
    try:
        runtime=BiasRuntime(parent);results=[];offset=0
        for i,n in enumerate(binding['real_batch_sizes']):
            batch=physical[offset:offset+n]
            rr=runtime.rollout([r['case'] for r in batch],np.array([r['vector'] for r in batch]),
                out/'physical'/f'batch_{i:02d}',allowed_real_batch_sizes=(16,64))
            for row,item in zip(rr,batch):row.update(physical_id=item['physical_id'],physical_index=len(results));results.append(row)
            offset+=n;state.update(stage='fixed_evaluation',completed_physical_cases=len(results),
                completed_logical_cases=sum(r['physical_index']<len(results) for r in logical));update()
        if offset!=240:raise ValueError('complete physical inventory required')
        atomic_json(out/'physical_results.json',results);state.update(phase='completed',stage='numeric_complete_analysis_pending');update()
    except BaseException as exc:
        (out/'traceback.txt').write_text(traceback.format_exc());state.update(phase='error',error=repr(exc));update();raise
