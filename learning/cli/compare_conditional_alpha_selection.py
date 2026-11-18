"""Known-DEV finite route/speed/alpha selection from frozen predicted gates."""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import numpy as np

p = argparse.ArgumentParser(description=__doc__)
for name in ['before', 'after', 'data', 'protocol', 'output']:
    p.add_argument('--' + name, type=Path, required=True)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=True)
manifest = json.loads((a.data / 'manifest.json').read_text())
families = {r['map']: r['map_family'] for r in manifest['rows'] if r['split'] == 'DEV'}
pairs = [(x, y) for x in [0., .5, 1.] for y in [0., .5, 1.]]
protocol = json.loads(a.protocol.read_text())
results = []
identities = {}
initial_cache = {}
reset_audit = {}
reference_reset = {}

# Selection deliberately receives prediction/command fields only, never outcomes.
def select(candidates):
    accepted = [r for r in candidates if r['admitted']]
    return min(accepted, key=lambda r: (r['goal_time_s'], r['approach_alpha'],
               r['departure_alpha'], r['route'], r['speed'])) if accepted else None

for label, root in [('before', a.before), ('after', a.after)]:
    rows = json.loads((root / 'cases.json').read_text())
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        key = (families[row['source']], row['root'])
        path = row['prefix']
        if path not in initial_cache:
            with np.load(path) as z:
                schema=json.loads((Path(path).parent.parent/'controller_schema.json').read_text())
                offset=0;reset_ids=[];field_names=[]
                for field,shape in schema.items():
                    size=int(np.prod(shape)) if shape else 1
                    field_names.extend([field]*size)
                    if field=='frames':reset_ids.extend(range(offset+size-20,offset+size))
                    elif field in ['nominal','rates'] or field.startswith(('projection_','follower_')):
                        reset_ids.extend(range(offset,offset+size))
                    offset+=size
                controller=z['controller_state'][0];carry=np.delete(controller,reset_ids)
                state=np.concatenate([z[k][0].reshape(-1) for k in ['qpos','qvel','rb_state']]+[carry])
                reset=controller[reset_ids]
                if key not in reference_reset:reference_reset[key]=reset.copy()
                differences=np.flatnonzero(reset!=reference_reset[key])
                reset_audit[path]=dict(different_fields=sorted({field_names[reset_ids[i]] for i in differences}),different_values=len(differences))
            initial_cache[path] = hashlib.sha256(state.tobytes()).hexdigest()
        identity = initial_cache[path]
        if key in identities and identities[key] != identity:
            raise ValueError(f'Candidate initial state/history differs: {key}, {path}')
        identities[key] = identity
        groups[key].append(i)
    for (family, initial_root), indices in sorted(groups.items()):
        commands = [(rows[i]['route'], rows[i]['speed'], rows[i]['approach_alpha'],
                     rows[i]['departure_alpha']) for i in indices]
        expected = {(route, speed, x, y) for route in range(9) for speed in [1.8, 1.9] for x, y in pairs}
        if len(commands) != 162 or set(commands) != expected:
            raise ValueError('Expected one complete162-candidate matched group')
        subsets = [(f'fixed_alpha{alpha:g}', [i for i in indices if
                    rows[i]['approach_alpha'] == rows[i]['departure_alpha'] == alpha])
                   for alpha in [0., .5, 1.]]
        for offset in range(9):
            subsets.append((f'joint_budget18_offset{offset}', [i for i in indices if
                (rows[i]['approach_alpha'], rows[i]['departure_alpha']) ==
                pairs[(rows[i]['route'] + 3 * int(rows[i]['speed'] == 1.9) + offset) % 9]]))
        subsets.append(('joint_full162', indices))
        for method, subset in subsets:
            inputs = [dict(index=i, **{k: rows[i][k] for k in ['admitted', 'goal_time_s',
                      'approach_alpha', 'departure_alpha', 'route', 'speed']}) for i in subset]
            chosen = select(inputs)
            outcome = rows[chosen['index']] if chosen else None
            results.append(dict(model=label, map_family=family, root=initial_root, method=method,
                queries=len(subset), feasible_candidates=sum(rows[i]['actual_success'] for i in subset),
                no_plan=chosen is None, selected=chosen,
                actual_success=bool(outcome and outcome['actual_success']),
                actual_termination=outcome['actual_termination'] if outcome else None,
                trajectory=outcome['prefix'] if outcome else None))
summary=[]
for label, method in sorted({(r['model'], r['method']) for r in results}):
    group=[r for r in results if r['model']==label and r['method']==method]
    summary.append(dict(model=label,method=method,tasks=len(group),success=sum(r['actual_success'] for r in group),
        no_plan=sum(r['no_plan'] for r in group),oracle_feasible=sum(r['feasible_candidates']>0 for r in group),
        queries_per_task=group[0]['queries']))
(a.output/'decisions.json').write_text(json.dumps(results,indent=2)+'\n')
(a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
(a.output/'contract.json').write_text(json.dumps(dict(protocol=protocol,before=str(a.before),after=str(a.after),
    data=str(a.data),initial_state_hashes={str(k):v for k,v in identities.items()},
    scope='KnownDEV four map/root tasks; nine budget offsets reuse these same tasks, not independent seeds. '
          'Speed-only is fixed_alpha0, not an additional distinct algorithm. Raw uncalibrated gates. Task-start selection: candidate-dependent deterministic controller reset fields differ; not common-full-history online replanning.',
    reset_field_audit=reset_audit,
    actual_label_access='Only after selection, for outcome scoring and Oracle feasibility'),indent=2)+'\n')
print(json.dumps(summary,indent=2))

# Four task panels retain every method and both predictors, including no-plan.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
predictions = {}
source_rows = {}
for label, root in [('before', a.before), ('after', a.after)]:
    entries=json.loads((root/'entries.json').read_text())
    for e in entries:
        folder=Path(e['output']);source_rows[label,e['layout']]=json.loads((folder/'cases.json').read_text())
        predictions[label,e['layout']]=np.load(folder/'predictions.npz')
all_rows={label:json.loads((root/'cases.json').read_text()) for label,root in [('before',a.before),('after',a.after)]}
methods=[r['method'] for r in summary if r['model']=='after']
figures=[]
for task_number,(family,initial_root) in enumerate(sorted(identities)):
    fig,axes=plt.subplots(4,4,figsize=(18,9),layout='constrained')
    for ax,method in zip(axes.flat,methods):
        notes=[]
        for label,color in [('before','#0072B2'),('after','#D55E00')]:
            result=next(r for r in results if (r['model'],r['map_family'],r['root'],r['method'])==(label,family,initial_root,method))
            if result['no_plan']:
                notes.append(label+': no plan');continue
            row=all_rows[label][result['selected']['index']];actual=np.load(row['prefix'])
            cfg=json.loads((Path(row['prefix']).parent.parent/'resolved_config.json').read_text())
            with np.load(cfg['ground_handoff']['reference_path']) as bank:reference=bank['xy'][row['route']]
            ax.plot(reference[:,0],reference[:,1],':',color=color,alpha=.5)
            ax.plot(actual['qpos'][:,0],actual['qpos'][:,1],color=color,label=label+' actual')
            ax.scatter(*actual['qpos'][-1,:2],color=color,marker='x',s=20)
            candidates=source_rows[label,row['source']]
            lane=next(i for i,r in enumerate(candidates) if (r['route'],r['root'],r['speed'])==(row['route'],row['root'],row['speed']))
            z=predictions[label,row['source']]
            for stage,length in [('approach',int(z['entry_steps'][lane])+1),('jump',int(z['jump_steps'][lane])),('departure',int(z['departure_steps'][lane])+1)]:
                xy=z[stage][lane,:length,:2];ax.plot(xy[:,0],xy[:,1],'--',color=color,lw=.8)
            notes.append(f'{label}: r{row["route"]} v{row["speed"]} a{row["approach_alpha"]}/{row["departure_alpha"]} success={int(result["actual_success"])}')
        ax.set_title(method+'\n'+'\n'.join(notes),fontsize=7)
        ax.set(xlabel='World x (m)',ylabel='World y (m)');ax.set_xlim(1.3,20.5);ax.set_ylim(-1.5,1.5);ax.set_aspect('equal',adjustable='box');ax.grid(alpha=.2)
    for ax in list(axes.flat)[len(methods):]:ax.set_visible(False)
    axes.flat[0].legend(fontsize=7)
    fig.suptitle(f'Known DEV task {task_number}: terrain {family[:12]}, root{initial_root}; travel +x\nSolid=actual, dashed=prediction, dotted=chosen route reference; X=actual end. Same physical start, candidate-conditioned reset; not online replanning.')
    filename=f'task{task_number}_xy.png';fig.savefig(a.output/filename,dpi=140);plt.close(fig);figures.append(filename)
(a.output/'figures.json').write_text(json.dumps(figures,indent=2)+'\n')
