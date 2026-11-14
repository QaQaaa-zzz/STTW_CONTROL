import json,copy,os,subprocess
from pathlib import Path
base=Path('/media/qy/6A7E3DB07E3D75C3/STTW_runs/chapter3_playground_sttw_rb_sttw');src=base/'e2_primitive_spatial_envelope';out=base/'e2_backtracking';out.mkdir(exist_ok=False)
cs=json.loads((src/'candidates.json').read_text());ps=json.loads((src/'plans.json').read_text());manifest=json.loads((src/'manifest.json').read_text());entries=[];plans=[]
for p in ps[::2]:
 feasible=[c for c,v in zip(cs,p['all_candidate_clearance_m']) if v>=0 and c['eligible']]
 groups=sorted([c for c in cs if c['downstream']==0],key=lambda c:c['prefix_cost_s'])
 for rank,g in enumerate(groups):
  choices=[c for c in feasible if (c['activation_id'],c['upstream'])==(g['activation_id'],g['upstream'])]
  if choices:choice=min(choices,key=lambda c:c['cost_s']);break
 plan=copy.deepcopy(p);plan.update(method='backtracking',feasible=True,candidate=choice['index'],route=choice['route'],activation_id=choice['activation_id'],activation_x_m=choice['activation_x_m'],prefix_rank=rank+1,clearance_m=p['all_candidate_clearance_m'][choice['index']])
 reused=next((x for x in ps if x['layout']==p['layout'] and x['candidate']==choice['index']),None)
 if reused:
  plan.update(evaluation=reused['evaluation'],case_dirs=reused['case_dirs'],reused=True)
 else:
  folder=out/f"layout{p['layout']:02d}";folder.mkdir();config=json.loads((Path(p['evaluation']).parent/'config.json').read_text());world=json.loads(Path(config['world_map']).read_text());config['planned_activation_offset_m']=choice['activation_x_m']-world['terrain']['front_x_m'];config['evaluation_lanes']=[[choice['route'],'h75',i] for i in range(2)];(folder/'config.json').write_text(json.dumps(config,indent=2)+'\n');plan.update(evaluation=str(folder/'evaluation'),case_dirs=[f"rb_run3_route{choice['route']}_h75_v{i}" for i in range(2)],reused=False);entries.append(dict(layout=p['layout'],config=str(folder/'config.json'),evaluation=plan['evaluation']))
 plans.append(plan)
manifest.update(entries=entries,role='E2 stronger backtracking DEV baseline; same frozen36 candidates and full feasibility, group order by canonical prefix cost, first feasible group then minimum full cost',source=str(src),physical_episodes=len(entries)*2,expected_wallclock_s=[200,600],hard_limit_s=1200,stop='One queue, first runtime error or20min; no retries',reused_conditions=sum(p['reused'] for p in plans)*2)
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');(out/'plans.json').write_text(json.dumps(plans,indent=2)+'\n');(out/'prepare.py').write_text(Path(__file__).read_text())
launch=(src/'launch.py').read_text().replace("'1500'","'1200'").replace('分段/联合规划首轮物理对照','回溯分段加强对照').replace('分段/联合规划物理对照','回溯分段加强对照');(out/'launch.py').write_text(launch)
env=os.environ.copy();env.update(PYTHONPATH='learning/src:/home/qy/DVGC/JIT/src',XLA_PYTHON_CLIENT_PREALLOCATE='false',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
with (out/'queue.log').open('w') as log:proc=subprocess.Popen(['/home/qy/mujoco_playground/.venv/bin/python',str(out/'launch.py')],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(out/'process.json').write_text(json.dumps(dict(pid=proc.pid,physical_episodes=manifest['physical_episodes']),indent=2));print(proc.pid,manifest['physical_episodes'])
