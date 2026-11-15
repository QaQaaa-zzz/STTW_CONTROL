from pathlib import Path
import json,subprocess,os
root=Path(__file__).parent;dest=Path('docs/evidence/chapter3_playground_sttw_rb_sttw/taskbook_20261115/p4_sequence/handoff_focus');rows=json.loads((root/'evaluation_candidates.json').read_text());env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1');py='/home/qy/mujoco_playground/.venv/bin/python'
for which in ['best','last']:
 candidates=[r for r in rows if r[which]]
 for mapid in ['map06','map07','map48']:
  first=candidates[0];label=lambda r:f"{r['arm'][0].upper()}{r['seed']}_{r['update']}"
  cmd=[py,'learning/cli/report_skill_sequence.py','--run',str(root/'evaluation'/first['name']/mapid),'--label',label(first),'--routes',str(root.parent/'rb_adaptation/approach_departure_turns/routes.npz'),'--output',str(dest/which/mapid)]
  for r in candidates[1:]:cmd+=['--extra',str(root/'evaluation'/r['name']/mapid),'--extra-label',label(r)]
  subprocess.run(cmd,env=env,check=True);print(which,mapid,flush=True)
for r in json.loads((root/'selected_comparison.json').read_text()):
 subprocess.run([py,'learning/cli/report_continuation_planner.py','--run',str(root/'comparison'/r['name']),'--output',str(dest/'comparison'/r['name'])],env=env,check=True)
