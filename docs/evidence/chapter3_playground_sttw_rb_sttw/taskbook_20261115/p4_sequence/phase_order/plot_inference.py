from pathlib import Path
import subprocess,os
r=Path(__file__).parent;old=r.parent/'taskbook_20261115_p4_cosine';d=Path('docs/evidence/chapter3_playground_sttw_rb_sttw/taskbook_20261115/p4_sequence/phase_order/inference');py='/home/qy/mujoco_playground/.venv/bin/python';names=['cosine_s0_u6075','cosine_s1_u6225','cosine_s2_u5200']
for m in ['map06','map07','map48']:
 cmd=[py,'learning/cli/report_skill_sequence.py','--run',str(old/'evaluation'/names[0]/m),'--label','OldS0','--routes',str(r.parent/'rb_adaptation/approach_departure_turns/routes.npz'),'--output',str(d/m)]
 for seed,name in enumerate(names):
  if seed:cmd+=['--extra',str(old/'evaluation'/name/m),'--extra-label',f'OldS{seed}']
  cmd+=['--extra',str(r/'ordered'/name/m),'--extra-label',f'OrderedS{seed}']
 subprocess.run(cmd,env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'),check=True)
