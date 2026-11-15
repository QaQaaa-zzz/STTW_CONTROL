from pathlib import Path
import subprocess,os,json
r=Path(__file__).parent;root=r.parent;d=Path('docs/evidence/chapter3_playground_sttw_rb_sttw/taskbook_20261115/p4_sequence/direct_current');d.mkdir(parents=True,exist_ok=True);py='/home/qy/mujoco_playground/.venv/bin/python';env=dict(os.environ,PYTHONPATH='learning/src',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
def run(args):subprocess.run([py]+args,env=env,check=True)
for mapid in ['map06','map07','map48']:
 residual=root/'taskbook_20261115_p4_cosine'
 run(['learning/cli/report_skill_sequence.py','--run',str(r/mapid/'C'),'--label','DirectC','--compare',str(r/mapid/'I'),'--compare-label','DirectI','--extra',str(residual/'evaluation/cosine_s1_u6225'/mapid),'--extra-label','ResidualC','--extra',str(residual/'independent/cosine_s1_u6225'/mapid),'--extra-label','ResidualI','--routes',str(root/'rb_adaptation/approach_departure_turns/routes.npz'),'--output',str(d/mapid)])
run(['learning/cli/calibrate_skill_sequence.py','--supported-only','--dev',str(r),'--evaluation',str(r/'map48'),'--external-calibration',str(root/'taskbook_20261115_p4_cosine/calibration/cosine_s1_u6225/calibration.json'),'--output',str(r/'calibration')])
run(['learning/cli/evaluate_continuation_planner.py','--continuous',str(r/'map48/C'),'--independent',str(r/'map48/I'),'--oracle-matrix','docs/evidence/chapter3_playground_sttw_rb_sttw/taskbook_20261115/p2_oracle_wall/oracle_matrix.csv','--routes',str(root/'rb_adaptation/approach_departure_turns/routes.npz'),'--output',str(r/'comparison')])
run(['learning/cli/report_continuation_planner.py','--run',str(r/'comparison'),'--output',str(d/'comparison')])
