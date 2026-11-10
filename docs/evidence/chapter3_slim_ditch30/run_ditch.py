"""One declared geometry-only ditch probe; retain imported source unchanged."""
import os,sys,runpy,json
from pathlib import Path
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
root=Path('/home/qy/STTW_CONTROL/imports/sttw_landing_slim_20261110/STTW_Landing_Slim')
sys.path.insert(0,str(root/'offroad_jump_20261109/method_experiments/motor_leg_hold'))
import method as M
C=M.C
original=C.model
def model():
 _,xml=original();tree=C.ET.fromstring(xml)
 ground=tree.find("worldbody/geom[@name='ground']")
 ground.set('pos','-0.85 0 -.1');ground.set('size','4.15 4 .1')
 xml=C.ET.tostring(tree,encoding='unicode');m=C.mj.MjModel.from_xml_string(xml)
 C.SURFACE=C.SurfaceQuery(m)
 return m,xml
C.model=model
sys.argv=[str(root/'reproduce.py'),'--mode','physics','--dt-us','20','--out',str(Path(__file__).parent/'ditch30')]
runpy.run_path(str(root/'reproduce.py'),run_name='__main__')
