"""Write a new paired task/runner config on the pinned 30 cm ditch branch."""
import argparse
import json
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--base',type=Path,default=Path('learning/configs/isaac_rb_jump_ditch_goal150.json'))
p.add_argument('--output',type=Path,required=True)
p.add_argument('--support-mode',choices=('rough','strict'),default='rough')
p.add_argument('--stone-height-mm',type=float,nargs=2,metavar=('MIN','MAX'),help='Override surface roughness amplitude in millimetres')
p.add_argument('--far-bank-height-cm',type=float,help='Landing bank elevation relative to takeoff bank')
a = p.parse_args()
runner = json.loads(a.base.read_text())
if a.stone_height_mm is not None:
    runner['ditch_terrain']['stone_height_min_m'],runner['ditch_terrain']['stone_height_max_m'] = [v/1000 for v in a.stone_height_mm]
if a.far_bank_height_cm is not None:
    runner['ditch_terrain']['far_bank_height_m'] = a.far_bank_height_cm/100
task = json.loads(Path(runner['task_spec']).read_text())
runner['verified_ditch_reward'] = True
runner['verified_ditch_support_mode'] = a.support_mode
runner['plant_options']['track_wheel_contacts'] = True
runner['plant_options']['wheel_patch_capacity'] = 64
runner['plant_options']['wheel_raw_contacts_only'] = True
runner['plant_options']['wheel_contact_buffer_capacity'] = 256
runner['ppo_overrides']['gamma'] = 1.
task['task']['ppo']['discounting'] = 1.
status = 'Candidate: physics-step scoring and undiscounted finite-horizon PPO'
runner['status'] = task['status'] = status
a.output.mkdir(parents=True,exist_ok=False)
path = a.output/'task.json'
runner['task_spec'] = str(path.resolve())
path.write_text(json.dumps(task,indent=2)+'\n')
(a.output/'runner.json').write_text(json.dumps(runner,indent=2)+'\n')
