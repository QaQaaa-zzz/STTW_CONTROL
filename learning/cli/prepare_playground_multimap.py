"""Declare grouped terrain-map training and DEV collection, preserving flat jump banks."""
import argparse,copy,json,shutil,subprocess
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--source',type=Path,required=True)
p.add_argument('--world',type=Path,default=Path('learning/configs/chapter3_playground_world_map.json'))
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
base=json.loads((a.source/'resolved_config.json').read_text());world=json.loads(a.world.read_text());entries=[]
# Fixed before collection: six TRAIN maps and two DEV maps, no outcome-based selection.
layouts=[(-.4,-.4),(-.4,.4),(0.,-.4),(0.,.4),(.4,-.4),(.4,.4),(-.2,.2),(.2,-.2)]
for i,(near,far) in enumerate(layouts):
    role='TRAIN' if i<6 else 'DEV_MAP';roots=[0,3] if i<6 else [6,7]
    folder=a.output/f'map{i:02d}';folder.mkdir();w=copy.deepcopy(world);w['map_id']=f'p3_{role.lower()}_{i:02d}'
    w['terrain']['seed']=20261120+i
    for ridge,offset in zip(w['terrain']['ridges'],[near,far]):ridge['x_interval_m']=[x+offset for x in ridge['x_interval_m']]
    (folder/'world_map.json').write_text(json.dumps(w,indent=2)+'\n')
    c=copy.deepcopy(base)
    for key in ['world_map_spec','ditch_terrain','snapshot_times_s']:c.pop(key,None)
    c.update(world_map=str(folder/'world_map.json'),role=f'P3 map-grouped {role}, not P4 independent TEST',planned_activation_offset_m=8.55-10.3,task_evaluation=True,audit_actuator_owner=False)
    variants=[]
    for speed in [1.8,1.9]:
        for root in roots:
            v=copy.deepcopy(next(v for v in base['ground_handoff']['handoff_variants'] if v['root_variant']==root))
            v.update(data_split='TRAIN' if role=='TRAIN' else 'DEV',root_variant=root,requested_speed_m_s=speed,approach_speed_m_s=speed,continuation_speed_m_s=speed)
            variants.append(v)
    c['ground_handoff']['handoff_variants']=variants;c['evaluation_lanes']=[[route,'h75',v] for route in range(9) for v in range(4)]
    (folder/'config.json').write_text(json.dumps(c,indent=2)+'\n');entries.append(dict(layout=f'map{i:02d}',map_role=role,config=str(folder/'config.json'),evaluation=str(folder/'evaluation'),episodes=36,stone_seed=w['terrain']['seed'],ridge_offsets_m=[near,far]))
m=dict(entries=entries,routes=list(range(9)),seconds=10,ground_only=False,physical_episodes=288,training_episodes=216,dev_episodes=72,role='Six new TRAIN maps/two new DEV maps, both initial-root sets already used historically, not independent P4 TEST',purpose='Identify terrain/speed response without fitting only the original map; preserve failed episodes and pre-failure transitions',fixed='Original terrain generator/heights2cm4cm/stones1-4mm, ditch0.3m and both2m flat banks, RB225/H75alpha0, activation8.55, physical model/control/goal unchanged',change='Stone seed and declared ridge x translation only; no ridge enters flat banks. Two discrete constant ground speed requests.',budget='Eight36-lane GPU batches;0policy training, hard1500seconds, no retry or autoextension',expected_wallclock_s=[1100,1450],hard_limit_s=1500,stop='First error or25min. Retain completed/failed data; no outcome-based map replacement.',source_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
(a.output/'manifest.json').write_text(json.dumps(m,indent=2)+'\n');(a.output/'source').mkdir()
for file in [Path(__file__),Path('learning/src/sttw_control/playground_handoff_evaluation.py'),Path('learning/src/sttw_control/ditch_terrain.py')]:shutil.copyfile(file,a.output/'source'/file.name)
print(json.dumps(dict(maps=8,TRAIN=216,DEV=72,executed=False)))
