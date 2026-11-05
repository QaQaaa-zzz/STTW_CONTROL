"""Bounded stage-end H75 review using the existing export/evaluation/report CLIs."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--training-run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--config',type=Path,default=Path('learning/configs/go2_h75_review.json'));p.add_argument('--plan-only',action='store_true');a=p.parse_args();spec=json.loads(a.config.read_text());root=Path.cwd();run=a.training_run.resolve()
if a.plan_only:
 print(json.dumps(dict(config=spec,candidates='reward peak, reward-ranked periodic and final; deduplicate actual completed update',new_episodes_per_candidate=16,baseline_physics_reused=True,optimizer_updates=0),indent=2));raise SystemExit(0)
training_status=json.loads((run/'status.json').read_text())
if training_status['stage']!='training_complete':p.error('stage-end review requires completed training')
reference=json.loads((Path(spec['reference_bundle'])/'identity.json').read_text());base=json.loads((Path(spec['baseline_root'])/'lab_config.json').read_text())
for domain in spec['domains']:
 cached=json.loads((Path(spec['baseline_root'])/domain/'contract.json').read_text())
 if any(cached['settings'][key]!=base[key] for key in ('asset','controller_config','lower_actor','lower_sha256','plant_options','initial_state')) or any(cached['identity']['files'][key]!=reference['files'][key] for key in ('config.json','route.npz')):p.error('cached baseline contract differs: '+domain)
env=dict(os.environ,PYTHONPATH=str(root/'learning/src'),JAX_PLATFORMS='cpu');a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=False)
def execute(python,script,args,log):
 with log.open('w') as stream:subprocess.run([python,'-u','learning/cli/'+script,*map(str,args)],env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
execute(spec['lab_python'],'select_go2_local.py',['--training-run',run,'--metric','Episode/reward'],a.output/'selection.log')
peaks=[]
for segment in [*map(Path,spec['earlier_segments']),run]:
 row=json.loads((segment/'best_model.json').read_text());peaks.append(dict(row,checkpoint=str((segment/row['checkpoint']).resolve()),training_run=str(segment.resolve())))
peak=max(peaks,key=lambda row:row['episode_reward']);periodic=json.loads((run/'best_training.json').read_text())['best'];final=dict(checkpoint=str(run/training_status['final_checkpoint']),completed_updates=training_status['completed_updates'],training_run=str(run))
periodic['training_run']=str(run);candidates=[]
for role,row in zip(spec['candidate_roles'],[peak,periodic,final]):
 if row['completed_updates'] not in [item['completed_updates'] for item in candidates]:candidates.append(dict(row,role=role))
(a.output/'selection.json').write_text(json.dumps(dict(candidates=candidates,episode_peaks=peaks,scope='reward-based nomination before fixed-case results; cross-update episodes; no automatic adoption',budget=spec['max_new_physics_lane_steps']),indent=2))
status=dict(stage='evaluation',completed_cases=0,physics_lane_steps=0,pid=os.getpid(),started=time.time());(a.output/'status.json').write_text(json.dumps(status,indent=2));index=['# H75 offroad continuation: fixed-case DEV','','Candidates nominated from training rewards before these fixed-case results. All use the selected original orange Local300. No automatic adoption. Clone translations can differ with lane count; comparison uses the same saved local initial state, route, controller, friction and clocks.','','Original H75 reference: [static cases]('+str((root/spec['baseline_root']/'RESULTS.md').resolve())+'), [switch cases]('+str((root/spec['switch_reference']/'analysis/RESULTS.md').resolve())+').']
for candidate in candidates:
 folder=a.output/('update%04d'%candidate['completed_updates']);folder.mkdir();bundle=folder/'bundle';label=f"H75 update {candidate['completed_updates']} ({candidate['role']})"
 execute(spec['lab_python'],'export_go2_local_actor.py',['--training-run',candidate['training_run'],'--checkpoint',candidate['checkpoint'],'--h75-reference-bundle',spec['reference_bundle'],'--output',bundle],folder/'export.log')
 config=json.loads((Path(spec['baseline_root'])/'lab_config.json').read_text());config.update(num_envs=5,alphas=spec['alphas'],zero_middle=[False]*5,evaluation_role=spec['role'],role=label);config_path=folder/'static_config.json';config_path.write_text(json.dumps(config,indent=2));index+=['','## '+label,'',str(candidate['checkpoint'])]
 for domain,terrain in spec['domains'].items():
  output=folder/domain
  execute(spec['lab_python'],'evaluate_isaac_gpu.py',['--backend','lab','--configured-lanes','--bundle',bundle,'--config',config_path,'--terrain',terrain,'--output',output,'--seconds',spec['seconds']],folder/(domain+'.log'))
  execute(spec['analysis_python'],'plot_isaac_gpu_evaluation.py',['--run',output,'--baseline-run',Path(spec['baseline_root'])/domain,'--title',label+' | '+domain],folder/(domain+'_figures.log'))
  status['completed_cases']+=5;status['physics_lane_steps']+=250000;(a.output/'status.json').write_text(json.dumps(status,indent=2));index+=['',f'![{domain} five alpha XY]({output.relative_to(a.output)}/xy_all_alpha.png)',f'[{domain} reward/error diagnostics]({output.relative_to(a.output)}/figure_manifest.json)']
 switch=folder/'switch';switch.mkdir();switch_cfg=json.loads((Path(spec['switch_reference'])/'config.json').read_text());switch_cfg['role']=label;switch_cfg['evaluation_role']=spec['role'];switch_config=switch/'config.json';switch_config.write_text(json.dumps(switch_cfg,indent=2))
 execute(spec['lab_python'],'evaluate_isaac_gpu.py',['--backend','lab','--configured-lanes','--bundle',bundle,'--config',switch_config,'--terrain',spec['domains']['bump'],'--output',switch/'evaluation','--seconds',spec['seconds']],switch/'evaluation.log')
 execute(spec['analysis_python'],'analyze_isaac_switch.py',['--run',switch/'evaluation','--title',label],switch/'analysis.log');status['completed_cases']+=6;status['physics_lane_steps']+=300000;(a.output/'status.json').write_text(json.dumps(status,indent=2));index+=['',f'![Switch XY]({switch.relative_to(a.output)}/analysis/xy.png)',f'[Switch metrics and limitations]({switch.relative_to(a.output)}/analysis/RESULTS.md)'];(a.output/'RESULTS.md').write_text('\n'.join(index)+'\n')
status.update(stage='complete',finished=time.time(),adoption_status='pending full-trajectory comparison',optimizer_updates=0);(a.output/'status.json').write_text(json.dumps(status,indent=2));print(json.dumps(status))
