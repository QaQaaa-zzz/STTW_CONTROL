"""Bounded priority-return-v2 audit/tests/probe/screen. Defaults to dry run."""
import argparse,copy,csv,hashlib,importlib.util,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
SOURCE=Path('/home/qy/STTW_CONTROL/runs/soft_budget_ecbc1_20260920/training')
OLD=Path('/home/qy/STTW_CONTROL_mode_isolation/runs/mode_isolation')
SCENES=['ordinary_accel','core_left','core_right','disturbance_left','disturbance_right','synthetic_turn']

def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');tmp.replace(p)

def verify_cached_trace(path,config,identity=None):
 """Reuse only a completed trace with the same declared physical task and actor."""
 from dataclasses import asdict
 declaration=json.loads((Path(path)/'declaration.json').read_text())
 status=json.loads((Path(path)/'status.json').read_text())
 if status.get('status')!='complete' or declaration['config']!=json.loads(json.dumps(asdict(config))) or declaration['seed']!=49001:
  raise ValueError(f'cached trace task/status mismatch: {path}')
 if declaration.get('policy')!=identity:raise ValueError(f'cached trace policy mismatch: {path}')


def configurations():
 """Checked-in frozen first-screen settings, not an implicit old-run resume."""
 train=json.loads((ROOT/'learning/configs/ppo_priority_v2_screen.json').read_text())
 out={}
 for arm in ('A','B','C'):
  task=json.loads((ROOT/f'learning/configs/priority_v2_{arm.lower()}.json').read_text())
  out[arm]={'task':task,'training':copy.deepcopy(train),'source_checkpoint':None,'formal_transitions':2097152,'role':'oracle diagnostic' if arm=='C' else 'wheel-speed observation'}
 return out


def run_contracts(out):
 import xml.etree.ElementTree as ET
 modules=['test_priority_return_v2','test_priority_v2_training','test_priority_v2_environment','test_priority_v2_campaign','test_rsl_training','test_soft_budget_reward','test_tracking_reward']
 command=[sys.executable,'-m','pytest',*[f'learning/tests/{name}.py' for name in modules],'-q','--junitxml='+str(out/'tests.xml')]
 with (out/'tests.log').open('w') as f:
  result=subprocess.run(command,cwd=ROOT,env=dict(os.environ,JAX_PLATFORMS='cpu'),stdout=f,stderr=subprocess.STDOUT)
 suites=ET.parse(out/'tests.xml').getroot().iter('testsuite')
 totals={k:0 for k in ('tests','failures','errors','skipped')}
 for suite in suites:
  for key in totals:totals[key]+=int(suite.attrib.get(key,0))
 passed=result.returncode==0 and totals['tests']>0 and all(totals[k]==0 for k in ('failures','errors','skipped'))
 write(out/'tests_receipt.json',{'passed':passed,**totals,'command':command})
 if not passed:raise RuntimeError('contract tests failed or skipped; see tests.log')

def scenarios(cfg):
 spec=importlib.util.spec_from_file_location('review',ROOT/'learning/cli/five_scene_review.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 from dataclasses import replace
 cfg=replace(cfg,timed_reference=replace(cfg.timed_reference,priority_v2_screen=False))
 _,cs,ws=m.review_configurations(cfg,49001,[0.])
 cs={k:replace(v,timed_reference=replace(v.timed_reference,priority_v2_screen=False)) for k,v in cs.items() if k in SCENES}
 return cs,ws

def evaluate_checkpoint(task,checkpoint,out):
 import numpy as np
 from dataclasses import asdict
 from sttw_control.env import RecoveryEnv,config_from_dict
 from sttw_control.network import load_policy,make_policy_identity
 from sttw_control.evaluation import evaluate
 from sttw_control.priority_v2_analysis import metrics
 cfg=config_from_dict(task);identity=make_policy_identity(RecoveryEnv(cfg).bundle.identity,asdict(cfg),cfg.observation.history_steps);policy=load_policy(checkpoint,expected=identity)
 import torch
 snapshot=torch.load(Path(checkpoint)/'rsl_snapshot.pt',map_location='cpu',weights_only=False)
 logits=load_policy(checkpoint,expected=identity,return_logits=True)
 latent_std=snapshot['policy']['log_std'].exp().numpy()
 cs,ws=scenarios(cfg);rows=[]
 for name,c in cs.items():
  env=RecoveryEnv(c);prepared=env.reset(49001);dest=Path(out)/name
  for alpha in cfg.priority.validation_alphas:
   for method in ['baseline','policy','policy_stochastic']:
    p=dest/method if method=='baseline' else dest/f'alpha_{alpha:g}'/method
    if not (p/'trace.npz').exists():
     count=[0];rng=np.random.default_rng(49001)
     def actor(obs):
      count[0]+=1
      if count[0]<=round(ws.get(name,0)/c.controller.dt):return np.zeros(2)
      return np.tanh(np.asarray(logits(obs))+latent_std*rng.normal(size=2)) if method=='policy_stochastic' else policy(obs)
     evaluate(env,p,seed=49001,priority_alpha=float(alpha) if method!='baseline' else 0.,policy=actor if method!='baseline' else None,policy_identity=identity if method!='baseline' else None,initial_state=prepared)
    verify_cached_trace(p,c,identity if method!='baseline' else None)
    tr=dict(np.load(p/'trace.npz'));tr['priority_alpha']=np.full_like(tr['priority_alpha'],alpha);mt,states,parts=metrics(tr,asdict(c));mt.update(scenario=name,method=method,alpha=float(alpha));scoring=p/f'score_alpha_{alpha:g}';write(scoring/'v2_metrics.json',mt);write(scoring/'v2_replay.json',{'states':states,'reward_parts':[{k:float(v) for k,v in d.items()} for d in parts]});rows.append(mt)
 rows_policy=[x for x in rows if x['method']=='policy'];summary={'checkpoint':str(checkpoint),'Jp':float(np.mean([x['Jp'] for x in rows_policy])),'Jv':float(np.mean([x['Jv'] for x in rows_policy])),'engineering_accepted':all(x['v2_engineering_accepted'] for x in rows_policy),'strict_accepted':all(x['v2_strict_accepted'] for x in rows_policy),'physical_failures':sum(x['physical_failed'] for x in rows_policy),'stochastic_physical_failures':sum(x['physical_failed'] for x in rows if x['method']=='policy_stochastic'),'stochastic_scope':'one fixed Gaussian draw sequence per scene; diagnostic only, not a failure probability estimate','rows':rows};write(Path(out)/'summary.json',summary);return summary

def audit(out):
 from sttw_control.priority_v2_analysis import metrics
 import numpy as np
 from importlib.metadata import version
 rows=[]
 for group in ['source_review','modes20_review','modes_review']:
  for p in (OLD/group/'evaluation').glob('alpha_*/seed_49001/*/residual'):
   t=dict(np.load(p/'trace.npz'));c=json.loads((p/'declaration.json').read_text())['config'];m,_,_=metrics(t,c);m.update(group=group,path=str(p));rows.append(m)
 write(out/'old_contract_comparison.json',rows)
 write(out/'audit.json',{'complete':True,'scope':'read-only existing traces and V2 counterfactual scoring, no identity bypass','source_training':str(SOURCE),'source_budget':250,'source_actor_update':244,'local_updates':[20,250],'dependencies':{x:version(x) for x in ['rsl-rl-lib','jax','torch','mujoco','flax']},'base':'3db2ab271085847367f533099d5234df3661c813','git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'dirty_diff':subprocess.check_output(['git','diff'],cwd=ROOT,text=True),'reference':'positive budget terms verified against uploaded scalar reference'})

def probe(out):
 import numpy as np
 from dataclasses import asdict
 from sttw_control.env import RecoveryEnv,config_from_dict
 from sttw_control.network import load_policy,make_policy_identity
 from sttw_control.evaluation import evaluate
 from sttw_control.priority_v2_analysis import metrics
 source=OLD/'specialization/alpha_0/training';cfg=config_from_dict(json.loads((source/'declaration.json').read_text())['task']);cp=source/'checkpoints/update_0020'
 identity=make_policy_identity(RecoveryEnv(cfg).bundle.identity,asdict(cfg),cfg.observation.history_steps);policy=load_policy(cp,expected=identity);cs,_=scenarios(cfg)
 candidates=[{'id':i,'steer':steer,'rear':rear,'start':1.,'duration':duration} for i,(steer,rear,duration) in enumerate((s,r,d) for s in [-.15,0.,.15] for r in [0.,-.15,-.30,-.45] for d in [.5,1.])]
 write(out/'candidates.json',candidates)
 with (out/'candidates.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=candidates[0]);w.writeheader();w.writerows(candidates)
 rows=[]
 for name in ['core_left','core_right']:
  env=RecoveryEnv(cs[name]);prepared=None
  for cand in candidates:
   p=out/name/f"candidate_{cand['id']:02d}";count=[0]
   def actor(obs):
    t=count[0]*env.config.controller.dt;count[0]+=1;u=(t-cand['start'])/cand['duration'];b=np.sin(np.pi*u)**2 if 0<=u<=1 else 0.;return np.clip(np.asarray(policy(obs))+b*np.array([cand['steer'],cand['rear']]),-1,1)
   if not (p/'trace.npz').exists():
    if prepared is None:prepared=env.reset(49001)
    evaluate(env,p,seed=49001,priority_alpha=0.,policy=actor,policy_identity=identity,initial_state=prepared)
   verify_cached_trace(p,env.config,identity)
   tr=dict(np.load(p/'trace.npz'));m,_,_=metrics(tr,asdict(env.config));m.update(scenario=name,candidate=cand,path=str(p));rows.append(m);write(out/'results.json',{'complete':False,'rows':rows});print(name,cand['id'],m['v2_engineering_accepted'],flush=True)
 def frontier(xs):return [x['path'] for x in xs if not any(y['Jp']<=x['Jp'] and y['Jv']<=x['Jv'] and (y['Jp']<x['Jp'] or y['Jv']<x['Jv']) for y in xs)]
 fronts={name:{'final':frontier([x for x in rows if x['scenario']==name and not x['physical_failed'] and x['v2_final_complete']]),'old_deadline':frontier([x for x in rows if x['scenario']==name and not x['physical_failed'] and x['v1_final_hold'] and not x['v1_deadline_missed']]),'v2':frontier([x for x in rows if x['scenario']==name and x['v2_engineering_accepted']]),'strict':frontier([x for x in rows if x['scenario']==name and x['v2_strict_accepted']])} for name in ['core_left','core_right']}
 write(out/'results.json',{'complete':True,'rows':rows,'frontiers':fronts,'actual_transitions':sum(x['transitions'] for x in rows),'prepared_transitions':1400,'finite_family_only':True})
 # Fail-closed gate: compare real accepted recovery with the most favorable zero-cost immediate physical failure.
 good=[x for x in rows if x['v2_engineering_accepted']];passing=bool(good) and min(x['v2_return'] for x in good)>-200
 write(out/'escape_gate.json',{'passed':passing,'scope':'observed finite probe recovery trajectories versus optimistic immediate physical-failure return -200; not an unbounded-state guarantee','accepted_candidates':len(good),'worst_accepted_return':min((x['v2_return'] for x in good),default=None),'failure_return':-200,'limitations':'Additional stopped/late/end-departure and discounted trajectory escape cases required before screen','screen_authorized':False})

def escape(out):
 """Bounded physical attempts; names are intended failures, never assumed outcomes."""
 import numpy as np
 from dataclasses import asdict
 from sttw_control.env import RecoveryEnv,config_from_dict
 from sttw_control.network import load_policy,make_policy_identity
 from sttw_control.evaluation import evaluate
 from sttw_control.priority_v2_analysis import metrics,replay,absorbed_returns
 probe_result=json.loads((out/'probe/results.json').read_text())
 if not probe_result['complete']:raise ValueError('probe must complete first')
 source=OLD/'specialization/alpha_0/training';cfg=config_from_dict(json.loads((source/'declaration.json').read_text())['task']);cp=source/'checkpoints/update_0020'
 identity=make_policy_identity(RecoveryEnv(cfg).bundle.identity,asdict(cfg),cfg.observation.history_steps)
 policy=load_policy(cp,expected=identity);cs,_=scenarios(cfg);env=RecoveryEnv(cs['core_left']);prepared=None
 cases=[dict(name='stop_attempt',start=1.,end=10.,offset=[0.,-1.]),
        dict(name='nonreturn_attempt',start=1.,end=10.,offset=[.4,0.]),
        dict(name='late_return_attempt',start=1.,end=6.,offset=[0.,-.8]),
        dict(name='early_fall_attempt',start=.1,end=10.,offset=[1.,0.]),
        dict(name='end_departure_attempt',start=9.,end=10.,offset=[1.,0.])]
 write(out/'escape/cases.json',{'cases':cases,'max_control_transitions':10000,'scope':'bounded physical attempts; classify actual outcomes; do not modify physics or residual limits'})
 rows=[]
 for case in cases:
  dest=out/'escape'/case['name'];count=[0]
  def actor(obs):
   t=count[0]*env.config.controller.dt;count[0]+=1
   return np.clip(np.asarray(policy(obs))+(np.asarray(case['offset']) if case['start']<=t<case['end'] else 0.),-1.,1.)
  if not (dest/'trace.npz').exists():
   if prepared is None:prepared=env.reset(49001)
   evaluate(env,dest,seed=49001,priority_alpha=0.,policy=actor,policy_identity=identity,initial_state=prepared)
  verify_cached_trace(dest,env.config,identity)
  tr=dict(np.load(dest/'trace.npz'));m,_,_=metrics(tr,asdict(env.config));m.update(name=case['name'],path=str(dest))
  m['stationary_seconds']=float(np.sum(abs(tr['true_forward_speed'][1:])<.1)*env.config.controller.dt)
  m['scores_by_alpha']={}
  for alpha in (0.,.5,1.):
   _,_,parts=replay(tr,asdict(env.config),alpha)
   m['scores_by_alpha'][str(alpha)]=absorbed_returns([sum(p.values()) for p in parts],env.horizon)
  rows.append(m);write(out/'escape/results.json',{'complete':False,'rows':rows})
 good=[x for x in probe_result['rows'] if x['scenario']=='core_left' and x['v2_engineering_accepted']]
 reference=None;comparisons=[]
 if good:
  best=min(good,key=lambda x:x['Jp']);path=Path(best['path']);tr=dict(np.load(path/'trace.npz'));c=json.loads((path/'declaration.json').read_text())['config']
  _,_,parts=replay(tr,c,0.);reference=absorbed_returns([sum(p.values()) for p in parts],env.horizon);reference['path']=str(path)
  for row in rows:
   score=row['scores_by_alpha']['0.0'];invalid=not row['v2_engineering_accepted']
   comparisons.append({'case':row['name'],'invalid':invalid,'invalid_outscores_recovery':invalid and any(score[k]>reference[k] for k in ('discounted','undiscounted'))})
 coverage={'feasible_recovery':bool(good),'stopped':any(x['stationary_seconds']>=.5 for x in rows),
           'nonreturn':any(not x['physical_failed'] and not x['v2_final_complete'] for x in rows),
           'late':any(x['v2_late_recovery'] for x in rows),
           'early_fall':any(x['physical_failed'] and x['transitions']*env.config.controller.dt<1. for x in rows),
           'end_departure':any(x['name']=='end_departure_attempt' and not x['v2_final_complete'] for x in rows)}
 passed=all(coverage.values()) and not any(x['invalid_outscores_recovery'] for x in comparisons)
 write(out/'escape/results.json',{'complete':True,'rows':rows,'coverage':coverage,'reference':reference,'comparisons':comparisons})
 write(out/'probe/escape_gate.json',{'passed':passed,'screen_authorized':passed,'coverage':coverage,'comparisons':comparisons,
  'reason':'finite physical escape checks passed' if passed else 'finite physical escape coverage or return-order gate failed; do not enlarge penalty or training budget automatically',
  'scope':'alpha0 matched core-left physical trajectories, gamma=.9995, terminal failure once and zero absorbing tail; no unbounded-state guarantee'})
 return passed


def screen(out,configs):
 gate=out/'probe/escape_gate.json'
 if not gate.exists() or not json.loads(gate.read_text()).get('screen_authorized'):
  write(out/'status.json',{'phase':'blocked','complete':False,'formal_transitions':0,'reason':'feasibility and termination-escape gate has not passed','gate':str(gate)})
  raise RuntimeError('screen blocked: complete feasibility and termination-escape gate required; no automatic larger failure penalty')
 run_contracts(out)
 receipt=out/'tests_receipt.json'
 if not receipt.exists() or not json.loads(receipt.read_text()).get('passed'):raise RuntimeError('contract tests must pass before screen')
 for arm,c in configs.items():
  dest=out/arm/'training';log=out/arm/'training.log';log.parent.mkdir(parents=True,exist_ok=True)
  if dest.exists():raise FileExistsError(f'refusing to overwrite or auto-resume {dest}')
  write(out/'status.json',{'phase':'training','arm':arm,'complete':False,'total_formal_budget':6291456})
  with log.open('w') as stream:subprocess.run([sys.executable,str(ROOT/'learning/cli/train.py'),'--task',str(out/f'frozen/{arm}_task.json'),'--config',str(out/f'frozen/{arm}_ppo.json'),'--output',str(dest)],cwd=ROOT,env=dict(os.environ,JAX_PLATFORMS='cuda',XLA_PYTHON_CLIENT_PREALLOCATE='false',STTW_SOURCE_REVISION=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()),stdout=stream,stderr=subprocess.STDOUT,check=True)
  status=json.loads((dest/'status.json').read_text())
  if status.get('phase')=='error':raise RuntimeError(f'{arm}: training error')
 write(out/'status.json',{'phase':'complete','complete':True,'scope':'bounded three-arm alpha0 screen; consult each arm for early stops'})

def report(out):
 import numpy as np
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 from sttw_control.priority_v2_analysis import comparison_figure,replay
 result=json.loads((out/'probe/results.json').read_text());rows=result['rows'];dest=out/'analysis';dest.mkdir(exist_ok=True)
 lines=['# Priority V2 有限候选检查','',f"已完成候选：{len(rows)}/48；完整数据见 [results.json](../probe/results.json)。",'',
        '这是冻结 local-update20 Actor 上叠加有限平滑偏置的物理搜索，不是新 PPO 训练。工程恢复与严格过程门槛分别统计；有限家族失败不证明全局无解。','',
        '| 场景 | 候选数 | 工程恢复 | 严格过程 | 实测速度下降 |','|---|---:|---:|---:|---:|']
 figures=[]
 for scene in ('core_left','core_right'):
  group=[r for r in rows if r['scenario']==scene];good=[r for r in group if r['v2_engineering_accepted']]
  lines.append(f"| {scene} | {len(group)} | {len(good)} | {sum(r['v2_strict_accepted'] for r in group)} | {sum(r['active_slowdown_observed'] for r in group)} |")
  fig,ax=plt.subplots(figsize=(7,5),constrained_layout=True)
  for label,predicate,color in [('Not engineering qualified',lambda r:not r['v2_engineering_accepted'],'#888888'),('Engineering only',lambda r:r['v2_engineering_accepted'] and not r['v2_strict_accepted'],'#0072B2'),('Strict working bands',lambda r:r['v2_strict_accepted'],'#009E73')]:
   selected=[r for r in group if predicate(r)]
   if selected:ax.scatter([r['Jp'] for r in selected],[r['Jv'] for r in selected],label=label,color=color)
  for r in group:ax.annotate(str(r['candidate']['id']),(r['Jp'],r['Jv']),fontsize=7,xytext=(3,3),textcoords='offset points')
  ax.set(xlabel='Jp: whole-episode normalized path cost',ylabel='Jv: whole-episode normalized speed cost',title=scene+' / finite 24-candidate family');ax.legend(fontsize=8);ax.grid(alpha=.2)
  for suffix in ('.png','.pdf'):fig.savefig(dest/(scene+'_frontier'+suffix),dpi=150)
  plt.close(fig)
  figures.extend(['',f'![{scene} candidates]({scene}_frontier.png)'])
  selected=sorted(good or group,key=lambda r:r['Jp'])[:3]
  baseline=OLD/'source_review/evaluation/alpha_0/seed_49001'/scene/'baseline'
  paths=[Path(r['path']) for r in selected];labels=[f"Candidate {r['candidate']['id']}" for r in selected]
  if baseline.exists():
   bt=dict(np.load(baseline/'trace.npz'));st=dict(np.load(paths[0]/'trace.npz'))
   if not all(np.array_equal(bt[k],st[k]) for k in ('command_schedule','event')) or not all(np.allclose(bt[k][0],st[k][0],atol=1e-7,rtol=0) for k in ('qpos','qvel')):raise ValueError('historical baseline physical pairing mismatch')
   paths.insert(0,baseline);labels.insert(0,'ECBC + ESO')
  comparison_figure(paths,labels,dest/(scene+'_comparison'),scene+' / selected diagnostic candidates; local update20 + bias')
  figures.extend(['',f'![{scene} trajectories and errors]({scene}_comparison.png)'])
 lines.extend(figures)
 replay_dir=dest/'replay';replay_dir.mkdir(exist_ok=True)
 cross=[]
 for row in rows:
  path=Path(row['path']);tr=dict(np.load(path/'trace.npz'));cfg=json.loads((path/'declaration.json').read_text())['config']
  _,states,parts,raw=replay(tr,cfg,0.,return_raw=True)
  data={'step':np.arange(1,len(parts)+1),'time_s':tr['time'][1:],'reward_v2':np.asarray([sum(p.values()) for p in parts]),'reward_recorded_v1':tr['reward'][1:],
        'speed_reference_active':tr['reference_command'][:-1,0],'speed_true':tr['true_forward_speed'][1:],'speed_wheel':tr['measurement'][1:,5]*.1,
        'lateral_error_m':tr['path_features'][1:,0],'heading_error_rad':tr['path_features'][1:,1]}
  for prefix,values in [('reward_',parts),('raw_',raw),('state_',states)]:
   for k in values[0]:data[prefix+k]=np.asarray([v[k] for v in values])
  name=row['scenario']+'_'+path.name
  np.savez_compressed(replay_dir/(name+'.npz'),**data)
  np.savetxt(replay_dir/(name+'.csv'),np.column_stack(list(data.values())),delimiter=',',header=','.join(data),comments='')
  cross.append({'scenario':row['scenario'],'candidate':row['candidate'],'returns':{str(a):float(sum(sum(p.values()) for p in replay(tr,cfg,a)[2])) for a in (0.,.5,1.)}})
 write(dest/'common_trajectory_alpha_rescore.json',cross)
 lines.extend(['','所有48条候选的 V2逐步奖励、原始成本、状态与速度误差在 [replay/](replay/)（CSV/NPZ）；同轨迹三种alpha重评分见 [common_trajectory_alpha_rescore.json](common_trajectory_alpha_rescore.json)。这不是三种alpha的独立闭环运行。'])
 gate=out/'probe/escape_gate.json'
 if gate.exists():
  g=json.loads(gate.read_text());lines.extend(['','## 正式训练门槛','',f"screen_authorized={g.get('screen_authorized',False)}；原因：{g.get('reason',g.get('limitations',''))}",'','详见 [escape_gate.json](../probe/escape_gate.json)。未通过时正式训练保持停止，不扩大失败罚或训练预算。'])
 lines.extend(['','速度下降仅指指令后最低真实速度比指令前低超过0.01m/s，是观测描述，不单独证明策略学会主动减速。','V2 统一重评分图不等于旧模型当时实际得到的 V1 奖励；原始 trace 中的 reward、分项及终止记录保持不变。共同最终保持要求速度±0.05m/s、横向±0.10m、航向±0.15rad、roll/roll-rate±0.30，保持0.5s且通过原出口+1m。'])
 (dest/'REPORT.md').write_text('\n'.join(lines)+'\n')

def main():
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=['audit','tests','probe','escape','screen','evaluate','report'],required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--updates',type=int,default=16);p.add_argument('--execute',action='store_true');p.add_argument('--task',type=Path);p.add_argument('--checkpoint',type=Path);a=p.parse_args()
 if a.updates!=16:raise ValueError('first screen is exactly16 updates per arm; no automatic extension')
 out=a.output.resolve();out.mkdir(parents=True,exist_ok=True);cfgs=configurations();write(out/'dry_run.json',{'stage':a.stage,'execute':a.execute,'arms':cfgs,'total_formal_budget':6291456,'source_model':None,'probe_max_transitions':96000,'scenes':SCENES,'evaluation_updates':[0,8,16],'evaluation_episodes_per_checkpoint':18,'evaluation_max_transitions_per_checkpoint':34800,'stochastic_action_seed':49001})
 for arm,c in cfgs.items():write(out/f'frozen/{arm}_task.json',c['task']);write(out/f'frozen/{arm}_ppo.json',c['training'])
 if not a.execute:print('DRY RUN',out/'dry_run.json');return
 if a.stage=='audit':audit(out)
 elif a.stage=='probe':probe(out/'probe')
 elif a.stage=='escape':escape(out)
 elif a.stage=='report':report(out)
 elif a.stage=='tests':run_contracts(out)
 elif a.stage=='evaluate':evaluate_checkpoint(json.loads(a.task.read_text()),a.checkpoint,out)
 else:screen(out,cfgs)

if __name__=='__main__':main()
