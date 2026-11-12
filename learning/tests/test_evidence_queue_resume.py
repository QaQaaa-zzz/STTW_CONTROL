"""Resume/plot separation must reuse saved cases without invoking a simulator."""
import json,subprocess,sys,tempfile,unittest
from pathlib import Path

ENTRY=Path(__file__).resolve().parents[1]/'cli/run_alpha_mechanism.py'

class EvidenceResumeTests(unittest.TestCase):
 def test_completed_case_physics_and_plot_phases_reuse_artifacts(self):
  with tempfile.TemporaryDirectory() as directory:
   root=Path(directory);case=root/'saved';case.mkdir()
   (root/'jobs.json').write_text(json.dumps([{'name':'saved'}]))
   (root/'protocol.json').write_text(json.dumps({'batch_cases':1,'case_labels':['fixed0'],'title':'CPU fixture'}))
   (case/'status.json').write_text('{"stage":"complete"}')
   for name in ('figure_manifest.json','condition_metrics.json','reward_components.npz'):(case/name).write_text('cached fixture')
   for mode,expected in [('--defer-plots','physics_complete'),('--plots-only','complete')]:
    result=subprocess.run([sys.executable,str(ENTRY),'--run',str(root),'--resume',mode],capture_output=True,text=True)
    self.assertEqual(result.returncode,0,result.stderr)
    self.assertEqual(json.loads((root/'status.json').read_text())['stage'],expected)
    self.assertFalse((root/'batches').exists())
    self.assertFalse((root/'saved_plot.log').exists())
 def test_plot_only_rejects_missing_physics(self):
  with tempfile.TemporaryDirectory() as directory:
   root=Path(directory);(root/'jobs.json').write_text('[{"name":"missing"}]');(root/'protocol.json').write_text('{"batch_cases":1,"case_labels":["fixed0"]}')
   result=subprocess.run([sys.executable,str(ENTRY),'--run',str(root),'--plots-only'],capture_output=True,text=True)
   self.assertNotEqual(result.returncode,0)
   self.assertIn('unfinished physical cases',result.stderr)
   self.assertFalse((root/'batches').exists())

class EvidenceNotificationTests(unittest.TestCase):
 def test_notification_failure_does_not_invalidate_completed_work(self):
  import runpy
  from unittest.mock import patch
  entry=ENTRY.with_name('run_evidence_queue.py')
  original_run=subprocess.run
  with tempfile.TemporaryDirectory() as directory:
   root=Path(directory);marker=root/'analysis_complete'
   (root/'plan.json').write_text(json.dumps(dict(stages=[],total_episodes=0,overlap_cpu_analysis=True,final_analysis_command=[sys.executable,'-c',f'from pathlib import Path; Path({str(marker)!r}).write_text("done")'])))
   def run(args,**kwargs):
    if 'learning/scripts/notify_sttw_job.py' in args:return subprocess.CompletedProcess(args,1)
    return original_run(args,**kwargs)
   with patch.object(sys,'argv',[str(entry),'--queue',str(root)]),patch.object(subprocess,'run',side_effect=run):
    runpy.run_path(str(entry),run_name='__main__')
   self.assertTrue(marker.exists())
   self.assertEqual(json.loads((root/'status.json').read_text())['stage'],'complete')
   self.assertEqual(json.loads((root/'completion_notice.json').read_text())['status'],'failed')

 def test_simulation_failure_still_stops_queue(self):
  import runpy
  from unittest.mock import patch
  entry=ENTRY.with_name('run_evidence_queue.py')
  with tempfile.TemporaryDirectory() as directory:
   root=Path(directory);stage=root/'stage';stage.mkdir()
   (root/'plan.json').write_text(json.dumps(dict(stages=[dict(name='fixture',path=str(stage),contexts=1,episodes=6)],total_episodes=6)))
   def run(args,**kwargs):
    if 'learning/scripts/notify_sttw_job.py' in args:return subprocess.CompletedProcess(args,1)
    raise subprocess.CalledProcessError(2,args)
   with patch.object(sys,'argv',[str(entry),'--queue',str(root)]),patch.object(subprocess,'run',side_effect=run):
    with self.assertRaises(subprocess.CalledProcessError):runpy.run_path(str(entry),run_name='__main__')
   self.assertEqual(json.loads((root/'status.json').read_text())['stage'],'error')
