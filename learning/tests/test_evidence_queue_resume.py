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
