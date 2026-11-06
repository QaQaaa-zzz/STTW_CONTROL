import unittest
from sttw_control.empirical_capability import ConnectionCapability

class ConnectionCapabilityTests(unittest.TestCase):
    def test_task_and_skill_conditioning_changes_selection(self):
        rows=[dict(state=[0.,0.],task_id=task,skill_id='frozen',candidate_id=c,
                   complete=int(c==winner),success_cost=5. if c==winner else None)
              for task,winner in [('left','slow'),('right','fast')] for c in ['slow','fast']]
        q=ConnectionCapability(rows,[1,1],1,.5)
        self.assertEqual(q.query([0,0],'left','frozen',['slow','fast'])['best_candidate'],'slow')
        self.assertEqual(q.query([0,0],'right','frozen',['slow','fast'])['best_candidate'],'fast')
        self.assertIsNone(q.query([0,0],'left','different_policy',['slow','fast'])['best_candidate'])
        self.assertIsNone(q.query([10,10],'left','frozen',['slow','fast'])['best_candidate'])

    def test_failure_cost_excluded_and_completion_precedes_efficiency(self):
        rows=[dict(state=[0.],task_id='left',skill_id='frozen',candidate_id=c,complete=y,success_cost=cost)
              for c,y,cost in [('a',1,10.),('a',1,20.),('b',1,1.),('b',0,-999.),('c',0,None)]]
        q=ConnectionCapability(rows,[1],1,.5).query([0],'left','frozen',['a','b','c'])
        a,b,c=q['estimates']
        self.assertEqual(q['ranked_candidates'],['a','b','c'])
        self.assertEqual(a['success_cost'],15.)
        self.assertEqual(b['complete_probability'],.5)
        self.assertEqual(b['success_cost'],1.)
        self.assertIsNone(c['success_cost'])
