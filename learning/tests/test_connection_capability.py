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

    def test_origin_run_is_held_out_across_all_snapshots_and_branches(self):
        from sttw_control.empirical_capability import evaluate_connection_groups
        # Unit fixture, not experimental outcomes. Same origin has two exits;
        # its own successful records must never support its held-out queries.
        rows=[dict(origin_id='one_run',snapshot_id=s,task_id=t,skill_id='frozen',
                   candidate_id=c,state=[0.],complete=1,success_cost=1.)
              for s in ['exit_a','exit_b'] for t in ['left','right'] for c in ['slow','fast']]
        result=evaluate_connection_groups(rows,['slow','fast'],[1.],1.,.5)
        self.assertEqual(result['groups'],4)
        self.assertEqual(result['selected_groups'],0)
        self.assertEqual(result['unknown_groups'],4)
        self.assertIsNone(result['brier_score'])

    def test_missing_candidate_is_not_silently_treated_as_failure(self):
        from sttw_control.empirical_capability import evaluate_connection_groups
        rows=[dict(origin_id='r',snapshot_id='s',task_id='left',skill_id='frozen',candidate_id='slow',state=[0.],complete=1,success_cost=1.)]
        with self.assertRaises(ValueError):
            evaluate_connection_groups(rows,['slow','fast'],[1.],1.,.5)
