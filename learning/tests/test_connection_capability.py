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

    def test_fixed_comparison_retains_failures_and_unknown_queries(self):
        from sttw_control.empirical_capability import evaluate_connection_groups
        rows=[dict(origin_id='only_origin',snapshot_id='exit',task_id='left',skill_id='frozen',
                   candidate_id=c,state=[0.],complete=y,success_cost=cost)
              for c,y,cost in [('slow',1,5.),('fast',0,None)]]
        result=evaluate_connection_groups(rows,['slow','fast'],[1.],1.,.5)
        assert result['unknown_groups']==1
        assert result['selected_groups']==0
        assert result['observed_oracle_complete_groups']==1
        slow,fast=result['fixed_candidates']
        assert slow['complete_count']==1 and slow['successful_mean_cost']==5.
        assert fast['groups']==1 and fast['complete_count']==0
        assert fast['successful_mean_cost'] is None
        assert slow['matched_decided_groups']==0

    def test_deployed_selection_threshold_and_tolerance(self):
        from sttw_control.empirical_capability import select_connection_estimate
        rows=[dict(candidate_id='slow',known=True,complete_probability=.9,success_cost=8.),
              dict(candidate_id='fast',known=True,complete_probability=.88,success_cost=5.)]
        self.assertEqual(select_connection_estimate(rows)['candidate_id'],'fast')
        self.assertEqual(select_connection_estimate(rows,probability_tolerance=.01)['candidate_id'],'slow')
        self.assertIsNone(select_connection_estimate(rows,minimum_completion=.95))

    def test_known_failures_are_not_selected_or_called_unknown(self):
        from sttw_control.empirical_capability import evaluate_connection_groups
        rows=[dict(origin_id=o,snapshot_id=o,task_id='left',skill_id='frozen',
                   candidate_id='fast',state=[0.],complete=0,success_cost=None) for o in ['a','b']]
        result=evaluate_connection_groups(rows,['fast'],[1.],1.,.5)
        self.assertEqual(result['selected_groups'],0)
        self.assertEqual(result['unknown_groups'],0)
        self.assertEqual(result['no_feasible_groups'],2)

    def test_partial_support_is_unknown_not_proven_infeasible(self):
        from sttw_control.empirical_capability import connection_selection_status
        self.assertEqual(connection_selection_status([dict(known=True),dict(known=False)],None),'unknown')
        self.assertEqual(connection_selection_status([dict(known=True)],None),'no_feasible')

    def test_near_tie_keeps_previous_only_within_probability_and_route_limits(self):
        from sttw_control.empirical_capability import select_connection_estimate
        rows=[dict(candidate_id='a',route_id='left',known=True,complete_probability=.9,success_cost=5.),
              dict(candidate_id='b',route_id='left',known=True,complete_probability=.88,success_cost=5.01)]
        self.assertEqual(select_connection_estimate(rows,previous_candidate='b',previous_route='left',time_tolerance=.02)['candidate_id'],'b')
        self.assertEqual(select_connection_estimate(rows,previous_candidate='b',previous_route='right',time_tolerance=.02)['candidate_id'],'a')
        rows[1]['complete_probability']=.7
        self.assertEqual(select_connection_estimate(rows,previous_candidate='b',previous_route='left',time_tolerance=.02)['candidate_id'],'a')

    def test_uninterrupted_decision_records_are_not_labeled_as_snapshots(self):
        from sttw_control.empirical_capability import evaluate_connection_groups
        rows=[dict(origin_id=o,decision_id=o+':exit',task_id='jump',skill_id='frozen',candidate_id='a',state=[0.],complete=1,success_cost=2.) for o in ['one','two']]
        result=evaluate_connection_groups(rows,['a'],[1.],1.,.5)
        self.assertEqual(result['selected_complete_count'],2)
        self.assertEqual(result['fixed_candidates'][0]['matched_decided_groups'],2)
        self.assertTrue(all(r['snapshot_id'] is None and r['decision_id'] for r in result['rows']))
