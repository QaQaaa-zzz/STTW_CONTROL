import unittest
import numpy as np
from sttw_control.isaac_evaluation_batch import batch_settings, split_case

class BatchEvaluationTest(unittest.TestCase):
    def test_all_speed_terrain_alpha_combinations(self):
        cases=[dict(terrain_path=t,user_speed_m_s=v) for v in (1.5,1.8,2.,2.3,2.6) for t in ('flat','2cm','4cm','6cm')]
        s=batch_settings({},cases)
        self.assertEqual(s['num_envs'],120)
        for i,c in enumerate(cases):
            self.assertEqual(s['lane_speeds'][6*i:6*i+6],[c['user_speed_m_s']]*6)
            self.assertEqual(s['lane_terrain_ids'][6*i:6*i+6],[i%4]*6)
            self.assertEqual(s['lane_alphas'][6*i:6*i+6],[0,.25,.5,.75,1,0])
            self.assertEqual(s['zero_middle'][6*i:6*i+6],[False]*5+[True])

    def test_switch_targets_follow_case_lane_order(self):
        cases=[dict(terrain_path='ridge',user_speed_m_s=2.3,alphas=[0,.5],zero_middle=[False,False],alpha_switch=dict(target_alpha=t,execution_modes=['independent']*2,progress_s_m=2.5,progress_frame='original_route')) for t in ([.25,1],[.75,0])]
        settings=batch_settings({},cases)
        self.assertEqual(settings['alpha_switch']['target_alpha'],[.25,1,.75,0])
        self.assertEqual(len(settings['alpha_switch']['execution_modes']),4)
        cases[1]['alpha_switch']['progress_s_m']=3.
        with self.assertRaises(ValueError):batch_settings({},cases)

    def test_mixed_seed_groups_preserve_frozen_parameters(self):
        import json
        from pathlib import Path
        spec=json.loads(Path('learning/configs/isaac_h75_tradeoff_repeats.json').read_text())
        cases=[dict(terrain_path='flat',user_speed_m_s=2.,seed=seed,
                    randomization=spec['randomization_by_seed'][str(seed)])
               for seed in spec['seeds'] for _ in range(20)]
        settings=batch_settings({},cases)
        self.assertEqual(settings['num_envs'],600)
        self.assertEqual(len(settings['fixed_randomization_groups']),5)
        for i,group in enumerate(settings['fixed_randomization_groups']):
            self.assertEqual(group['env_ids'],list(range(120*i,120*(i+1))))
            self.assertEqual(group['config'],cases[20*i]['randomization'])
        self.assertEqual(settings['randomization']['latency_s'][1],
                         max(c['randomization']['latency_s'][1] for c in cases))
        self.assertFalse(settings['auto_reset_horizon'])

    def test_split_keeps_policy_control_contact_axes_and_identity(self):
        n=120; lane=np.arange(n)
        arrays={'reward':np.broadcast_to(lane,(2,n)).copy(),
                'lower_trace':np.broadcast_to(lane[None,None,:,None],(2,4,n,11)).copy(),
                'measurement':np.broadcast_to(lane[None,:,None],(8,n,7)).copy(),
                'wheel_contact_points':np.broadcast_to(lane[None,None,:,None,None,None],(8,25,n,2,1,3)).copy()}
        c={'settings':batch_settings({},[dict(terrain_path='flat',user_speed_m_s=2.)]*20),
           'origin':list(range(n)), 'terrain':list(range(n)),
           'randomization_readback':{'masses':list(range(n))},'delay_steps':{'imu':list(range(n))},
           'lab_physics':dict(num_envs=n,env_origins=list(range(n)),masses=list(range(n)),materials=list(range(n)),terrain_layout={'clone_origins':list(range(n))})}
        part,meta=split_case(arrays,c,dict(terrain_path='4cm',user_speed_m_s=2.6,batch_output='batch'),19)
        np.testing.assert_array_equal(part['reward'][0],np.arange(114,120))
        np.testing.assert_array_equal(part['lower_trace'][0,0,:,0],np.arange(114,120))
        np.testing.assert_array_equal(part['measurement'][0,:,0],np.arange(114,120))
        np.testing.assert_array_equal(part['wheel_contact_points'][0,0,:,0,0,0],np.arange(114,120))
        self.assertEqual(meta['origin'],list(range(114,120)))
        self.assertEqual(meta['settings']['user_speed_m_s'],2.6)
        self.assertEqual(meta['settings']['terrains'],{'evaluation':'4cm'})
        self.assertEqual(meta['delay_steps']['imu'],list(range(114,120)))
        self.assertEqual(c['settings']['num_envs'],120)

if __name__=='__main__':unittest.main()
