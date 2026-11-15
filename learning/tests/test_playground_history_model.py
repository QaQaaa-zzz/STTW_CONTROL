import unittest
import numpy as np
import torch
from sttw_control.playground_prediction import NominalRollout
from sttw_control.playground_history_model import HistoryDynamics


class HistoryModelTest(unittest.TestCase):
    def test_zero_correction_preserves_nominal_and_gradients(self):
        follower=dict(lookahead_time_s=.7,lookahead_min_speed_m_s=.3,lookahead_min_m=.3,lookahead_max_m=2.)
        weights=np.zeros((34,9));weights[:9]=np.eye(9)*.99
        nominal=NominalRollout(dict(mean=np.zeros(33),scale=np.ones(33),weights=weights),follower,'cpu')
        model=HistoryDynamics(nominal,np.zeros(381),np.ones(381),np.zeros(26),np.ones(26),np.ones(9))
        s=torch.linspace(0,10,501);xy=torch.stack((s,torch.zeros_like(s)),1)[None];heading=torch.zeros(1,501)
        pose=torch.zeros(1,12);pose[:,3]=1;velocity=torch.zeros(1,9);velocity[:,0]=1.8
        predicted,_=model.rollout(pose,velocity,velocity,torch.zeros(1,381),s,xy,heading,heading,5)
        with self.assertRaisesRegex(ValueError,'does not support alpha'):
            model.rollout(pose,velocity,velocity,torch.zeros(1,381),s,xy,heading,heading,5,alpha_request=torch.zeros(1))
        expected,_=nominal(pose,velocity,velocity,s,xy,heading,heading,5)
        torch.testing.assert_close(predicted,expected)
        checked,_,domain=model.rollout(pose,velocity,velocity,torch.zeros(1,381),s,xy,heading,heading,5,return_domain=True)
        torch.testing.assert_close(checked,predicted)
        self.assertEqual(tuple(domain.shape),(1,6,3))
        torch.testing.assert_close(domain[:,:,:2],torch.zeros(1,6,2))
        self.assertTrue(bool((domain[:,:,2]>0).all()))
        predicted[:,:,:3].square().mean().backward()
        gradients=[p.grad for p in model.parameters() if p.grad is not None]
        self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
        self.assertTrue(any(g.abs().max()>0 for g in gradients))


    def test_direct_terrain_model_does_not_apply_nominal_dynamics(self):
        from sttw_control.playground_terrain_features import TerrainFeatures
        follower=dict(lookahead_time_s=.7,lookahead_min_speed_m_s=.3,lookahead_min_m=.3,lookahead_max_m=2.)
        weights=np.zeros((34,9));weights[:9]=np.eye(9)*.2
        nominal=NominalRollout(dict(mean=np.zeros(33),scale=np.ones(33),weights=weights),follower,'cpu')
        terrain=TerrainFeatures(np.zeros((21,101)),np.ones((21,101)),[-2,8,-1,1],.1)
        model=HistoryDynamics(nominal,np.zeros(381),np.ones(381),np.zeros(101),np.ones(101),np.ones(9),terrain=terrain,model_kind='direct')
        s=torch.linspace(0,6,301);xy=torch.stack((s,torch.zeros_like(s)),1)[None];heading=torch.zeros(1,301)
        pose=torch.zeros(1,12);pose[:,3]=1;velocity=torch.zeros(1,9);velocity[:,0]=1.8
        prediction,vel=model.rollout(pose,velocity,velocity,torch.zeros(1,381),s,xy,heading,heading,5)
        torch.testing.assert_close(vel,velocity[:,None].expand(-1,6,-1))
        self.assertAlmostEqual(float(prediction[0,-1,0].detach()),.18,places=5)
        prediction[:,:,:3].square().mean().backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    def test_speed_request_reaches_causal_nominal_features(self):
        follower=dict(lookahead_time_s=.7,lookahead_min_speed_m_s=.3,lookahead_min_m=.3,lookahead_max_m=2.)
        weights=np.zeros((34,9));weights[31,0]=1
        nominal=NominalRollout(dict(mean=np.zeros(33),scale=np.ones(33),weights=weights),follower,'cpu')
        model=HistoryDynamics(nominal,np.zeros(381),np.ones(381),np.zeros(26),np.ones(26),np.ones(9))
        s=torch.linspace(0,10,501);xy=torch.stack((s,torch.zeros_like(s)),1)[None].expand(2,-1,-1);heading=torch.zeros(2,501)
        pose=torch.zeros(2,12);pose[:,3]=1;velocity=torch.zeros(2,9)
        _,predicted=model.rollout(pose,velocity,velocity,torch.zeros(2,381),s,xy,heading,heading,2,speed_request=torch.tensor([1.8,1.9]))
        torch.testing.assert_close(predicted[:,1:,0],torch.tensor([[1.8,1.8],[1.9,1.9]]))

    def test_stage_alpha_is_required_and_preserved_across_continuation(self):
        import io
        follower=dict(lookahead_time_s=.7,lookahead_min_speed_m_s=.3,lookahead_min_m=.3,lookahead_max_m=2.)
        weights=np.zeros((34,9));weights[:9]=np.eye(9)
        nominal=NominalRollout(dict(mean=np.zeros(33),scale=np.ones(33),weights=weights),follower,'cpu')
        torch.manual_seed(17)
        model=HistoryDynamics(nominal,np.zeros(381),np.ones(381),np.zeros(27),np.ones(27),np.ones(9),alpha_conditioned=True)
        with torch.no_grad():model.correction[-1].weight.normal_(std=.03)
        s=torch.linspace(0,10,501);xy=torch.stack((s,torch.zeros_like(s)),1)[None].expand(2,-1,-1);heading=torch.zeros(2,501)
        pose=torch.zeros(2,12);pose[:,3]=1;velocity=torch.zeros(2,9);velocity[:,0]=1.8
        state=model.initialize(pose,velocity,velocity,torch.zeros(2,381));alpha=torch.tensor([0.,1.])
        with self.assertRaisesRegex(ValueError,'explicit stage alpha'):
            model.rollout_state(state,s,xy,heading,heading,10)
        full=model.rollout_state(state,s,xy,heading,heading,10,alpha_request=alpha)
        first=model.rollout_state(state,s,xy,heading,heading,4,alpha_request=alpha)
        stream=io.BytesIO();torch.save(first['next_state'],stream);stream.seek(0)
        restored=torch.load(stream,weights_only=False)
        second=model.rollout_state(restored,s,xy,heading,heading,6,alpha_request=alpha)
        torch.testing.assert_close(full['pose'][:,4:],second['pose'],rtol=0,atol=0)
        torch.testing.assert_close(full['next_state'].hidden,second['next_state'].hidden,rtol=0,atol=0)
        self.assertEqual(second['next_state'].elapsed_steps,10)
        import tempfile,json
        from pathlib import Path
        from sttw_control.playground_history_model import load_history_model
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);torch.save(model.state_dict(),root/'best.pt')
            np.savez(root/'nominal.npz',mean=np.zeros(33),scale=np.ones(33),weights=weights)
            (root/'contract.json').write_text(json.dumps(dict(follower=follower,alpha_conditioned=True)))
            loaded=load_history_model(root,device='cpu')
            check=loaded.rollout_state(state,s,xy,heading,heading,10,alpha_request=alpha)
            torch.testing.assert_close(full['pose'],check['pose'],rtol=0,atol=0)

        self.assertGreater(float((full['velocity'][0]-full['velocity'][1]).abs().max().detach()),1e-6)


if __name__=='__main__':unittest.main()
