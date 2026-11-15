import io
import unittest
import torch
from sttw_control.skill_world_model import SkillWorldModel


class SkillWorldModelTest(unittest.TestCase):
    def test_joint_displacement_preserves_zero_and_serialized_continuation(self):
        torch.manual_seed(7)
        size=12+11+7+3+5+4
        old=SkillWorldModel(7,4,torch.zeros(size),torch.ones(size),torch.ones(11)*.01,torch.ones(7)*.01)
        model=SkillWorldModel.from_weights(old.state_dict());model.enable_joint_displacement(torch.ones(5)*.002)
        pose=torch.zeros(2,12);pose[:,3]=1;context=torch.zeros(2,4)
        state=old.initialize(pose,torch.zeros(2,11),torch.zeros(2,4),torch.zeros(2,3),torch.zeros(2,3),torch.zeros(2,5),context)
        with torch.no_grad():
            baseline=old.rollout(state,context,100);zero=model.rollout(state,context,100)
            torch.testing.assert_close(baseline['qpos'],zero['qpos'],rtol=0,atol=0)
            model.joint_displacement.bias.fill_(1.)
            moved=model.rollout(state,context,1)
            torch.testing.assert_close(moved['qpos'][:,1,7:]-baseline['qpos'][:,1,7:],torch.full((2,5),.002))
            torch.testing.assert_close(moved['qvel'][:,1],baseline['qvel'][:,1],rtol=0,atol=0)
            full=model.rollout(state,context,100);first=model.rollout(state,context,40)
            buffer=io.BytesIO();torch.save((model.state_dict(),first['next_state']),buffer);buffer.seek(0)
            weights,resumed=torch.load(buffer,weights_only=False);restored=SkillWorldModel.from_weights(weights)
            last=restored.rollout(resumed,context,60)
            torch.testing.assert_close(full['qpos'],torch.cat((first['qpos'],last['qpos'][:,1:]),1),rtol=0,atol=0)

    def test_full_joints_rotation_and_serialized_continuation(self):
        torch.manual_seed(3)
        history_dim=7;context_dim=4;size=12+11+history_dim+3+5+context_dim
        model=SkillWorldModel(history_dim,context_dim,torch.zeros(size),torch.ones(size),torch.ones(11)*.01,torch.ones(history_dim)*.01)
        pose=torch.zeros(2,12);pose[:,3]=1
        velocity=torch.zeros(2,11);velocity[:,9:]=torch.tensor([1.,-2.]);velocity[:,5]=.2
        context=torch.zeros(2,context_dim)
        state=model.initialize(pose,velocity,torch.zeros(2,4),torch.zeros(2,3),torch.zeros(2,3),torch.nn.functional.one_hot(torch.ones(2,dtype=torch.long),5).float(),context)
        with torch.no_grad():
            full=model.rollout(state,context,100)
            first=model.rollout(state,context,40)
            buffer=io.BytesIO();torch.save(first['next_state'],buffer);buffer.seek(0)
            resumed=torch.load(buffer,weights_only=False)
            last=model.rollout(resumed,context,60)
        torch.testing.assert_close(full['qpos'],torch.cat((first['qpos'],last['qpos'][:,1:]),1),rtol=0,atol=0)
        torch.testing.assert_close(full['next_state'].hidden,last['next_state'].hidden,rtol=0,atol=0)
        torch.testing.assert_close(full['qpos'][:,:,3:7].norm(dim=-1),torch.ones(2,101))
        self.assertGreater(float(full['qpos'][0,1,10]),0)
        self.assertLess(float(full['qpos'][0,1,11]),0)
        self.assertEqual(last['next_state'].elapsed_steps,100)
        self.assertEqual(tuple(full['contact_logits'].shape),(2,100,3))
        self.assertEqual(tuple(full['phase_logits'].shape),(2,100,5))

    def test_ordered_phases_reject_regression_and_resume(self):
        size=12+11+7+3+5+4
        model=SkillWorldModel(7,4,torch.zeros(size),torch.ones(size),torch.ones(11)*.01,torch.ones(7)*.01)
        model.enable_ordered_phases()
        pose=torch.zeros(1,12);pose[:,3]=1;context=torch.zeros(1,4)
        state=model.initialize(pose,torch.zeros(1,11),torch.zeros(1,4),torch.zeros(1,3),torch.zeros(1,3),torch.tensor([[0.,1.,0.,0.,0.]]),context)
        with torch.no_grad():
            model.decoder[-1].weight.zero_();model.decoder[-1].bias.zero_()
            model.decoder[-1].bias[-5:]=torch.tensor([100.,0.,30.,10.,20.])
            full=model.rollout(state,context,5)
            self.assertEqual(full['phase_logits'].argmax(-1).tolist(),[[3,4,2,2,2]])
            self.assertEqual(full['raw_phase_logits'].argmax(-1).tolist(),[[0,0,0,0,0]])
            first=model.rollout(state,context,2)
            buffer=io.BytesIO();torch.save((model.state_dict(),first['next_state']),buffer);buffer.seek(0)
            weights,resumed=torch.load(buffer,weights_only=False);restored=SkillWorldModel.from_weights(weights)
            last=restored.rollout(resumed,context,3)
            torch.testing.assert_close(full['qpos'],torch.cat((first['qpos'],last['qpos'][:,1:]),1),rtol=0,atol=0)
            torch.testing.assert_close(full['phase_logits'][:,2:],last['phase_logits'],rtol=0,atol=0)
        predicted=model.rollout(state,context,2)
        loss=torch.nn.functional.cross_entropy(predicted['raw_phase_logits'].flatten(0,1),torch.tensor([3,4]))
        loss.backward()
        self.assertGreater(float(model.decoder[-1].bias.grad[-5]),0.)

    def test_native_joint_loss_scale_mapping(self):
        from sttw_control.skill_world_model import JOINT_POSITION_SCALES,JOINT_VELOCITY_SCALES
        native=['rearwheel_joint','steering_joint','frontwheel_joint','hip_joint','knee_joint']
        self.assertEqual([JOINT_POSITION_SCALES[n] for n in native],[1.,.2,1.,.3,.3])
        self.assertEqual([JOINT_VELOCITY_SCALES[n] for n in native],[5.,1.,5.,1.,1.])

    def test_predicted_stop_without_true_event_time(self):
        size=12+11+7+3+5+4
        model=SkillWorldModel(7,4,torch.zeros(size),torch.ones(size),torch.ones(11)*.01,torch.ones(7)*.01)
        pose=torch.zeros(2,12);pose[:,3]=1
        context=torch.zeros(2,4)
        state=model.initialize(pose,torch.zeros(2,11),torch.zeros(2,4),torch.zeros(2,3),torch.zeros(2,3),torch.nn.functional.one_hot(torch.ones(2,dtype=torch.long),5).float(),context)
        with torch.no_grad():
            model.decoder[-1].weight.zero_();model.decoder[-1].bias.zero_();model.decoder[-1].bias[-3]=10
            result=model.rollout_until(state,context,10)
            self.assertTrue(result['event_found'].all())
            self.assertTrue((result['event_step']==1).all())
            self.assertEqual(int(result['valid'].sum()),2)
            def delayed_guard(values):
                valid=torch.zeros(values['qpos'].shape[:2],dtype=torch.bool)
                valid[0,3:]=True
                return valid
            guarded=model.rollout_until(state,context,10,stop_guard=delayed_guard)
            self.assertEqual(guarded['event_found'].tolist(),[True,False])
            self.assertEqual(guarded['event_step'].tolist(),[4,10])
            self.assertEqual(guarded['valid'].sum(-1).tolist(),[4,10])
            torch.testing.assert_close(guarded['next_state'].qpos[0],guarded['qpos'][0,3],rtol=0,atol=0)
            model.decoder[-1].bias[-3]=0;model.decoder[-1].bias[-4]=10
            missing=model.rollout_until(state,context,10)
            self.assertFalse(missing['event_found'].any())
            self.assertTrue((missing['event_step']==10).all())
            self.assertTrue((missing['next_state'].elapsed_steps==10).all())

class HandoffFocusTest(unittest.TestCase):
    def test_rejects_early_event_and_lateral_exit_bias(self):
        from sttw_control.skill_world_model import handoff_focus_loss
        q=torch.zeros(1,4,12);q[...,3]=1;v=torch.zeros(1,4,11)
        phase=torch.tensor([[1,3,4,2]]);onset=phase==2
        correct=torch.full((1,4,5),-4.);correct.scatter_(-1,phase[...,None],4.)
        early=correct.clone();early[...,2]=4.
        good,zero=handoff_focus_loss(q,v,correct,q,v,phase,onset)
        bad,_=handoff_focus_loss(q,v,early,q,v,phase,onset)
        self.assertGreater(bad.item(),good.item());self.assertEqual(zero.item(),0.)
        shifted=q.clone();shifted[:,-1,1]=.05
        _,bias=handoff_focus_loss(shifted,v,correct,q,v,phase,onset)
        self.assertAlmostEqual(bias.item(),1/3,places=6)
        opposite=q.clone();opposite[...,3:7]*=-1
        _,equivalent=handoff_focus_loss(opposite,v,correct,q,v,phase,onset)
        self.assertEqual(equivalent.item(),0.)

if __name__=='__main__':unittest.main()
