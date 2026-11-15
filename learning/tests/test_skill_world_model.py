import io
import unittest
import torch
from sttw_control.skill_world_model import SkillWorldModel


class SkillWorldModelTest(unittest.TestCase):
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

if __name__=='__main__':unittest.main()
