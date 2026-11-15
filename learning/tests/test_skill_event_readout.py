import unittest
import numpy as np
import torch
from sttw_control.skill_event_readout import event_labels,SkillEventReadout

class SkillEventReadoutTest(unittest.TestCase):
    def test_short_event_between_model_samples_is_preserved(self):
        labels=event_labels(np.array([0.,.02,.04,.06]),[.025,float('nan')])
        np.testing.assert_array_equal(labels,np.array([[0,0],[0,0],[1,0],[1,0]]))

    def test_sequence_continuation_preserves_event_memory(self):
        torch.manual_seed(1);model=SkillEventReadout(torch.zeros(4),torch.ones(4));x=torch.randn(2,100,4);initial=torch.zeros(2,2)
        full,_=model.predict(x,initial);first,memory=model.predict(x[:,:40],initial);last,end=model.predict(x[:,40:],memory)
        torch.testing.assert_close(full,torch.cat((first,last),1),rtol=0,atol=0)
        self.assertTrue((full[:,1:]>=full[:,:-1]).all())

    def test_world_and_event_serialized_continuation(self):
        import io
        from sttw_control.skill_world_model import SkillWorldModel
        from sttw_control.skill_event_readout import EventWorldModel
        torch.manual_seed(2);size=12+11+7+3+5+4
        world=SkillWorldModel(7,4,torch.zeros(size),torch.ones(size),torch.ones(11)*.01,torch.ones(7)*.01)
        model=EventWorldModel(world,SkillEventReadout(torch.zeros(152),torch.ones(152)))
        q=torch.zeros(2,12);q[:,3]=1;context=torch.zeros(2,4)
        state=model.initialize(q,torch.zeros(2,11),torch.zeros(2,4),torch.zeros(2,3),torch.zeros(2,3),torch.zeros(2,5),context)
        with torch.no_grad():
            full=model.rollout(state,context,100);first=model.rollout(state,context,40)
            buffer=io.BytesIO();torch.save(first['next_state'],buffer);buffer.seek(0)
            last=model.rollout(torch.load(buffer,weights_only=False),context,60)
        torch.testing.assert_close(full['event_probability'],torch.cat((first['event_probability'],last['event_probability']),1),rtol=0,atol=0)
        torch.testing.assert_close(full['qpos'],torch.cat((first['qpos'],last['qpos'][:,1:]),1),rtol=0,atol=0)

if __name__=='__main__':unittest.main()

