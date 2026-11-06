import torch
from sttw_control.phase_u_reward import phase_u_episode_reward


def test_failed_episode_total_is_fixed_without_reclipping_terminal_reward():
    # Matches original Phase U: terminal correction can exceed [-400, 50].
    raw=torch.tensor([-400.,-100.,5.]);previous=torch.tensor([700.,-300.,20.])
    reward,total,override=phase_u_episode_reward(raw,previous,torch.tensor([True,False,False]),torch.tensor([False,True,False]),{'failed_episode_return':-100.})
    torch.testing.assert_close(reward,torch.tensor([-800.,200.,5.]))
    torch.testing.assert_close(total,torch.tensor([-100.,-100.,25.]))
    torch.testing.assert_close(override,torch.tensor([-400.,300.,0.]))
