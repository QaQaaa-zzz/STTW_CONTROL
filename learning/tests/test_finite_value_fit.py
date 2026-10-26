import torch
from sttw_control.finite_value_fit import FiniteValue,discount_budget


def base():
    return torch.nn.Sequential(torch.nn.Linear(352,8),torch.nn.ELU(),torch.nn.Linear(8,8),torch.nn.ELU(),torch.nn.Linear(8,8),torch.nn.ELU(),torch.nn.Linear(8,1))


def test_discount_budget_has_exact_terminal_and_known_last_step():
    got=discount_budget(torch.tensor([0.,1/500,1.]),gamma=.997,steps=500)
    torch.testing.assert_close(got[:2],torch.tensor([0.,1.]),atol=1e-5,rtol=1e-5)
    assert 250<float(got[2])<260


def test_matched_initial_functions_keep_source_weights_and_boundary_distinct():
    source=base();before={k:v.clone() for k,v in source.state_dict().items()}
    x=torch.randn(7,352);x[:,-1]=torch.tensor([0.,.002,.1,.3,.5,.8,1.])
    models=[FiniteValue(source,m,rate=.15) for m in ['reset_free','time_free','time_nonpositive']]
    for model in models[1:]:torch.testing.assert_close(model(x),models[0](x),atol=1e-5,rtol=1e-6)
    for k,v in source.state_dict().items():torch.testing.assert_close(v,before[k])
    with torch.no_grad():models[2].network[6].bias.fill_(2.)
    assert torch.all(models[2](x)<=0) and models[2](x)[0].item()==0
    assert models[1](x)[0].item()==0
