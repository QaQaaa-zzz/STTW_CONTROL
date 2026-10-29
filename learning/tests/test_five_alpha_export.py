"""Authentication/transport tests only; no controller execution or fitting."""
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from sttw_control.tolerance_coordination import load_five_alpha_export,load_export,infer_probabilities,sha


def write(path,obj):path.write_text(json.dumps(obj))


@pytest.fixture
def artifact(tmp_path):
    cfg=dict(architecture=dict(hidden=[16,16]),training=dict(seeds=[0,1,2],fixed_last=True,updates_per_model_seed=2),features=dict(full_dim=41,full_indices=list(range(300,320))+list(range(337,351))))
    analysis=tmp_path/'data/analysis';analysis.mkdir(parents=True)
    features=np.zeros((5,41),np.float32);features[:,34]=[0,.25,.5,.75,1];features[:,39]=1.5
    np.savez(analysis/'features_labels.npz',features=features)
    write(analysis.parent/'plan.json',dict(protocol=dict(duration_s=.15,candidate_alphas=[0,.25,.5,.75,1],source_groups=[dict(source_user_alpha=.5)])))
    cfg['dataset']=dict(analyses=[str(analysis)]);write(tmp_path/'config.json',cfg)
    receipt=dict(inputs={str(f):sha(f) for f in [analysis/'features_labels.npz',analysis.parent/'plan.json']});write(tmp_path/'data_receipt.json',receipt)
    params={};selected=[]
    for seed in range(3):
        sd={}
        for i,(layer,ins,outs) in enumerate([(0,41,16),(2,16,16),(4,16,1)]):
            sd[f'{layer}.weight']=torch.full((outs,ins),.01*(seed+1));sd[f'{layer}.bias']=torch.zeros(outs)
            params.setdefault(f'w{i}',[]).append(sd[f'{layer}.weight'].numpy().T);params.setdefault(f'b{i}',[]).append(sd[f'{layer}.bias'].numpy())
        checkpoint=tmp_path/f'seed{seed}.pt';state=dict(state_dict=sd,dim=41,seed=seed,step=2,arm='bce_rank',selector='last');torch.save(state,checkpoint)
        selected.append(dict(file=str(checkpoint),sha256=sha(checkpoint),**{k:v for k,v in state.items() if k!='state_dict'}))
    np.savez(tmp_path/'parameters.npz',**{k:np.stack(v)[None] for k,v in params.items()})
    manifest=dict(kind='five_alpha_fixed_last',feature_dimension=41,candidate_alpha_grid=[0,.25,.5,.75,1],selected=selected,config=str(tmp_path/'config.json'),config_sha256=sha(tmp_path/'config.json'),data_receipt_sha256=sha(tmp_path/'data_receipt.json'))
    write(tmp_path/'selection_manifest.json',manifest);write(tmp_path/'selection_seal.json',dict(selection_manifest_sha256=sha(tmp_path/'selection_manifest.json'),selected_sha256={r['file']:r['sha256'] for r in selected}))
    write(tmp_path/'export_manifest.json',dict(parameters_sha256=sha(tmp_path/'parameters.npz'),selection_manifest_sha256=sha(tmp_path/'selection_manifest.json'),candidate_alpha_grid=[0,.25,.5,.75,1],source_user_alpha=.5,arms=['bce_rank_last']))
    return tmp_path


def test_explicit_transport_loads_single_arm_and_leaves_legacy_default(artifact):
    p=load_five_alpha_export(artifact,expected_export_sha256=sha(artifact/'export_manifest.json'))
    assert np.asarray(infer_probabilities(p,np.zeros((5,41),np.float32))).shape==(1,5)
    with pytest.raises(FileNotFoundError):load_export(artifact)


@pytest.mark.parametrize('name',['parameters.npz','selection_manifest.json','selection_seal.json','seed1.pt','config.json','data_receipt.json','data/analysis/features_labels.npz'])
def test_reject_tampered_chain(artifact,name):
    frozen=sha(artifact/'export_manifest.json');file=artifact/name
    if name=='selection_seal.json':
        obj=json.loads(file.read_text());obj['selection_manifest_sha256']='0'*64;write(file,obj)
    else:file.write_bytes(file.read_bytes()+b' ')
    with pytest.raises((ValueError,json.JSONDecodeError)):load_five_alpha_export(artifact,expected_export_sha256=frozen)


@pytest.mark.parametrize('kwargs',[dict(expected_export_sha256='0'*64),dict(candidate_alphas=(0.,.5,1.)),dict(hypothetical_duration_s=.1)])
def test_reject_wrong_explicit_identity_and_runtime_semantics(artifact,kwargs):
    kw=dict(expected_export_sha256=sha(artifact/'export_manifest.json'));kw.update(kwargs)
    with pytest.raises(ValueError):load_five_alpha_export(artifact,**kw)


def test_rehashed_export_still_must_equal_selected_checkpoint(artifact):
    with np.load(artifact/'parameters.npz') as z:p={k:z[k] for k in z.files}
    p['b2']=p['b2']+.5;np.savez(artifact/'parameters.npz',**p)
    ex=json.loads((artifact/'export_manifest.json').read_text());ex['parameters_sha256']=sha(artifact/'parameters.npz');write(artifact/'export_manifest.json',ex)
    with pytest.raises(ValueError,match='weights differ'):load_five_alpha_export(artifact,expected_export_sha256=sha(artifact/'export_manifest.json'))


@pytest.mark.parametrize('mutation',['nonfinite','wrong_shape'])
def test_reject_rehashed_nonfinite_or_wrong_layout(artifact,mutation):
    with np.load(artifact/'parameters.npz') as z:p={k:z[k] for k in z.files}
    if mutation=='nonfinite':p['w0'][0,0,0,0]=np.nan
    else:p['w0']=p['w0'][...,:40,:]
    np.savez(artifact/'parameters.npz',**p)
    ex=json.loads((artifact/'export_manifest.json').read_text());ex['parameters_sha256']=sha(artifact/'parameters.npz');write(artifact/'export_manifest.json',ex)
    with pytest.raises(ValueError,match='layout/dtype/finiteness'):load_five_alpha_export(artifact,expected_export_sha256=sha(artifact/'export_manifest.json'))
