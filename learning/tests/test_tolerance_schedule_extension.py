"""Schedule continuation must not admit task, optimizer, or lineage drift."""
import copy,json
import pytest
from sttw_control.preference_best import config_hash,digest
from sttw_control.tolerance_training import schedule_resume_identity

@pytest.fixture
def extension(tmp_path):
 old={'training_future_phase_C':{'updates_initial':27,'updates_max':27,'actor_lr':3e-5},'reward':{'path':10},'physics':{'dt':.005}}
 identity={'config_sha256':config_hash(old),'seed':4,'parent_checkpoint_sha256':'parent'}
 manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps(dict(config=old,identity=identity)))
 record={'update':27,'schedule_extension':{'source_manifest':str(manifest),'source_manifest_sha256':digest(manifest)}}
 new=copy.deepcopy(old);new['training_future_phase_C'].update(updates_initial=77,updates_max=77)
 return record,dict(identity,config_sha256=config_hash(new)),new,identity

def test_extension_restores_original_identity(extension):
 record,identity,config,old=extension
 assert schedule_resume_identity(record,identity,config)==old
 assert schedule_resume_identity({},identity,None)==identity

@pytest.mark.parametrize('change',['reward','physics','lr','seed','partial','hash'])
def test_extension_rejects_other_changes(extension,change):
 record,identity,config,_=extension
 if change=='reward':config['reward']['path']=11
 if change=='physics':config['physics']['dt']=.01
 if change=='lr':config['training_future_phase_C']['actor_lr']=1e-4
 if change=='seed':identity['seed']=5
 if change=='partial':record['update']=25
 if change=='hash':record['schedule_extension']['source_manifest_sha256']='wrong'
 if change in ['reward','physics','lr']:identity['config_sha256']=config_hash(config)
 with pytest.raises(ValueError):schedule_resume_identity(record,identity,config)
