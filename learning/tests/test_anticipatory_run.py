import numpy as np
import pytest
from sttw_control.anticipatory_run import flatten_ticks, physical_outcome, stage_indices, claim_stage, architecture_groups


def test_flatten_preserves_lane_and_tick_order():
    x=np.arange(3*2*4*2).reshape(3,2,4,2)
    y=flatten_ticks({'x':x})['x']
    assert y.shape==(12,2,2)
    for upper in range(3):
        for lane in range(2):
            for lower in range(4):
                np.testing.assert_array_equal(y[upper*4+lower,lane],x[upper,lane,lower])


def test_fault_is_unknown_even_with_contact_and_missing_ticks_not_survival():
    a=np.ones(4,bool)
    assert physical_outcome(a,[0,1,0,0],[0,0,1,0],4)=='unknown'
    assert physical_outcome(a,[0,1,0,0],np.zeros(4,bool),4)=='contact'
    assert physical_outcome(a,np.zeros(4,bool),np.zeros(4,bool),5)=='unknown'
    assert physical_outcome(a,np.zeros(4,bool),np.zeros(4,bool),4)=='survived'


def test_stage_selection_never_mixes_nominal_and_disturbed():
    rows=[dict(stage='disturbed'),dict(stage='nominal'),dict(stage='nominal')]
    assert stage_indices(rows,'nominal')==[1,2]
    assert stage_indices(rows,'disturbed')==[0]
    with pytest.raises(ValueError):stage_indices(rows,'all')


def test_stage_accepts_only_preregistered_manifest_never_existing_data(tmp_path):
    stage=tmp_path/'nominal';stage.mkdir()
    (stage/'ACTIVE_RUN.json').write_text('{}')
    claim_stage(stage)
    (stage/'status.json').write_text('{}')
    with pytest.raises(FileExistsError):claim_stage(stage)
    fresh=tmp_path/'disturbed';claim_stage(fresh)
    assert fresh.is_dir()


def test_groups_keep_plain_and_modulated_parameters_separate():
    models={'plain':dict(actor_config=dict(network={})),
            'mod':dict(actor_config=dict(network=dict(architecture='alpha_modulated_v1')))}
    rows=[dict(actor_role='plain'),dict(actor_role='mod'),dict(actor_role='plain'),dict(actor_role='mod'),dict(actor_role='mod')]
    groups=architecture_groups(list(range(5)),rows,models,{},batch_size=2)
    assert sorted(i for g in groups for i in g['case_indices'])==list(range(5))
    assert [len(g['case_indices']) for g in groups]==[2,2,1]
    for g in groups:
        assert len({rows[i]['actor_role'] for i in g['case_indices']})==1


def test_endpoints_stage_is_explicit_and_does_not_mix_other_cases():
    assert stage_indices([dict(stage='endpoints'),dict(stage='nominal')],'endpoints')==[0]
