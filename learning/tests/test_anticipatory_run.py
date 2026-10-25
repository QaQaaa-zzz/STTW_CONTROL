import numpy as np
import pytest
from sttw_control.anticipatory_run import flatten_ticks, physical_outcome, stage_indices, claim_stage


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
