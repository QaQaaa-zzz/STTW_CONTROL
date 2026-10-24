from sttw_control.temporal_reference_evidence import pair_transitions


def row(label,safe=False,task=False):
    return dict(balance_label=label,eligible_no_contact=safe,joint_success=safe and task)


def test_unknown_or_unexposed_is_not_a_fall_or_confirmed_joint_loss():
    safe=row('NO_FLOOR_CONTACT',True,True)
    for uncertain in [row('UNKNOWN'),row('ENGINEERING_ERROR'),row('NO_FLOOR_CONTACT')]:
        for a,b in [(uncertain,safe),(safe,uncertain)]:
            result=pair_transitions(a,b)
            assert not result['safe_gain'] and not result['safe_loss']
            assert not result['joint_gain'] and not result['joint_loss']
            assert result['unresolved_pair']


def test_observed_contact_and_full_safe_define_signed_physical_transitions():
    contact=row('FLOOR_CONTACT');safe=row('NO_FLOOR_CONTACT',True,False)
    assert pair_transitions(contact,safe)['safe_gain']
    assert pair_transitions(safe,contact)['safe_loss']
    assert not pair_transitions(contact,safe)['unresolved_pair']


def test_task_improvement_among_known_safe_is_separate_from_physical_gain():
    result=pair_transitions(row('NO_FLOOR_CONTACT',True,False),row('NO_FLOOR_CONTACT',True,True))
    assert result['joint_gain'] and not result['safe_gain'] and not result['safe_loss']
