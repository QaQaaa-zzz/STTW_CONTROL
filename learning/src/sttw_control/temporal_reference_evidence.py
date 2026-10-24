"""Confirmed physical transitions, separate from missing or task outcomes."""


def pair_transitions(baseline,candidate):
    base_safe=bool(baseline['eligible_no_contact'])
    candidate_safe=bool(candidate['eligible_no_contact'])
    base_contact=baseline['balance_label']=='FLOOR_CONTACT'
    candidate_contact=candidate['balance_label']=='FLOOR_CONTACT'
    known=(base_safe or base_contact) and (candidate_safe or candidate_contact)
    return dict(safe_gain=base_contact and candidate_safe,
                safe_loss=base_safe and candidate_contact,
                joint_gain=bool(known and not baseline['joint_success'] and candidate['joint_success']),
                joint_loss=bool(known and baseline['joint_success'] and not candidate['joint_success']),
                unresolved_pair=not known)
