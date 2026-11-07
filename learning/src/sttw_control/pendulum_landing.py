"""Existing online two-wheel platform handoff predicate, also usable for shaping."""

def platform_support_mask(wheel_positions, contact_forces, roll, pitch, front, back):
    on=((wheel_positions[:,:,0]>front)&(wheel_positions[:,:,0]<back)
        &(wheel_positions[:,:,1].abs()<.9)&((wheel_positions[:,:,2]-.23).abs()<.025)).all(-1)
    return on&(contact_forces[:,:,2]>1.).all(-1)&(roll.abs()<.15)&(pitch.abs()<.15)
