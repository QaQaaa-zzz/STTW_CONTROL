"""Existing online two-wheel platform handoff predicate, also usable for shaping."""

def platform_support_mask(wheel_positions, contact_forces, roll, pitch, front, back, contact_mask=None):
    on=((wheel_positions[:,:,0]>front)&(wheel_positions[:,:,0]<back)
        &(wheel_positions[:,:,1].abs()<.9)&((wheel_positions[:,:,2]-.23).abs()<.025)).all(-1)
    contact=(contact_forces[:,:,2]>1.) if contact_mask is None else contact_mask
    return on&contact.all(-1)&(roll.abs()<.15)&(pitch.abs()<.15)
