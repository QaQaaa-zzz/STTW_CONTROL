"""Descriptive finite-grid physical outcomes, never continuous safety bounds."""
import math


def directional_grid_summary(magnitudes, outcomes):
    """Summarize one signed direction, given actual zero and increasing queries.

    Unknowns interrupt evidence, and a larger successful force never fills an
    earlier failure. A prefix consists solely of the supplied measured points.
    It does not assert that intervening untested amplitudes are safe.
    """
    f=list(magnitudes); y=list(outcomes)
    if (not f or len(f)!=len(y) or f[0]!=0 or
            not all(math.isfinite(v) for v in f) or
            any(b<=a for a,b in zip(f,f[1:])) or
            not set(y)<={'survived','contact','unknown'}):
        raise ValueError('zero followed by strictly increasing finite magnitudes and physical outcomes required')
    stop=next((i for i,v in enumerate(y) if v!='survived'),len(y))
    passes=[v for v,o in zip(f,y) if o=='survived']
    contacts=[i for i,o in enumerate(y) if o=='contact']
    return dict(zero_force_survived=y[0]=='survived',
        all_pass_grid_prefix_N=f[stop-1] if stop else None,
        first_nonpass_N=f[stop] if stop<len(y) else None,
        first_nonpass_outcome=y[stop] if stop<len(y) else None,
        largest_observed_pass_N=max(passes) if passes else None,
        first_observed_contact_N=f[contacts[0]] if contacts else None,
        right_censored=stop==len(y),
        nonmonotone_observed=any(o=='survived' for i in contacts for o in y[i+1:]),
        measured_count=len(f),
        scope='Supplied measured force grid only; no interpolation or continuous safety certification.')
