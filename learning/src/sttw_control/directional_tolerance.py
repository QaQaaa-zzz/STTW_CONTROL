#!/usr/bin/env python3
"""Directional tolerance primitives adapted from the submitted 2026-10-22 derivation.

No MuJoCo, JAX, checkpoint loading, training, or robot simulation is performed.
Examples are hypothetical linear systems, NOT measured STTW force tolerances.
Run: python verify_directional_tolerance.py --output verification_results.json
Dependencies: numpy, scipy, sympy.
"""
from __future__ import annotations
import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import quad_vec, solve_ivp
from scipy.linalg import expm, solve
import sympy as sy

Array = NDArray[np.float64]

@dataclass(frozen=True)
class MarginResult:
    plus: float
    minus: float
    a: float
    b: float
    c: float
    budget: float
    rank: int
    condition_on_range: float
    status: str


def return_gramian(A: ArrayLike, B: ArrayLike, R: ArrayLike, T: float) -> Array:
    """Initial-coordinate weighted return Gramian; R is fixed as T changes."""
    A, B, R = (np.asarray(v, dtype=float) for v in (A, B, R))
    if T <= 0 or not math.isfinite(T):
        raise ValueError('T must be finite and positive')
    n = A.shape[0]
    if A.shape != (n, n) or B.ndim != 2 or B.shape[0] != n:
        raise ValueError('A/B dimensions do not match')
    if R.shape != (B.shape[1], B.shape[1]):
        raise ValueError('R dimensions do not match B')
    if not all(np.isfinite(v).all() for v in (A, B, R)):
        raise ValueError('nonfinite matrix')
    if not np.allclose(R, R.T):
        raise ValueError('R must be symmetric')
    np.linalg.cholesky(R)
    Q = B @ solve(R, B.T, assume_a='pos')
    def integrand(t: float) -> Array:
        E = expm(-A*t)
        return E @ Q @ E.T
    W, _ = quad_vec(integrand, 0., T, epsabs=1e-11, epsrel=1e-11)
    return .5*(W + W.T)


def _range_solver(W: ArrayLike, rtol: float = 1e-11):
    """Rank-aware solve. Never silently regularizes an unreachable component."""
    W = np.asarray(W, dtype=float)
    if W.ndim != 2 or W.shape[0] != W.shape[1] or not np.isfinite(W).all():
        raise ValueError('finite square W required')
    if not np.allclose(W, W.T, rtol=1e-9, atol=1e-13):
        raise ValueError('W is not symmetric')
    w, V = np.linalg.eigh(.5*(W + W.T))
    scale = max(float(np.max(np.abs(w))), np.finfo(float).tiny)
    if w.min() < -rtol*scale:
        raise ValueError('W is not positive semidefinite')
    keep = w > rtol*scale
    U, d = V[:, keep], w[keep]
    rank = int(keep.sum())
    condition = float(d.max()/d.min()) if rank else math.inf
    def reachable(x: Array) -> bool:
        return np.linalg.norm(x - U@(U.T@x)) <= 1e-8*max(np.linalg.norm(x), 1e-12)
    def psolve(x: Array) -> Array:
        if not reachable(x):
            raise ValueError('vector outside reachable subspace')
        return U@((U.T@x)/d) if rank else np.zeros_like(x)
    return rank, condition, reachable, psolve


def impulse_margins(W: ArrayLike, e: ArrayLike, D: ArrayLike,
                    budget: float) -> MarginResult:
    """Exact margins for the stated unconstrained linear quadratic problem.

    e and D must use consistent scaled state coordinates. D maps a scalar
    impulse to state; the returned unit is the impulse unit used to define D.
    Infinity only means NO_EFFECT_IN_SELECTED_MODEL, not infinite real safety.
    """
    e, D = np.asarray(e, float).reshape(-1), np.asarray(D, float).reshape(-1)
    W = np.asarray(W, float)
    if W.shape != (e.size, e.size) or D.shape != e.shape:
        raise ValueError('dimension mismatch')
    if not np.isfinite(e).all() or not np.isfinite(D).all():
        raise ValueError('nonfinite state/direction')
    if not math.isfinite(budget) or budget < 0:
        raise ValueError('finite nonnegative budget required')
    rank, cond, in_range, psolve = _range_solver(W)
    def result(p: float, m: float, a: float, b: float, c: float, status: str):
        return MarginResult(p, m, a, b, c, budget, rank, cond, status)
    # A numerical rank cutoff is not a structural unreachability proof.
    if rank < e.size:
        return result(math.nan, math.nan, math.nan, math.nan, math.nan, 'UNKNOWN_NUMERICAL')
    if not in_range(e):
        return result(0., 0., math.nan, math.nan, math.inf, 'BASE_UNREACHABLE')
    Pe = psolve(e)
    c = float(e@Pe)
    if c > budget:
        return result(0., 0., math.nan, math.nan, c, 'BUDGET_INSUFFICIENT')
    if not in_range(D):
        return result(0., 0., math.inf, math.nan, c, 'DIRECTION_UNREACHABLE')
    PD = psolve(D)
    a, b = float(D@PD), float(D@Pe)
    if np.linalg.norm(D) == 0.:
        return result(math.inf, math.inf, a, b, c, 'NO_EFFECT_IN_SELECTED_MODEL')
    if a <= 0.:
        return result(0., 0., a, b, c, 'UNKNOWN_NUMERICAL')
    gap = max(0., budget-c)
    root = math.sqrt(max(0., b*b + a*gap))
    def endpoint(sigma: int) -> float:
        t = sigma*b
        # Rationalized root avoids subtracting two nearly equal quantities.
        if t >= 0.:
            denominator = root+t
            return gap/denominator if denominator > 0. else 0.
        return (root-t)/a
    return result(endpoint(+1), endpoint(-1), a, b, c, 'MODEL_ONLY')


def pulse_map_initial(A: ArrayLike, D: ArrayLike,
                      start: float, duration: float) -> Array:
    """K = integral exp(-A*t) D dt; maps force amplitude, not impulse."""
    A, D = np.asarray(A, float), np.asarray(D, float).reshape(-1)
    if start < 0 or duration <= 0:
        raise ValueError('start >= 0 and duration > 0 required')
    K, _ = quad_vec(lambda t: expm(-A*t)@D, start, start+duration,
                    epsabs=1e-12, epsrel=1e-12)
    return K


def local_constraint_margin(g0: ArrayLike, q_sigma: ArrayLike,
                            force_trust: float) -> dict[str, Any]:
    """First-order fixed-policy proxy, not a certified physical lower bound.

    Constraints have form g <= 0. q_sigma is dg/dF for positive F in
    the specified signed direction. Include process AND terminal rows.
    """
    g0, q = np.asarray(g0, float).ravel(), np.asarray(q_sigma, float).ravel()
    if g0.shape != q.shape or not np.isfinite(g0).all() or not np.isfinite(q).all():
        raise ValueError('finite matching constraint arrays required')
    if force_trust <= 0 or not math.isfinite(force_trust):
        raise ValueError('finite positive validity cap required')
    if np.any(g0 > 0.):
        return dict(value=0., status='NOMINAL_FAILURE', active_row=None)
    approaching = q > 0.
    indices = np.flatnonzero(approaching)
    if not len(indices):
        return dict(value=force_trust, status='LOCAL_CENSORED', active_row=None)
    limits = np.maximum(0., -g0[indices])/q[indices]
    index = int(np.argmin(limits))
    raw = float(limits[index])
    return dict(value=min(force_trust, raw),
                status='LOCAL_CENSORED' if raw >= force_trust else 'LOCAL_PROXY',
                active_row=int(indices[index]))

