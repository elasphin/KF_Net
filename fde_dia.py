"""Fault detection, identification and exclusion (Yan Sec. II-D, Ref. [33]),
DIA adaptation (Yan Eq. (34)) and SBAS-style integrity (HPL/VPL, Stanford).

All tests run on the clock-projected innovation of `measurement_model`; a raw
pseudorange fault e_j appears there as projector @ e_j.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.stats import chi2

from navigation_models import clock_observable, ecef_llh, ecef_to_ned

# Standard SBAS/APV protection-level multipliers (Yan gives no K factors).
K_H, K_V = 6.0, 5.33
EPS = np.finfo(float).eps
TINY = np.finfo(float).tiny


def _sym(M) -> np.ndarray:
    M = np.asarray(M, dtype=float)
    return 0.5 * (M + M.T)


def psd_pinv(Q) -> tuple[np.ndarray, int]:
    """Moore-Penrose inverse and numerical rank of a PSD matrix (Q_nu_nu)."""
    Q = _sym(Q)
    if Q.ndim != 2 or Q.shape[0] != Q.shape[1]:
        raise ValueError('Q_nu_nu must be square')
    if Q.size == 0:
        return np.zeros_like(Q), 0
    if not np.all(np.isfinite(Q)):
        raise FloatingPointError('Q_nu_nu contains non-finite values')
    w, V = np.linalg.eigh(Q)
    tol = 100.0 * EPS * max(Q.shape) * max(float(np.max(np.abs(w))), 1.0)
    if float(np.min(w)) < -100.0 * tol:
        raise FloatingPointError('Q_nu_nu is materially indefinite in Ref. [33] FDE')
    keep = w > tol
    rank = int(np.count_nonzero(keep))
    if rank == 0:
        return np.zeros_like(Q), 0
    B = V[:, keep]
    return _sym(B / w[keep] @ B.T), rank


def _check_psd(M, name) -> np.ndarray:
    M = _sym(M)
    if not np.all(np.isfinite(M)):
        raise FloatingPointError(f'{name} contains non-finite values')
    if M.size == 0:
        return M
    w = np.linalg.eigvalsh(M)
    if float(np.min(w)) < -1000.0 * EPS * max(M.shape) * max(float(np.max(np.abs(w))), 1.0):
        raise FloatingPointError(f'{name} is materially indefinite: min eigenvalue={float(np.min(w)):.6g}')
    return M


def _projector(model, n) -> np.ndarray:
    P = np.asarray(getattr(model, 'projector', np.eye(n)), dtype=float)
    if P.shape != (n, n):
        raise ValueError('clock_projector shape is inconsistent with innovation')
    if not np.all(np.isfinite(P)):
        raise FloatingPointError('clock_projector contains non-finite values')
    return P


def _fault_direction(Q_pinv, c):
    """Return (c^T Q+ c, estimable?) for fault direction c."""
    den = float(c @ Q_pinv @ c)
    tol = 100.0 * EPS * max(float(np.linalg.norm(Q_pinv, ord=2)) * float(c @ c), TINY)
    return den, math.isfinite(den) and den > tol


# --- Detection ---------------------------------------------------------------
@dataclass(frozen=True)
class Detection:
    detected: bool
    statistic: float
    threshold: float
    dof: int
    alpha: float
    Q_nu_nu: np.ndarray


def detect_fault(P_prior, model, alpha) -> Detection:
    """Overall model test: nu^T Q+ nu > chi2(1-alpha, rank Q), Q = H P H^T + R."""
    alpha = float(alpha)
    if not 0.0 < alpha < 1.0:
        raise ValueError('FDE significance alpha must lie strictly between 0 and 1')
    H, R, nu = (np.asarray(x, dtype=float) for x in (model.H, model.R, model.innovation))
    Q = _sym(H @ np.asarray(P_prior, dtype=float) @ H.T + R)
    Q_pinv, dof = psd_pinv(Q)
    if dof == 0:
        return Detection(False, 0.0, float('inf'), 0, alpha, Q.copy())
    stat = float(nu @ Q_pinv @ nu)
    thr = float(chi2.ppf(1.0 - alpha, df=dof))
    if not math.isfinite(stat) or not math.isfinite(thr):
        raise FloatingPointError('non-finite Ref. [33] FDE statistic/threshold')
    return Detection(bool(stat > thr), stat, thr, dof, alpha, Q.copy())


# --- Identification ----------------------------------------------------------
@dataclass(frozen=True)
class Identification:
    identified: bool
    index: int                    # row of the faulty pseudorange, -1 if none
    sat_id: str
    constellation: str
    local_statistic: float
    local_score: float
    fault_m: float                # estimated pseudorange fault
    ambiguous_sat_ids: tuple
    candidate_count: int


def identify_fault(model, Q) -> Identification:
    """w-test: pick the row with maximum T_j = (c_j^T Q+ nu)^2 / c_j^T Q+ c_j.

    Ties or statistically identical directions are reported as ambiguous.
    """
    nu = np.asarray(model.innovation, dtype=float).reshape(-1)
    sat_ids, groups = tuple(model.sat_ids), tuple(model.constellations)
    if len(sat_ids) != len(nu) or len(groups) != len(nu):
        raise ValueError('measurement metadata is inconsistent with innovation')
    Q_pinv, _ = psd_pinv(Q)
    Pc = _projector(model, len(nu))

    cands = []
    for j in range(len(nu)):
        c = Pc[:, j].copy()
        den, ok = _fault_direction(Q_pinv, c)
        if not ok:
            continue
        num = float(c @ Q_pinv @ nu)
        T = float(num * num / den)
        score, fault = float(chi2.cdf(T, df=1)), float(num / den)
        if not (math.isfinite(T) and math.isfinite(score) and math.isfinite(fault)):
            raise FloatingPointError('non-finite Ref. [33] identification term')
        cands.append((j, c, den, T, score, fault))
    if not cands:
        return Identification(False, -1, '', '', 0.0, 0.0, 0.0, (), 0)

    # Rank by T (chi2.cdf saturates at 1.0 for large faults).
    j_best, c_best, den_best, T_best, score_best, fault_best = max(cands, key=lambda x: x[3])
    stat_tol = 1000.0 * EPS * max(1.0, abs(T_best))
    sub_tol = 1000.0 * EPS * max(1, len(nu))
    ambiguous = []
    for j, c, den, T, _, _ in cands:
        cosine = abs(float(c_best @ Q_pinv @ c)) / math.sqrt(max(den_best * den, TINY))
        if 1.0 - min(max(cosine, 0.0), 1.0) <= sub_tol or abs(T - T_best) <= stat_tol:
            ambiguous.append(j)
    ambiguous = tuple(sorted(set(ambiguous)))
    if len(ambiguous) != 1:
        return Identification(False, -1, '', '', T_best, score_best, fault_best,
                              tuple(sat_ids[j] for j in ambiguous), len(cands))
    return Identification(True, j_best, str(sat_ids[j_best]), str(groups[j_best]), T_best, score_best,
                          fault_best, (), len(cands))


# --- Exclusion ---------------------------------------------------------------
@dataclass(frozen=True)
class Exclusion:
    applied: bool
    excluded_index: int
    excluded_sat_id: str
    retained: tuple
    dropped_sat_ids: tuple        # clock-group partners left as singletons


def exclude_fault(meas, ident: Identification) -> Exclusion:
    """Remove the identified row, then any clock group left with one member."""
    meas = tuple(meas)
    if not ident.identified:
        return Exclusion(False, -1, '', meas, ())
    j = int(ident.index)
    if not 0 <= j < len(meas):
        raise IndexError('identified pseudorange index lies outside measurements')
    remaining = tuple(m for i, m in enumerate(meas) if i != j)
    retained = clock_observable(remaining)
    kept = {id(m) for m in retained}
    return Exclusion(True, j, str(meas[j].sat_id), retained,
                     tuple(str(m.sat_id) for m in remaining if id(m) not in kept))


# --- DIA adaptation (Yan Eq. (34)); not used by the pipeline ------------------
@dataclass(frozen=True)
class DIAAdaptation:
    available: bool
    index: int
    sat_id: str
    x0: np.ndarray                # nominal KF correction under H0
    P0: np.ndarray                # nominal posterior covariance
    x: np.ndarray                 # adapted correction x_i
    P: np.ndarray                 # adapted covariance P_i
    dx: np.ndarray                # -L_i nu
    dP: np.ndarray                # L_i Q L_i^T
    L: np.ndarray
    fault_direction: np.ndarray
    fault_m: float


def adapt(P_prior, model, Q, ident: Identification) -> DIAAdaptation:
    """x_i = x0 - L_i nu, P_i = P0 + L_i Q L_i^T with L_i = K0 c_i c_i^+ (Ref. [33] Eq. (39))."""
    P = np.asarray(P_prior, dtype=float)
    H, R = np.asarray(model.H, dtype=float), np.asarray(model.R, dtype=float)
    nu = np.asarray(model.innovation, dtype=float).reshape(-1)
    n_x, n_y = (P.shape[0] if P.ndim == 2 else 0), len(nu)
    if P.ndim != 2 or P.shape[0] != P.shape[1]:
        raise ValueError('prior covariance must be square')
    if H.shape != (n_y, n_x) or R.shape != (n_y, n_y):
        raise ValueError('measurement model shape is inconsistent with state/innovation')
    P, R = _check_psd(P, 'Ref. [33] prior P_minus'), _check_psd(R, 'Ref. [33] measurement R')
    if not ident.identified:
        zx, zP = np.zeros(n_x), np.zeros((n_x, n_x))
        return DIAAdaptation(False, -1, '', zx, zP, zx.copy(), zP.copy(), zx.copy(), zP.copy(),
                             np.zeros((n_x, n_y)), np.zeros(n_y), 0.0)
    j = int(ident.index)
    if not 0 <= j < n_y:
        raise IndexError('identified pseudorange index lies outside the measurement vector')
    Q = np.asarray(Q, dtype=float)
    if Q.shape != (n_y, n_y):
        raise ValueError('Q_nu_nu shape is inconsistent with innovation')
    Q_pinv, rank = psd_pinv(Q)
    if rank == 0:
        raise FloatingPointError('DIA adaptation cannot be formed from rank-zero Q_nu_nu')

    K0 = P @ H.T @ Q_pinv
    x0 = K0 @ nu
    A = np.eye(n_x) - K0 @ H
    P0 = _check_psd(A @ P @ A.T + K0 @ R @ K0.T, 'Ref. [33] nominal P0_plus')
    c = _projector(model, n_y)[:, [j]].copy()
    den = float((c.T @ Q_pinv @ c).item())
    tol = 100.0 * EPS * max(float(np.linalg.norm(Q_pinv, ord=2)) * float((c.T @ c).item()), TINY)
    if not math.isfinite(den) or den <= tol:
        raise FloatingPointError('identified fault direction is not estimable in Q_nu_nu')
    c_plus = (c.T @ Q_pinv) / den
    L = K0 @ c @ c_plus
    fault = float((c_plus @ nu).item())
    if not math.isfinite(fault):
        raise FloatingPointError('non-finite DIA pseudorange-fault estimate')
    if abs(fault - float(ident.fault_m)) > 1e-10 * max(1.0, abs(float(ident.fault_m)), abs(fault)):
        raise RuntimeError('Phase-3 identification and Phase-4 DIA fault estimates disagree')
    dx = -(L @ nu)
    dP = _check_psd(L @ Q @ L.T, 'Yan Eq. (34) DIA covariance increment')
    P_i = _check_psd(P0 + dP, 'Yan Eq. (34) DIA posterior covariance')
    x = x0 + dx
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(L)):
        raise FloatingPointError('non-finite Yan Eq. (34) DIA state/L_i')
    return DIAAdaptation(True, j, str(ident.sat_id), x0.copy(), P0.copy(), x.copy(), P_i.copy(), dx.copy(),
                         dP.copy(), L.copy(), c.reshape(-1).copy(), fault)


# --- Integrity (not used by the pipeline) ------------------------------------
STANFORD_LABELS = ('NO', 'MI', 'HO', 'SU', 'SU&MI', 'UNRESOLVED')


def protection_levels(P, pos) -> tuple[float, float]:
    """HPL = K_H sqrt(major horizontal variance), VPL = K_V sigma_down."""
    P = np.asarray(P, dtype=float)
    pos = np.asarray(pos, dtype=float).reshape(3)
    if P.shape[0] < 3 or P.shape[1] < 3:
        raise ValueError('covariance must contain at least a 3x3 position block')
    if not np.all(np.isfinite(P[:3, :3])) or not np.all(np.isfinite(pos)):
        return float('inf'), float('inf')
    lat, lon, _ = ecef_llh(pos)
    C = ecef_to_ned(lat, lon)
    Pn = _sym(C @ P[:3, :3] @ C.T)
    if not np.all(np.isfinite(Pn)):
        return float('inf'), float('inf')
    pnn, pee, pdd, pne = float(Pn[0, 0]), float(Pn[1, 1]), float(Pn[2, 2]), float(Pn[0, 1])
    major = 0.5 * pnn + 0.5 * pee + math.hypot(0.5 * pnn - 0.5 * pee, pne)
    hpl = K_H * math.sqrt(max(major, 0.0)) if math.isfinite(major) else float('inf')
    vpl = K_V * math.sqrt(max(pdd, 0.0)) if math.isfinite(pdd) else float('inf')
    return float(hpl), float(vpl)


def stanford_category(pe, pl, al, available=True) -> str:
    """NO: PE<=PL<=AL, MI: PL<PE<=AL, HO: PL<=AL<PE, SU: AL<PL, PE<=PL, SU&MI: AL<PL<PE."""
    pe, pl, al = float(pe), float(pl), float(al)
    if al <= 0.0 or not math.isfinite(al):
        raise ValueError('alert limit must be finite and positive')
    if not available or not math.isfinite(pe) or not math.isfinite(pl) or pe < 0.0 or pl < 0.0:
        return 'UNRESOLVED'
    if pl > al:
        return 'SU&MI' if pe > pl else 'SU'
    if pe > al:
        return 'HO'
    return 'MI' if pe > pl else 'NO'


def stanford_summary(pe, pl, al, available=None):
    pe = np.asarray(pe, dtype=float).reshape(-1)
    pl = np.asarray(pl, dtype=float).reshape(-1)
    if pe.shape != pl.shape:
        raise ValueError('position error and protection level must have equal length')
    ok = np.ones(pe.shape, bool) if available is None else np.asarray(available, dtype=bool).reshape(-1)
    if ok.shape != pe.shape:
        raise ValueError('integrity_available must match the position-error length')
    cat = np.asarray([stanford_category(e, p, al, a) for e, p, a in zip(pe, pl, ok)], dtype=str)
    counts = {k: int(np.count_nonzero(cat == k)) for k in STANFORD_LABELS}
    total = len(cat)
    classified = total - counts['UNRESOLVED']
    pct = {k: (100.0 * counts[k] / total if total else 0.0) for k in STANFORD_LABELS}
    usable = np.isin(cat, ('NO', 'MI', 'HO'))
    bounded = np.isin(cat, ('NO', 'SU'))
    return cat, {
        'alert_limit_m': float(al), 'total_epochs': total, 'classified_epochs': classified,
        'counts': counts, 'percent_of_all_epochs': pct,
        'nominal_operation_percent_of_all': pct['NO'],           # Yan Fig. 20 "safe operation"
        'pl_below_alert_limit_percent_of_all': 100.0 * float(np.count_nonzero(usable)) / total if total else 0.0,
        'unresolved_percent_of_all': pct['UNRESOLVED'],
        'protection_containment_percent_of_classified':
            100.0 * float(np.count_nonzero(bounded)) / classified if classified else 0.0,
    }
