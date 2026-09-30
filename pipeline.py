"""Training pipeline of the Masked-KalmanNet GNSS/LEO/INS filter (Yan et al., 2026).

    python leo.py        # once: simulate the LEO caches
    python pipeline.py   # train on Data01 -> OUTPUT_DIR/best_model.pt
    python test.py       # recursive test on Data02 -> OUTPUT_DIR/test

One fusion step (training and test): INS propagation -> state-dependent
measurements -> [test only: FDE on the full set] -> network Kalman gain ->
error injection -> lagged features for the next epoch.
"""
from __future__ import annotations

from pathlib import Path
from typing import NamedTuple
import json
import math

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint

import config as cfg
from fde_dia import detect_fault, exclude_fault, identify_fault
from leo import load_cache
from mknet import (DTYPE, FEATURE_VERSION, FIXED_GROUPS, STATE_ORDER, MaskedCLA, TorchNav,
                   clip_grad_norm_float64_, network_input, pad_measurement_model, save_checkpoint)
import mknet
from navigation_models import (ATT_SIGN, C_LIGHT, N_STATE, NavState, clock_observable, clock_projector, ecef_llh,
                               ecef_to_ned,
                               inject_error, joseph_reset, kf_update, mechanize, measurement_model,
                               prepare_measurements, propagate, select_measurements, so3_log, state_diff)
from simulation_data import Dataset, load_dataset

torch.set_default_dtype(DTYPE)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
STATE_MODEL_NAME = 'navigation_9_state'
GAIN_SCALE_FLOOR = 1e-12
FIXED_SCALE_FLOOR = 1e-12


def f64(x) -> torch.Tensor:
    return torch.as_tensor(x, dtype=torch.float64, device=DEVICE)


# --- Fusion epochs -----------------------------------------------------------
class FusionEpoch(NamedTuple):
    time: float                  # GPST [s]
    index: int                   # index into ds.t_fusion / truth arrays
    gnss_epoch: object           # ObsEpoch
    leo_meas: tuple              # cached LEO PseudoObs
    segments: tuple              # IMU steps since the previous usable epoch: ((imu_index, dt), ...)
    accel: np.ndarray            # last IMU sample before this epoch (network feature)
    gyro: np.ndarray
    n_boot: int                  # measurement count on the INS-only bootstrap trajectory


def prepare_fusion_epochs(ds: Dataset, leo_by_epoch) -> tuple[FusionEpoch, ...]:
    """Schedule IMU segments between GNSS epochs; keep epochs usable on an INS-only trajectory."""
    t_fus = np.asarray(ds.t_fusion, dtype=float).reshape(-1)
    t_imu = np.asarray(ds.t_imu, dtype=float).reshape(-1)
    if len(leo_by_epoch) != len(t_fus):
        raise ValueError(f'LEO cache epoch count mismatch: raw fusion count={len(t_fus)}, '
                         f'cache count={len(leo_by_epoch)}')
    if t_imu.size == 0 or not np.all(np.isfinite(t_imu)) or np.any(np.diff(t_imu) <= 0.0):
        raise ValueError('IMU time axis is empty, non-finite, or not strictly increasing')
    if not np.all(np.isfinite(t_fus)) or np.any(np.diff(t_fus) <= 0.0):
        raise ValueError('fusion time axis is empty, non-finite, or not strictly increasing')
    if t_fus.size and (t_fus[0] < t_imu[0] or t_fus[-1] > t_imu[-1]):
        raise ValueError('fusion epochs must lie inside the IMU time span')
    imu_stop = min(max(int(np.searchsorted(t_imu, t_fus[-1], side='left')) + 1, 2), len(t_imu)) if t_fus.size else 1

    nav = ds.nav0.copy()                    # INS-only bootstrap reference
    epochs, segments = [], []
    gyro = np.asarray(ds.gyro[0], dtype=float).reshape(3)
    accel = np.asarray(ds.accel[0], dtype=float).reshape(3)
    k, seg_start = 0, float(t_imu[0])       # k: next fusion epoch
    for i in range(1, imu_stop):
        t_end = float(t_imu[i])
        while True:
            at_fusion = k < len(t_fus) and t_fus[k] <= t_end + 1e-12
            seg_end = float(t_fus[k]) if at_fusion else t_end
            if at_fusion and seg_end < seg_start - 1e-12:
                raise RuntimeError('fusion scheduler encountered a fusion epoch before the current propagation segment')
            if seg_end > seg_start + 1e-12:
                dt = seg_end - seg_start
                gyro = np.asarray(ds.gyro[i], dtype=float).reshape(3)
                accel = np.asarray(ds.accel[i], dtype=float).reshape(3)
                nav = mechanize(nav, gyro, accel, dt)
                segments.append((i, dt))
            if not at_fusion:
                seg_start = t_end
                break
            leo = tuple(leo_by_epoch[k])
            boot = prepare_measurements(ds.gnss, ds.gnss_epochs[k], leo, nav, ds.lever_b, capacity=None)
            if boot:
                ids = [m.sat_id for m in ds.gnss_epochs[k].measurements] + [m.sat_id for m in leo]
                if len(set(ids)) != len(ids):
                    raise ValueError(f'{seg_end}: duplicate satellite identifier')
                epochs.append(FusionEpoch(float(seg_end), k, ds.gnss_epochs[k], leo, tuple(segments),
                                          accel.copy(), gyro.copy(), len(boot)))
                segments = []
            seg_start = seg_end
            k += 1
    if k != len(t_fus):
        raise RuntimeError('not all requested fusion epochs were scheduled')
    if len(epochs) < 4:
        raise ValueError('preprocessing produced fewer than four usable fusion records; check GNSS products, '
                         'TLE coverage, masks, and time synchronization')
    return tuple(epochs)


def training_nmax(epochs) -> int:
    """Yan N_max: largest per-epoch measurement count in training (Eq. (16) padding)."""
    counts = [int(e.n_boot) for e in epochs]
    if not counts or max(counts) <= 0:
        raise ValueError('training epochs contain no measurements')
    return max(counts)


def measurement_sigma(meas) -> np.ndarray:
    """Eq. (3) standard deviation [m] of each row of measurement_model(nav, meas) (same order)."""
    return np.sqrt(np.asarray([float(m.variance_m2) for m in clock_observable(meas)], dtype=float))


STATE_BLOCKS = (('position', slice(0, 3)), ('velocity', slice(3, 6)), ('attitude', slice(6, 9)))


def classical_pass(ds: Dataset, epochs, *, nmax) -> dict:
    """One classical EKF pass over `epochs`; per-update records for the fixed scales.

    Returns per update k: time, gain K_k (9 x n_k), measurement sigma_k (n_k, same
    column order as K_k), prior std sqrt(diag P_k^-) (9,), and the 24 fixed network
    features built exactly as network_input builds them (from the second update on).
    """
    nav, P = ds.nav0.copy(), np.asarray(ds.P0, dtype=float).copy()
    rec = {'time': [], 'K': [], 'sigma': [], 'prior_std': [], 'feat': [], 'feat_time': []}
    prev = None                                        # (posterior nav, accel, gyro)
    for ep in epochs:
        nav, P = propagate(nav, P, ep.segments, ds.gyro, ds.accel, ds.Qc)
        meas = prepare_measurements(ds.gnss, ep.gnss_epoch, ep.leo_meas, nav, ds.lever_b, capacity=int(nmax))
        if not meas:
            continue
        m = measurement_model(nav, meas, ds.lever_b)
        if m.innovation.size == 0:
            continue
        prior, prior_std = nav.copy(), np.sqrt(np.clip(np.diag(P), 0.0, None))
        dx, P, K = kf_update(P, m.innovation, m.H, m.R)
        if not (np.all(np.isfinite(K)) and np.all(np.isfinite(dx))):
            raise FloatingPointError(f'non-finite classical EKF gain at {ep.time}')
        rec['time'].append(float(ep.time))
        rec['K'].append(K)
        rec['sigma'].append(measurement_sigma(meas))   # rows of measurement_model(nav, meas)
        rec['prior_std'].append(prior_std)
        nav = inject_error(nav, dx)
        accel = np.asarray(ep.accel, dtype=float).reshape(3)
        gyro = np.asarray(ep.gyro, dtype=float).reshape(3)
        if prev is not None:
            rec['feat'].append(np.concatenate((accel - prev[1], gyro - prev[2],
                                               state_diff(nav, prev[0]),      # dx_resid, Yan Eq. (13)
                                               state_diff(nav, prior))))      # dx_innov, Yan Eq. (12)
            rec['feat_time'].append(float(ep.time))
        prev = (nav.copy(), accel.copy(), gyro.copy())
    if not rec['K'] or not rec['feat']:
        raise ValueError('no classical EKF update was available for the fixed scales')
    return rec


def steady_state_mask(times, block_std, *, settle_factor, min_burn_in_s, min_epochs, name='') -> np.ndarray:
    """Epochs after the start-up transient of one state block.

    block_std: prior std of the block per update.  Steady level = median over the
    second half; the transient ends at the first update whose std is within
    settle_factor x that level, and never before min_burn_in_s.  Later spikes
    (geometry changes) are not transient and are left to the median.
    """
    times, block_std = np.asarray(times, dtype=float), np.asarray(block_std, dtype=float)
    ref = float(np.median(block_std[block_std.size // 2:]))
    settled = np.flatnonzero(block_std <= settle_factor * ref)
    t_settle = float(times[settled[0]]) if settled.size else float(times[-1])
    keep = times >= max(t_settle, float(times[0]) + float(min_burn_in_s))
    if int(keep.sum()) < int(min_epochs):
        raise ValueError(f'{name}: only {int(keep.sum())} steady-state epochs after burn-in (need {min_epochs}); '
                         'use a longer training split or relax SCALE_SETTLE_FACTOR / SCALE_MIN_BURN_IN_S / '
                         'SCALE_MIN_EPOCHS')
    return keep


def empirical_row_scale(rec) -> np.ndarray:
    """'empirical' g: RMS of each classical gain row over all columns and all updates."""
    sq_sum = sum(np.sum(K * K, axis=1) for K in rec['K'])
    n_cols = sum(int(K.shape[1]) for K in rec['K'])
    return np.sqrt(sq_sum / n_cols)


def analytic_row_scale(rec) -> tuple[np.ndarray, dict]:
    """'analytic' g = d (optionally x row balance r), robust to the P0 start-up transient.

    d_i  : median prior std of state i over the steady-state updates of its block
           (per-block adaptive burn-in), rounded to a power of 10 if SCALE_ROUND_POW10.
    K~   : classical gain in the scaled frame, K~ = diag(1/d) K diag(sigma)
           (State scaling: D'Souza, Zanetti, Woffinden 2019; measurement whitening).
    r_i  : median over steady updates of the per-update RMS of K~ row i (diagnostic;
           applied only if SCALE_ROW_BALANCE).
    """
    times = np.asarray(rec['time'], dtype=float)
    std = np.asarray(rec['prior_std'], dtype=float)                  # (updates, 9)
    d_raw, keep_rows, info = np.empty(N_STATE), np.zeros((times.size, N_STATE), dtype=bool), {'blocks': {}}
    for name, sl in STATE_BLOCKS:
        block = np.sqrt(np.mean(std[:, sl] ** 2, axis=1))
        keep = steady_state_mask(times, block, settle_factor=cfg.SCALE_SETTLE_FACTOR,
                                 min_burn_in_s=cfg.SCALE_MIN_BURN_IN_S, min_epochs=cfg.SCALE_MIN_EPOCHS, name=name)
        d_raw[sl] = np.median(std[keep][:, sl], axis=0)
        keep_rows[:, sl] = keep[:, None]
        info['blocks'][name] = {'burn_in_s': float(times[keep][0] - times[0]), 'steady_updates': int(keep.sum()),
                                'initial_std': float(block[0]), 'steady_median_std': float(np.median(block[keep]))}
    if not np.all(np.isfinite(d_raw)) or np.any(d_raw <= 0.0):
        raise FloatingPointError('non-finite or non-positive steady-state prior std')
    d = 10.0 ** np.round(np.log10(d_raw)) if cfg.SCALE_ROUND_POW10 else d_raw.copy()

    # Classical gain in the scaled frame; per-update RMS of each row.
    kt_rms = np.stack([np.sqrt(np.mean((K / d[:, None] * sig[None, :]) ** 2, axis=1))
                       for K, sig in zip(rec['K'], rec['sigma'])])        # (updates, 9)
    r = np.asarray([np.median(kt_rms[keep_rows[:, i], i]) for i in range(N_STATE)])
    r = np.maximum(r, GAIN_SCALE_FLOOR)
    g = d * r if cfg.SCALE_ROW_BALANCE else d
    info.update({'d_raw': d_raw, 'd': d, 'k_tilde_row_rms': r, 'k_tilde_spread': float(r.max() / r.min()),
                 'row_balance': bool(cfg.SCALE_ROW_BALANCE)})
    return g, info


def classical_scales(ds: Dataset, epochs, *, nmax, mode='empirical') -> tuple[np.ndarray, np.ndarray, dict]:
    """Fixed scales from one classical EKF pass over the training split.

    g           : output row scale of the gain for `mode` ('empirical' | 'none' | 'analytic').
    fixed_scale : RMS of each 3-component group of the 24 fixed network features
                  [d_accel, d_gyro, dx_resid(p, v, theta), dx_innov(p, v, theta)],
                  repeated to 24 entries (input scale, fixed / fixed_scale).  RMS about
                  zero, all updates; identical in every mode so the modes differ only in g.
    info        : diagnostics (saved with the training results).
    Both are set once before training (reproduction choice) and stored in the checkpoint.
    """
    rec = classical_pass(ds, epochs, nmax=nmax)
    F = np.asarray(rec['feat'], dtype=float).reshape(len(rec['feat']), len(FIXED_GROUPS), 3)
    group_rms = np.sqrt(np.mean(F ** 2, axis=(0, 2)))
    g_emp = empirical_row_scale(rec)
    info = {'mode': mode, 'updates': len(rec['K']), 'empirical_g': g_emp}
    if mode == 'empirical':
        g = g_emp
    elif mode in ('none', 'covariance'):
        g = np.ones(N_STATE)
    elif mode == 'analytic':
        g, extra = analytic_row_scale(rec)
        info.update(extra)
    else:
        raise ValueError(f"mode must be 'empirical', 'none', 'analytic' or 'covariance'; received {mode!r}")
    if not (np.all(np.isfinite(g)) and np.all(np.isfinite(group_rms))):
        raise FloatingPointError('non-finite classical scales')
    g = g if mode in ('none', 'covariance') else np.maximum(g, GAIN_SCALE_FLOOR)
    info['g'] = g
    return g, np.repeat(np.maximum(group_rms, FIXED_SCALE_FLOOR), 3), info


def classical_gain_row_scale(ds: Dataset, epochs, *, nmax, mode='empirical') -> np.ndarray:
    """Output scale g only (kept for existing callers)."""
    return classical_scales(ds, epochs, nmax=nmax, mode=mode)[0]


def padded_sigma(sigma, nmax, like: torch.Tensor) -> torch.Tensor:
    """Eq. (3) std of the first n rows, padded with 1 to nmax (padding is masked anyway)."""
    sigma = torch.as_tensor(sigma, dtype=like.dtype, device=like.device).reshape(-1)
    out = like.new_ones(int(nmax))
    out[:sigma.numel()] = sigma
    return out


def make_context(sat_ids, residual, sigma, dx_innov, dx_resid, nav, accel, gyro) -> dict:
    """Lagged quantities of epoch k-1 needed by the Eq. (15) input of epoch k.

    sigma: Eq. (3) standard deviation of each residual row (input normalization).
    """
    return {'sat_ids': tuple(sat_ids), 'residual': residual, 'sigma': sigma, 'dx_innov': dx_innov,
            'dx_resid': dx_resid, 'pos': nav.pos, 'vel': nav.vel, 'dcm': nav.dcm, 'accel': accel, 'gyro': gyro}


def classical_start(nav_prior: NavState, P, meas, lever_b, accel, gyro):
    """One classical EKF update -> (posterior nav, P, context).  Starts every recursion."""
    m = measurement_model(nav_prior, meas, lever_b)
    dx, P, _ = kf_update(P, m.innovation, m.H, m.R)
    nav = inject_error(nav_prior, dx)
    ctx = make_context(m.sat_ids, measurement_model(nav, meas, lever_b).innovation.copy(), measurement_sigma(meas),
                       state_diff(nav, nav_prior), np.zeros(N_STATE), nav.copy(), accel.copy(), gyro.copy())
    return nav, P, ctx


def _check_finite(where, stage, *values) -> None:
    for v in values:
        a = v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else np.asarray(v)
        if not np.all(np.isfinite(a)):
            raise FloatingPointError(f'{where}: non-finite {stage}')


# --- Recursive evaluation (test) ---------------------------------------------
def run_recursive(model, epochs, ds: Dataset, warm_start=True) -> dict:
    """Causal 9-state recursion with FDE before the network gain (Yan Sec. II-D).

    Per epoch: propagate -> FDE on all measurements -> exclude a uniquely
    identified fault -> network gain on the retained set -> Joseph covariance.
    No update ('hold_*'): prior kept, LSTM state carried unchanged.
    The first epoch with measurements uses one classical update (warm start).
    """
    epochs = tuple(epochs)
    if not epochs:
        raise ValueError('run_recursive requires at least one FusionEpoch')
    nav, P = ds.nav0.copy(), np.asarray(ds.P0, dtype=float).copy()
    ctx, h = None, None
    nmax = int(model.nmax)
    try:
        dev = next(model.parameters()).device
    except (AttributeError, StopIteration):
        dev = torch.device(DEVICE)
    rows, hold_counts, excluded_log = [], {}, []

    with torch.inference_mode():
        for ep in epochs:
            where = f'{ds.name} {ep.time}'
            nav, P = propagate(nav, P, ep.segments, ds.gyro, ds.accel, ds.Qc)
            accel, gyro = ep.accel, ep.gyro              # = last propagated IMU sample
            _check_finite(where, 'propagated state', nav.pos, nav.vel, nav.dcm, P)
            meas = prepare_measurements(ds.gnss, ep.gnss_epoch, ep.leo_meas, nav, ds.lever_b, capacity=None)
            prior, ctx_before = nav.copy(), ctx
            excluded, retained, hold = (), (), None

            # FDE on the full, INS-predicted innovation.
            if not meas:
                hold = 'hold_no_measurements'
            else:
                full = measurement_model(nav, meas, ds.lever_b)
                if full.sat_ids != tuple(m.sat_id for m in meas):
                    raise ValueError(f'{where}: measurement order mismatch')
                _check_finite(where, 'innovation', full.innovation)
                _check_finite(where, 'measurement Jacobian', full.H)
                det = detect_fault(P, full, cfg.FDE_ALPHA)
                if not det.detected:
                    retained = meas
                else:
                    ident = identify_fault(full, det.Q_nu_nu)
                    if ident.identified:
                        # Hard exclusion == Yan Eq. (34) for one fault; the network
                        # gain on the retained set is its learned analogue.
                        exc = exclude_fault(meas, ident)
                        retained = exc.retained
                        excluded = (exc.excluded_sat_id,) + exc.dropped_sat_ids
                    else:
                        hold = 'hold_unresolved_fault'           # not defined by the paper
                if hold is None and not retained:
                    hold = 'hold_no_retained_measurements'
            selected = ()
            if hold is None and not warm_start:
                selected = select_measurements(retained, nmax)
                if not selected:
                    hold = 'hold_no_retained_measurements'

            if hold is not None:
                dx, active, mode = np.zeros(N_STATE), (), hold
                hold_counts[hold] = hold_counts.get(hold, 0) + 1
            elif warm_start:
                m = measurement_model(nav, retained, ds.lever_b)
                dx, P, _ = kf_update(P, m.innovation, m.H, m.R)
                active, warm_start = retained, False
                mode = 'warm_classical_after_exclusion' if excluded else 'warm_classical_no_fault'
            else:
                if ctx is None:
                    ctx = make_context((), np.zeros(0), np.zeros(0), np.zeros(N_STATE), np.zeros(N_STATE),
                                       nav.copy(), accel.copy(), gyro.copy())
                m = measurement_model(nav, selected, ds.lever_b)
                if m.sat_ids != tuple(s.sat_id for s in selected):
                    raise ValueError(f'{where}: measurement order mismatch')
                t = lambda x: torch.as_tensor(x, dtype=torch.float64, device=dev)
                prev_t = {'sat_ids': ctx['sat_ids'], **{k: t(ctx[k]) for k in
                                                        ('residual', 'sigma', 'dx_innov', 'dx_resid', 'accel',
                                                         'gyro')}}
                fixed, obs, mask, channel, innov = network_input(prev_t, m.sat_ids, t(m.innovation), t(accel),
                                                                 t(gyro), nmax=nmax,
                                                                 sigma=t(measurement_sigma(selected)))
                col_sigma = padded_sigma(measurement_sigma(selected), nmax, innov)
                Hp, Rp = pad_measurement_model(t(m.H), t(m.R), nmax)
                out = model(fixed.to(DTYPE).unsqueeze(0), obs.to(DTYPE).unsqueeze(0), mask.unsqueeze(0),
                            channel.unsqueeze(0), recurrent_state=h, column_sigma=col_sigma.unsqueeze(0),
                            prior_cov=t(P).unsqueeze(0), H=Hp.unsqueeze(0), R=Rp.unsqueeze(0))
                h = out.recurrent_state
                _check_finite(where, 'gain', out.kalman_gain)
                dx_t = torch.bmm(out.kalman_gain.to(dtype=innov.dtype), innov.unsqueeze(0).unsqueeze(-1)).squeeze(-1)[0]
                _check_finite(where, 'correction', dx_t)
                dx = dx_t.detach().cpu().numpy().astype(float)
                K = out.kalman_gain[0, :, :len(m.sat_ids)].detach().cpu().numpy().astype(float)
                P = joseph_reset(P, K, m.H, m.R, dx)             # P_k for the next FDE test
                active = selected
                mode = 'neural_after_exclusion' if excluded else 'neural_no_fault'

            excluded_log.append(excluded)
            _check_finite(where, 'correction', dx)
            if hold is None:
                nav = inject_error(nav, dx)
            _check_finite(where, 'injected state', nav.pos, nav.vel, nav.dcm)

            # Yan Eqs. (10)-(14): lagged residual, state innovation and state residual.
            dx_resid = np.zeros(N_STATE) if ctx_before is None else \
                state_diff(nav, (ctx_before['pos'], ctx_before['vel'], ctx_before['dcm']))
            ctx = make_context((s.sat_id for s in clock_observable(active)),
                               np.asarray(measurement_model(nav, active, ds.lever_b).innovation, dtype=float).copy(),
                               measurement_sigma(active), state_diff(nav, prior), dx_resid, nav.copy(),
                               np.asarray(accel, dtype=float).copy(), np.asarray(gyro, dtype=float).copy())

            i = ep.index
            err = np.concatenate((ds.truth_pos[i] - nav.pos, ds.truth_vel[i] - nav.vel,
                                  so3_log(ds.truth_dcm[i] @ nav.dcm.T) / ATT_SIGN))
            rows.append((ep.time, nav.antenna(ds.lever_b), np.asarray(ds.truth_ant_pos[i], dtype=float).copy(),
                         float(err @ err), mode))

    return {
        'time': np.asarray([r[0] for r in rows], dtype=float),
        'estimate_ecef_m': np.asarray([r[1] for r in rows], dtype=float).reshape(-1, 3),
        'truth_ecef_m': np.asarray([r[2] for r in rows], dtype=float).reshape(-1, 3),
        'state_loss': np.asarray([r[3] for r in rows], dtype=float),
        'update_mode': np.asarray([r[4] for r in rows], dtype=str),
        'diagnostics': {'hold_counts': hold_counts, 'fde_excluded_sat_ids': tuple(excluded_log)},
    }


# --- Training ----------------------------------------------------------------
def training_plan(n_train: int) -> tuple:
    """Overlapping subsequences (start, stop, windows) over training samples [0, n_train).

    Starts every SEQUENCE_STRIDE samples; a final start is added so the tail
    of the training split is always covered.
    """
    L, stride, W = int(cfg.SEQUENCE_LENGTH), int(cfg.SEQUENCE_STRIDE), int(cfg.WINDOW_SIZE)
    if n_train <= 0:
        return ()
    starts = list(range(0, max(n_train - L, 0) + 1, stride))
    if starts[-1] + L < n_train:
        starts.append(n_train - L)
    return tuple((s, min(s + L, n_train),
                  tuple((w, min(w + W, s + L, n_train)) for w in range(s, min(s + L, n_train), W)))
                 for s in starts)


def train_model(model: MaskedCLA, epochs, ds: Dataset) -> dict:
    """Minimize Yan Eq. (32) = mean Eq. (30) state loss + L2, with TBPTT.

    Split: first samples -> independent subsequences of SEQUENCE_LENGTH
    (mini-batched, one optimizer step per WINDOW_SIZE steps), last
    VALIDATION_FRACTION -> one uninterrupted validation rollout.
    Sample i updates at epochs[i + 1]; epochs[0] is the causal warm start.
    """
    nmax = training_nmax(epochs)
    lever = ds.lever_b

    # Causal warm start on the first epoch (classical update from the INS prior).
    ep0 = epochs[0]
    nav, P = propagate(ds.nav0.copy(), ds.P0.copy(), ep0.segments, ds.gyro, ds.accel, ds.Qc)
    meas = prepare_measurements(ds.gnss, ep0.gnss_epoch, ep0.leo_meas, nav, lever, capacity=None)
    if not meas:
        raise RuntimeError(f'{ep0.time}: first FusionEpoch lost all warm-start measurements')
    first_start = classical_start(nav, P, meas, lever, ep0.accel, ep0.gyro)

    np.random.seed(cfg.SEED)
    torch.manual_seed(cfg.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.SEED)

    n_total = len(epochs) - 1
    n_val = max(1, int(round(n_total * cfg.VALIDATION_FRACTION)))
    n_train = n_total - n_val
    plan = training_plan(n_train)
    if not plan:
        raise ValueError('at least one independent training sequence is required')
    print(f'Sequence mini-batching | sequences={len(plan)} | requested batch={cfg.BATCH_SIZE} | '
          f'effective batch={min(cfg.BATCH_SIZE, len(plan))} | lr={cfg.LEARNING_RATE:g}')

    def truth_start(sample, split):
        """Subsequence start (KalmanNet4SensorFusion style): truth state + one classical update."""
        rec = epochs[int(sample)]
        seed_nav = NavState(ds.truth_pos[rec.index].copy(), ds.truth_vel[rec.index].copy(),
                            ds.truth_dcm[rec.index].copy(), np.zeros(3), np.zeros(3))
        meas = prepare_measurements(ds.gnss, rec.gnss_epoch, rec.leo_meas, seed_nav, lever, capacity=nmax)
        if not meas:
            raise ValueError(f'{split} boundary sample {int(sample)} has no usable measurements for cold start')
        return classical_start(seed_nav, ds.P0.copy(), meas, lever, rec.accel, rec.gyro)

    starts = {0: first_start, **{s: truth_start(s, 'training') for s, _, _ in plan if s != 0}}
    val_start = truth_start(n_train, 'validation')

    if int(model.nmax) != nmax:
        raise ValueError(f'model nmax={model.nmax} does not match training N_max={nmax}')
    mode = model.gain_scale_mode
    g, fixed_scale, scale_info = classical_scales(ds, epochs[:n_train + 1], nmax=nmax, mode=mode)
    model.set_gain_row_scale(g)
    model.set_fixed_scale(fixed_scale)
    print(f'Gain scale mode={mode} | g | ' + ' '.join(f'{v:.3e}' for v in g))
    if mode == 'analytic':
        for name, b in scale_info['blocks'].items():
            print(f"  {name}: burn-in={b['burn_in_s']:.1f} s | steady updates={b['steady_updates']} | "
                  f"prior std {b['initial_std']:.3g} -> {b['steady_median_std']:.3g}")
        print('  d (raw)             | ' + ' '.join(f'{v:.3e}' for v in scale_info['d_raw']))
        print('  RMS K~_EKF per row  | ' + ' '.join(f'{v:.3e}' for v in scale_info['k_tilde_row_rms']) +
              f" | spread={scale_info['k_tilde_spread']:.1f}x | row balance={scale_info['row_balance']}")
    print('  empirical g (ref.)  | ' + ' '.join(f'{v:.3e}' for v in scale_info['empirical_g']))
    print('Fixed input scale (group RMS, training split) | ' +
          ' '.join(f'{name}={v:.3e}' for name, v in zip(FIXED_GROUPS, fixed_scale[::3])))
    params = list(model.parameters())
    optimizer = torch.optim.Adam(params, lr=cfg.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=cfg.LR_PLATEAU_FACTOR, patience=cfg.LR_PLATEAU_PATIENCE,
        min_lr=cfg.LR_MIN)

    gyro_t, accel_t, lever_t = f64(ds.gyro), f64(ds.accel), f64(lever).reshape(3)
    truth_pos_t, truth_vel_t, truth_dcm_t = f64(ds.truth_pos), f64(ds.truth_vel), f64(ds.truth_dcm)

    use_cov = mode == 'covariance'

    def to_torch(start) -> dict:
        nav, P, ctx = start
        tctx = {k: (v if k == 'sat_ids' else f64(v)) for k, v in ctx.items()}
        return {'nav': TorchNav(*(f64(v).clone() for v in (nav.pos, nav.vel, nav.dcm, nav.ba, nav.bg))),
                'ctx': tctx, 'accel': tctx['accel'], 'gyro': tctx['gyro'], 'h': None,
                'P': np.asarray(P, dtype=float).copy()}

    def detach(state) -> dict:
        return {'nav': TorchNav(*(v.detach() for v in state['nav'])),
                'ctx': {k: v.detach() if isinstance(v, torch.Tensor) else v for k, v in state['ctx'].items()},
                'accel': state['accel'].detach(), 'gyro': state['gyro'].detach(),
                'h': None if state['h'] is None else tuple(v.detach() for v in state['h']),
                'P': state['P']}

    def innovation_t(nav: TorchNav, prep):
        sat_ids, sat_pos, pr, clk, iono, tropo, Pc = prep
        ant = nav.pos + nav.dcm @ lever_t
        predicted = torch.linalg.vector_norm(ant.unsqueeze(0) - sat_pos, dim=1) - C_LIGHT * clk + iono + tropo
        return Pc @ (pr - predicted)

    def propagate_t(nav: TorchNav, segments, training):
        segs = tuple((int(i), float(dt)) for i, dt in segments)

        def core(*state):
            s = TorchNav(*state)
            for i, dt in segs:
                s = mknet.mechanize(s, gyro_t[i].reshape(3), accel_t[i].reshape(3), dt)
            return tuple(s)
        if training and any(v.requires_grad for v in nav):
            out = checkpoint(core, *nav, use_reentrant=False)       # recompute INS in backward (memory)
        else:
            out = core(*nav)
        return TorchNav(*out), gyro_t[segs[-1][0]].reshape(3), accel_t[segs[-1][0]].reshape(3)

    def rollout(start, stop, state, training, weight=1.0):
        """Samples [start, stop); training: backward of weight * mean Eq. (30) loss."""
        if training and (not math.isfinite(float(weight)) or float(weight) <= 0.0):
            raise ValueError('training backward_weight must be positive and finite')
        nav, ctx, accel, gyro, h = state['nav'], state['ctx'], state['accel'], state['gyro'], state['h']
        P = state['P']                       # numpy covariance (no gradient); used in 'covariance' mode
        losses, loss_sum, pos_sum = [], 0.0, 0.0
        with (torch.enable_grad() if training else torch.inference_mode()):
            for i in range(start, stop):
                where = f'sample {i} | phase={"joint" if training else "validation"}'
                row = epochs[i + 1]
                if row.segments:
                    if use_cov:
                        nav_before = NavState(*(v.detach().cpu().numpy().astype(float).copy() for v in nav))
                        _, P = propagate(nav_before, P, row.segments, ds.gyro, ds.accel, ds.Qc)
                    nav, gyro, accel = propagate_t(nav, row.segments, training)
                _check_finite(where, 'predicted_state', *nav)

                nav_np = NavState(*(v.detach().cpu().numpy().astype(float).copy() for v in nav))
                meas = prepare_measurements(ds.gnss, row.gnss_epoch, row.leo_meas, nav_np, lever, capacity=nmax)
                if not meas:
                    raise FloatingPointError(f'no usable state-dependent measurements at Data01 sample {i}')
                sat_ids = tuple(m.sat_id for m in meas)
                sigma = f64(measurement_sigma(meas))
                prep = (sat_ids, f64(np.stack([m.sat_pos for m in meas])), f64([m.pseudorange_m for m in meas]),
                        f64([m.sat_clock_s for m in meas]), f64([m.iono_m for m in meas]),
                        f64([m.tropo_m for m in meas]), f64(clock_projector(meas)))
                innov = innovation_t(nav, prep)
                _check_finite(where, 'innovation', innov)

                fixed, obs, mask, channel, innov_pad = network_input(ctx, sat_ids, innov, accel, gyro, nmax=nmax,
                                                                     sigma=sigma)
                cov_kw = {}
                if use_cov:
                    m_np = measurement_model(nav_np, meas, lever)
                    if m_np.sat_ids != sat_ids:
                        raise ValueError(f'{where}: measurement order mismatch')
                    Hp, Rp = pad_measurement_model(f64(m_np.H), f64(m_np.R), nmax)
                    cov_kw = {'prior_cov': f64(P).unsqueeze(0), 'H': Hp.unsqueeze(0), 'R': Rp.unsqueeze(0)}
                out = model(fixed.to(dtype=DTYPE).unsqueeze(0), obs.to(dtype=DTYPE).unsqueeze(0),
                            mask.unsqueeze(0), channel.unsqueeze(0), recurrent_state=h,
                            column_sigma=padded_sigma(sigma, nmax, innov_pad).unsqueeze(0), **cov_kw)
                h = out.recurrent_state
                _check_finite(where, 'network_output', out.kalman_gain, *h)
                dx = torch.bmm(out.kalman_gain.to(dtype=innov_pad.dtype),
                               innov_pad.unsqueeze(0).unsqueeze(-1)).squeeze(-1)[0].reshape(N_STATE)
                _check_finite(where, 'correction', dx)

                # Yan Eq. (30): || x_true - (x_prior + K dy) ||^2.
                j = row.index
                err = torch.cat((truth_pos_t[j] - nav.pos, truth_vel_t[j] - nav.vel,
                                 mknet.so3_log(truth_dcm_t[j] @ nav.dcm.T) / ATT_SIGN)) - dx[:N_STATE]
                loss = torch.sum(err ** 2)
                if not bool(torch.isfinite(loss).detach().cpu()):
                    raise FloatingPointError(f'non-finite Yan Eq.(30) loss at sample {i}')
                losses.append(loss)
                loss_sum += float(loss.detach().cpu())
                pos_sum += float(torch.sum(err[:3] ** 2).detach().cpu())

                if use_cov:
                    K_np = out.kalman_gain[0, :, :len(sat_ids)].detach().cpu().numpy().astype(float)
                    P = joseph_reset(P, K_np, m_np.H, m_np.R, dx.detach().cpu().numpy().astype(float))
                prior = nav
                nav = TorchNav(nav.pos + dx[0:3], nav.vel + dx[3:6],
                               mknet.so3_exp(ATT_SIGN * dx[6:9]) @ nav.dcm, nav.ba, nav.bg)
                resid = innovation_t(nav, prep)
                _check_finite(where, 'posterior_state', *nav, resid)
                ctx = {'sat_ids': sat_ids, 'residual': resid, 'sigma': sigma,
                       'dx_innov': mknet.state_diff(nav, prior.pos, prior.vel, prior.dcm),
                       'dx_resid': mknet.state_diff(nav, ctx['pos'], ctx['vel'], ctx['dcm']),
                       'pos': nav.pos, 'vel': nav.vel, 'dcm': nav.dcm,
                       'accel': accel.reshape(3), 'gyro': gyro.reshape(3)}

                # TBPTT: keep the numerical state, cut the autograd history every k steps.
                if training and (i - start + 1) % cfg.TBPTT_DETACH_STEP == 0:
                    s = detach({'nav': nav, 'ctx': ctx, 'accel': accel, 'gyro': gyro, 'h': h, 'P': P})
                    nav, ctx, accel, gyro, h = s['nav'], s['ctx'], s['accel'], s['gyro'], s['h']

        carried = None
        if training:
            (torch.stack(losses).mean() * float(weight)).backward()
            carried = detach({'nav': nav, 'ctx': ctx, 'accel': accel, 'gyro': gyro, 'h': h, 'P': P})
        n = len(losses)
        return {'eq30': loss_sum / n, 'position_rmse_m': math.sqrt(pos_sum / n), 'state_sum': loss_sum,
                'position_sum': pos_sum, 'count': n, 'state': carried}

    history, since_best = [], 0
    best_state, best_loss, best_rmse, best_epoch = None, float('inf'), float('inf'), 0

    for epoch in range(1, cfg.TRAINING_EPOCHS + 1):
        try:
            rng = np.random.default_rng(np.random.SeedSequence((int(cfg.SEED), int(epoch))))
            shuffled = tuple(plan[k] for k in rng.permutation(len(plan)).tolist())
            batches = tuple(shuffled[k:k + cfg.BATCH_SIZE] for k in range(0, len(shuffled), cfg.BATCH_SIZE))

            for p in params:
                p.requires_grad_(True)
            model.train()
            loss_sum = pos_sum = 0.0
            count, reg_values, grad_norms, steps = 0, [], [], 0
            for batch in batches:
                carried = {s: to_torch(starts[s]) for s, _, _ in batch}
                for w in range(max(len(windows) for _, _, windows in batch)):
                    # One optimizer step per window index across the batch; each
                    # subsequence weighted by its share of valid steps (Eq. (32)).
                    members = tuple((seq[0], seq[2][w]) for seq in batch if w < len(seq[2]))
                    optimizer.zero_grad(set_to_none=True)
                    total = sum(b - a for _, (a, b) in members)
                    for s, (a, b) in members:
                        r = rollout(a, b, carried[s], training=True, weight=(b - a) / total)
                        carried[s] = r['state']
                        loss_sum += r['state_sum']
                        pos_sum += r['position_sum']
                        count += r['count']
                    reg = torch.zeros((), dtype=DTYPE, device=DEVICE)
                    for p in params:
                        reg = reg + torch.sum(p.to(dtype=DTYPE) ** 2)
                    reg = cfg.GAMMA_L2 * reg
                    reg.backward()
                    grad_norm = clip_grad_norm_float64_(params, max_norm=cfg.GRAD_CLIP_NORM)
                    optimizer.step()
                    steps += 1
                    reg_values.append(float(reg.detach().cpu()))
                    grad_norms.append(float(grad_norm.detach().cpu()))

            data_loss, reg_loss = loss_sum / count, float(np.mean(reg_values))
            joint = {'loss': data_loss + reg_loss, 'data_loss': data_loss, 'regularization_loss': reg_loss,
                     'learning_rate': float(optimizer.param_groups[0]['lr']),
                     'position_rmse_m': math.sqrt(pos_sum / count),
                     'gradient_l2_norm': float(np.mean(grad_norms)), 'optimizer_step_count': steps}

            for p in model.parameters():
                p.requires_grad_(False)
            model.eval()
            # Full-train rollout is report-only (no effect on training/selection): run every k epochs.
            train_eval = (rollout(0, n_train, to_torch(first_start), training=False)
                          if epoch == 1 or epoch % cfg.TRAIN_EVAL_EVERY == 0 else None)
            validation = rollout(n_train, n_total, to_torch(val_start), training=False)
        except FloatingPointError as error:
            print(f'Epoch {epoch:02d}: stopped - {error}')
            break
        except KeyboardInterrupt:
            # Ctrl+C: drop the unfinished epoch, keep the best checkpoint so far.
            print(f'Epoch {epoch:02d}: interrupted by user - keeping best epoch {best_epoch}.')
            break

        val_loss, val_rmse = float(validation['eq30']), float(validation['position_rmse_m'])
        if math.isfinite(val_loss) and val_loss < best_loss:
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_loss, best_rmse, best_epoch, since_best = val_loss, val_rmse, epoch, 0
        else:
            since_best += 1
        history.append({
            'epoch': epoch,
            'training_sequence_batches': [[int(seq[0]) for seq in b] for b in batches],
            'train_loss': float(joint['loss']), 'train_data_loss': float(joint['data_loss']),
            'train_regularization_loss': float(joint['regularization_loss']),
            'gradient_l2_norm': float(joint['gradient_l2_norm']), 'joint_training': joint,
            **({'overall_train_loss': float(train_eval['eq30'])} if train_eval is not None else {}),
            'validation_loss': val_loss, 'validation_position_rmse_m': val_rmse,
        })
        overall = f"{train_eval['eq30']:.6g}" if train_eval is not None else 'skipped'
        print(f"Epoch {epoch:02d}/{cfg.TRAINING_EPOCHS} | joint={joint['loss']:.6g} | "
              f"overall train loss={overall} | validation loss={val_loss:.6g} | "
              f'validation RMSE={val_rmse:.3f} m | lr={optimizer.param_groups[0]["lr"]:.2e}')
        if math.isfinite(val_loss):
            scheduler.step(val_loss)
        if since_best >= cfg.EARLY_STOP_PATIENCE:
            print(f'Early stopping after epoch {epoch}.')
            break

    if best_state is None:
        raise RuntimeError('training did not produce any finite validation checkpoint; no best_model.pt created')
    model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
    model.eval()
    return {'model': model, 'history': history, 'best_epoch': int(best_epoch), 'gain_scale': scale_info,
            'best_validation_state_loss': float(best_loss), 'best_validation_position_rmse_m': float(best_rmse)}


# --- Evaluation and output ---------------------------------------------------
def evaluate_run(run) -> dict:
    est = np.asarray(run['estimate_ecef_m'], dtype=float).reshape(-1, 3)
    truth = np.asarray(run['truth_ecef_m'], dtype=float).reshape(-1, 3)
    if est.shape != truth.shape or est.shape[0] == 0:
        raise ValueError('evaluation requires matching, non-empty (epoch, 3) estimate and truth')
    ned = np.stack([ecef_to_ned(*ecef_llh(t)[:2]) @ (e - t) for e, t in zip(est, truth)])
    modes, counts = np.unique(np.asarray(run['update_mode'], dtype=str), return_counts=True)
    return {
        'epoch_count': int(est.shape[0]),
        'state_loss_mean': float(np.mean(np.asarray(run['state_loss'], dtype=float))),
        'rmse_ned3d_m': [float(v) for v in np.append(np.sqrt(np.mean(ned ** 2, axis=0)),
                                                       math.sqrt(float(np.mean(np.sum(ned ** 2, axis=1)))))],
        'update_mode_counts': {str(m): int(c) for m, c in zip(modes, counts)},
    }


def _json_ready(v):
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, Path):
        return str(v)
    if isinstance(v, dict):
        return {str(k): _json_ready(x) for k, x in v.items()}
    if isinstance(v, (tuple, list)):
        return [_json_ready(x) for x in v]
    return v


def _new_output_dir(path) -> Path:
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f'refusing to overwrite non-empty output directory: {path}')
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_training_results(out_dir, history, validation, gain_scale=None) -> None:
    import matplotlib.pyplot as plt
    out = _new_output_dir(out_dir)
    if gain_scale is not None:
        (out / 'gain_scale.json').write_text(json.dumps(_json_ready(gain_scale), indent=2), encoding='utf-8')
    (out / 'history.json').write_text(json.dumps(_json_ready(history), indent=2), encoding='utf-8')
    (out / 'validation.json').write_text(json.dumps(_json_ready(validation), indent=2), encoding='utf-8')
    fig, ax = plt.subplots(figsize=(7, 4))
    if history:
        epochs = [r.get('epoch', i + 1) for i, r in enumerate(history)]
        tr = [(e, r['overall_train_loss']) for e, r in zip(epochs, history) if 'overall_train_loss' in r]
        if tr:
            ax.plot(*zip(*tr), marker='o', label='train')
        ax.plot(epochs, [r.get('validation_loss', np.nan) for r in history], label='validation')
        ax.legend()
    ax.set(xlabel='Epoch', ylabel='State loss', title='Training and validation loss')
    fig.tight_layout()
    fig.savefig(out / 'loss.png', dpi=150)
    plt.close(fig)


def save_test_results(out_dir, validation, test) -> None:
    import matplotlib.pyplot as plt
    out = _new_output_dir(out_dir)
    evaluation = evaluate_run(test)
    (out / 'evaluation.json').write_text(json.dumps(evaluation, indent=2), encoding='utf-8')
    summary = {'state_model': STATE_MODEL_NAME, 'state_order': STATE_ORDER, 'gain_scale_mode': cfg.GAIN_SCALE_MODE,
               'feature_semantics_version': FEATURE_VERSION, 'validation': _json_ready(validation),
               'test': evaluation}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')

    est = np.asarray(test['estimate_ecef_m'], dtype=float).reshape(-1, 3)
    truth = np.asarray(test['truth_ecef_m'], dtype=float).reshape(-1, 3)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(truth[:, 0], truth[:, 1], label='truth')
    ax.plot(est[:, 0], est[:, 1], label='estimate')
    ax.set(xlabel='ECEF x [m]', ylabel='ECEF y [m]', title='Horizontal path')
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / 'path.png', dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.asarray(test['time'], dtype=float), np.linalg.norm(est - truth, axis=1))
    ax.set(xlabel='GPST [s]', ylabel='3-D error [m]', title='Position error')
    fig.tight_layout()
    fig.savefig(out / 'position_error.png', dpi=150)
    plt.close(fig)


def main() -> None:
    if cfg.OUTPUT_DIR.exists() and any(cfg.OUTPUT_DIR.iterdir()):
        raise FileExistsError(f'refusing to overwrite non-empty output directory: {cfg.OUTPUT_DIR}')
    np.random.seed(cfg.SEED)
    torch.manual_seed(cfg.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.SEED)
    ds = load_dataset('train')
    epochs = prepare_fusion_epochs(ds, load_cache(cfg.TRAIN_LEO_CACHE, ds).meas)
    model = MaskedCLA(nmax=training_nmax(epochs), dropout=cfg.RECURRENT_DROPOUT,
                      gain_scale_mode=cfg.GAIN_SCALE_MODE).to(device=DEVICE, dtype=DTYPE)
    result = train_model(model, epochs, ds)
    validation = {'best_epoch': result['best_epoch'], 'state_loss_mean': result['best_validation_state_loss'],
                  'position_rmse_m': result['best_validation_position_rmse_m']}
    save_training_results(cfg.OUTPUT_DIR, result['history'], validation, result['gain_scale'])
    save_checkpoint(cfg.CHECKPOINT_PATH, result['model'], {
        'epoch': result['best_epoch'], 'validation_state_loss': result['best_validation_state_loss'],
        'validation_position_rmse_m': result['best_validation_position_rmse_m']})


if __name__ == '__main__':
    main()