"""GNSS/LEO/INS filter of paper Fig. 2 with a traditional EKF gain or the Masked-CLA gain.

One function runs the filter for training, validation and test, so the network
always sees the same filter. One fusion epoch k: INS propagation ->
measurements -> Kalman gain -> fault detection on the INS-predicted innovation
(Eq. (33)) -> state update, with Eq. (34) adaptation when faults are identified
(the test is repeated after each identified fault, A27).

Fault detection, identification and adaptation (paper Sec. II-D, Ref. [33]) and
integrity: the test runs on the innovation predicted by the INS, before the
network gain is applied (paper Sec. II-D). All vectors are receiver-clock free,
so a pseudorange fault e_i appears in the innovation as c_i = projector @ e_i.
"""
import numpy as np
import torch
from scipy.stats import chi2

import settings as cfg
from measurements.earth_models import ecef_to_llh, ecef_to_ned_matrix, skew
from navigation.ins_filter import (STATE_SIZE, apply_correction, classical_gain, error_matrix, initial_covariance,
                                   joseph_covariance, measurement_model, process_noise, propagate_ins,
                                   state_difference, transition_matrix, truth_state)
from navigation.masked_cla_network import build_network_input, network_input_values

PARALLEL_TOLERANCE = 1e-9     # relative: c_i parallel to an identified fault, or already explained by it


# --- Fault detection and integrity (paper Sec. II-D, Ref. [33]) --------------
def find_faults(innovation, H, R, P_prior, projector):
    """Fault detection and identification, repeated after each identified fault (A27).

    Paper Eq. (33) / Ref. [33] Eq. (6): T = nu^T M nu > chi2(1 - alpha, rank Q - q), Q = H P H^T + R,
    with M = Q^+ (I - C C^+) the metric after the q faults already found (columns of C, C^+ of
    Ref. [33] Eq. (39)); M = Q^+ when q = 0. Ref. [33] Eq. (7): the next fault is the row with the
    largest T_i = (c_i^T M nu)^2 / (c_i^T M c_i), c_i = projector column i (pseudorange fault seen in
    the clock-free innovation). Rows with c_i parallel to that of the identified row are the same
    hypothesis of Ref. [33] (two rows of one GNSS system) and are named with it.
    Returns (identified rows, [rows of each identified hypothesis], Q^+).
    """
    Q = H @ P_prior @ H.T + R
    Q_inverse = np.linalg.pinv(Q, hermitian=True)
    redundancy = np.linalg.matrix_rank(Q, hermitian=True)
    c = projector
    norm_without_faults = np.einsum('ji,jk,ki->i', c, Q_inverse, c)
    faulty, hypotheses = [], []
    while len(faulty) < redundancy:
        C = c[:, faulty]
        M = Q_inverse - Q_inverse @ C @ np.linalg.solve(C.T @ Q_inverse @ C, C.T @ Q_inverse)
        threshold = chi2.ppf(1.0 - cfg.FALSE_ALARM_PROBABILITY, redundancy - len(faulty))
        if innovation @ M @ innovation <= threshold:
            break
        norm = np.einsum('ji,jk,ki->i', c, M, c)
        testable = norm > PARALLEL_TOLERANCE * norm_without_faults        # not yet explained by the faults found
        T = np.where(testable, (c.T @ M @ innovation) ** 2 / np.where(testable, norm, 1.0), -np.inf)
        index = int(np.argmax(T))
        parallel = (c.T @ M @ c[:, index]) ** 2 >= (1.0 - PARALLEL_TOLERANCE) * norm * norm[index]
        faulty.append(index)
        hypotheses.append(np.flatnonzero(testable & parallel))
    return faulty, hypotheses, Q_inverse


def adapt_to_faults(dx, P_prior, K, H, R, innovation, projector, faulty, Q_inverse):
    """Paper Eq. (34): x_i = x_0 - L_i nu, L_i = K C_i C_i^+ (Ref. [33] Eq. (39)).

    C_i: projector columns of the faulty rows (one column per fault, Ref. [33] allows q_i >= 1).
    x_i is the update with the gain K - L_i, so P_i is the Joseph covariance of that gain. For the
    EKF gain it equals P_0 + L_i Q L_i^T of Eq. (34); that form assumes the optimal gain, so it
    does not hold for the network gain.
    """
    C = projector[:, faulty]
    L = K @ C @ np.linalg.solve(C.T @ Q_inverse @ C, C.T @ Q_inverse)
    return dx - L @ innovation, joseph_covariance(P_prior, K - L, H, R)


def protection_levels(P, state, lever_arm):
    """Horizontal and vertical protection levels K * sigma of the antenna position (A17).

    The antenna error is dp - (C l) x dtheta (ins_filter.measurement_model), so its covariance is J P J^T.
    """
    J = np.zeros((3, STATE_SIZE))
    J[:, 0:3], J[:, 6:9] = np.eye(3), -skew(state.attitude @ lever_arm)
    latitude, longitude, _ = ecef_to_llh(state.position)
    C = ecef_to_ned_matrix(latitude, longitude)
    P_ned = C @ J @ P @ J.T @ C.T
    half_sum, half_difference = 0.5 * (P_ned[0, 0] + P_ned[1, 1]), 0.5 * (P_ned[0, 0] - P_ned[1, 1])
    horizontal_major = half_sum + np.hypot(half_difference, P_ned[0, 1])
    return cfg.HORIZONTAL_PL_FACTOR * np.sqrt(horizontal_major), cfg.VERTICAL_PL_FACTOR * np.sqrt(P_ned[2, 2])


def stanford_percentages(error, protection_level):
    """Share of epochs in each region of the Stanford diagram (paper Fig. 20)."""
    al = cfg.ALERT_LIMIT
    regions = {
        'NO': (error <= protection_level) & (protection_level <= al),      # nominal operation
        'MI': (protection_level < error) & (error <= al),                  # misleading information
        'HO': (protection_level <= al) & (al < error),                     # hazardous operation
        'SU': (al < protection_level) & (error <= protection_level),       # system unavailable
        'SU&MI': (al < protection_level) & (protection_level < error),
    }
    return {name: 100.0 * float(np.mean(region)) for name, region in regions.items()}


# --- Filter (paper Fig. 2) ---------------------------------------------------
def run_filter(data, measurements, network=None, first=0, last=None, fault_detection=True, training=False,
               max_measurements=None, input_samples=None):
    """Filter over fusion epochs first..last, starting from the truth at 'first' (A7).

    network=None: traditional EKF gain, with at most max_measurements rows (A16). With a list input_samples, it
    appends the network input X_k of every epoch with a measurement as (X_k, present, sat_ids)
    (masked_cla_network.network_input_values), for the input normalization statistics (exp/input-norm).
    training=True: accumulates the gradient of the Eq. (32) loss (A15). The filter runs in NumPy; the gradient
    follows its linearization: 'link' is the shift of the prior state caused by the earlier corrections
    (a correction at k shifts the prior at k+1 by Phi dx_k), the innovation shifts by -H link, so
    dx_k = K_k (nu_k - H_k link_k) gives the sensitivity (I - K H) Phi S + (dK/dtheta) nu, and the network
    inputs of the next epoch (innovation, residual after the update, state residual and state innovation)
    carry the same shift. The gradient is truncated every BACKPROP_WINDOW epochs.
    The loss counts every epoch, also those without a usable measurement (no correction).
    The covariance P is used only by the EKF gain, the fault detection and the protection
    levels; a network run without fault detection (training, validation) leaves it out and
    returns NaN protection levels.
    """
    last = len(data.fusion_times) - 1 if last is None else last
    times = data.fusion_times
    state, P = truth_state(data, first), initial_covariance(data)
    i = np.searchsorted(data.imu_times, times[first], side='right') - 1            # last IMU sample <= start
    no_shift = torch.zeros(STATE_SIZE, dtype=torch.float64)
    previous = {'accel': data.accel[i], 'gyro': data.gyro[i], 'state': state, 'residuals': {}, 'shift': no_shift,
                'state_residual': no_shift, 'state_innovation': no_shift}
    hidden, link, window_loss, loss_sum = None, no_shift, 0.0, 0.0
    if network is not None:
        network.train(training)
    rows, gain_square_sum, gain_columns = [], np.zeros(STATE_SIZE), 0
    covariance = network is None or fault_detection

    for k in range(first + 1, last + 1):
        state, mean_force, accel, gyro = propagate_ins(state, data, times[k - 1], times[k])
        Phi = transition_matrix(error_matrix(state, mean_force), times[k] - times[k - 1])
        if covariance:
            P = Phi @ P @ Phi.T + process_noise(Phi, times[k] - times[k - 1])
        link = torch.from_numpy(Phi) @ link
        shift = link - link.detach()                   # value 0, gradient d(prior)/d(theta)
        model = measurement_model(state, measurements[k], data.lever_arm, times[k], data.klobuchar_alpha,
                                  data.klobuchar_beta, max_measurements if network is None else network.max_measurements)
        dx, dx_tensor, faulty_satellite, count = np.zeros(STATE_SIZE), no_shift, '', 0
        if model is not None:
            nu, H, R = model.innovation, model.H, model.R
            count = len(nu)
            if network is None:
                if input_samples is not None:
                    input_samples.append((*network_input_values(previous, model.measurements.sat_ids,
                                                                torch.from_numpy(nu), accel, gyro),
                                          model.measurements.sat_ids))
                K = classical_gain(P, H, R)
                dx = K @ nu
            else:
                with torch.set_grad_enabled(training):
                    innovation = torch.from_numpy(nu) - torch.from_numpy(H) @ shift
                    features, length = build_network_input(previous, model.measurements.sat_ids, innovation, accel,
                                                           gyro, network.max_measurements, network.input_normalization)
                    gain, hidden = network(features.float(), length, count, hidden)
                    dx_tensor = gain[:, :count].double() @ innovation
                K, dx = gain.detach().double().numpy()[:, :count], dx_tensor.detach().numpy()
            if covariance:
                P_posterior = joseph_covariance(P, K, H, R)
                if fault_detection:
                    faulty, hypotheses, Q_inverse = find_faults(nu, H, R, P, model.projector)
                    if faulty:
                        dx, P_posterior = adapt_to_faults(dx, P, K, H, R, nu, model.projector, faulty, Q_inverse)
                        faulty_satellite = ' '.join('/'.join(model.measurements.sat_ids[h]) for h in hypotheses)
                P = P_posterior
            gain_square_sum += np.sum(K ** 2, axis=1)
            gain_columns += count

        if network is not None:
            with torch.set_grad_enabled(training):
                prior_error = torch.from_numpy(state_difference(truth_state(data, k), state)[:3])
                error = prior_error - shift[:3] - dx_tensor[:3]           # p_k - (p_k,k-1 + K dy_k)
                loss = error @ error / (error.numel() * (last - first))   # MSE of Eq. (30), Table III
                window_loss = window_loss + loss
                link = link + dx_tensor
            loss_sum += loss.item()

        new_state = apply_correction(state, dx)
        if network is not None or input_samples is not None:   # network input of epoch k+1, paper Eq. (11)-(14)
            with torch.set_grad_enabled(training):
                after = None if model is None else measurement_model(
                    new_state, model.measurements, data.lever_arm, times[k], data.klobuchar_alpha, data.klobuchar_beta)
                new_shift = link - link.detach()
                residuals = ({} if after is None else dict(zip(
                    after.measurements.sat_ids, torch.from_numpy(after.innovation) - torch.from_numpy(after.H) @ new_shift)))
                previous = {'accel': accel, 'gyro': gyro, 'state': new_state, 'residuals': residuals, 'shift': new_shift,
                            'state_residual': (torch.from_numpy(state_difference(new_state, previous['state']))
                                               + new_shift - previous['shift']),                        # Eq. (13)
                            'state_innovation': torch.from_numpy(dx) + dx_tensor - dx_tensor.detach()}  # Eq. (12)
        state = new_state

        if training and (k % cfg.BACKPROP_WINDOW == 0 or k == last):
            if torch.is_tensor(window_loss) and window_loss.requires_grad:
                window_loss.backward()
            window_loss, link = 0.0, no_shift
            hidden = None if hidden is None else tuple(h.detach() for h in hidden)
            previous = {name: value.detach() if torch.is_tensor(value) else value for name, value in previous.items()}
            previous['residuals'] = {s: value.detach() for s, value in previous['residuals'].items()}
            previous['shift'] = no_shift

        antenna, truth = state.position + state.attitude @ data.lever_arm, data.truth_antenna_position[k]
        rows.append((times[k], ecef_to_ned_matrix(*ecef_to_llh(truth)[:2]) @ (antenna - truth),
                     *(protection_levels(P, state, data.lever_arm) if covariance else (np.nan, np.nan)), count,
                     faulty_satellite))

    ned_error = np.array([r[1] for r in rows])
    return {
        'time': np.array([r[0] for r in rows]),
        'ned_error': ned_error,                                          # estimate - truth, antenna [m]
        'position_rmse_m': float(np.sqrt(np.mean(np.sum(ned_error ** 2, axis=1)))),
        'horizontal_pl': np.array([r[2] for r in rows]),
        'vertical_pl': np.array([r[3] for r in rows]),
        'measurement_count': np.array([r[4] for r in rows]),
        'faulty_satellite': [r[5] for r in rows],
        'loss': loss_sum,                                                # Eq. (30) mean (network only)
        'gain_row_rms': np.sqrt(gain_square_sum / max(gain_columns, 1)),
    }
