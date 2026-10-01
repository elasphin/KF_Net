"""GNSS/LEO/INS filter of paper Fig. 2 with a traditional EKF gain or the Masked-CLA gain.

One function runs the filter for training, validation and test, so the network
always sees the same filter. One fusion epoch k: INS propagation ->
measurements -> Kalman gain -> fault detection on the INS-predicted innovation
(Eq. (33)) -> state update, with Eq. (34) adaptation when a fault is identified.
"""
import numpy as np
import torch

import settings as cfg
from earth_models import ecef_to_llh, ecef_to_ned_matrix
from fault_detection import adapt_to_fault, detect_fault, identify_fault, protection_levels
from ins_filter import (STATE_SIZE, apply_correction, classical_gain, error_matrix, initial_covariance,
                        joseph_covariance, measurement_model, process_noise, propagate_ins, state_difference,
                        transition_matrix, truth_state)
from masked_cla_network import build_network_input


def run_filter(data, measurements, network=None, first=0, last=None, fault_detection=True, training=False,
               max_measurements=None):
    """Filter over fusion epochs first..last, starting from the truth at 'first' (A7).

    network=None: traditional EKF gain, with at most max_measurements rows (A16).
    training=True: accumulates the gradient of the Eq. (32) loss. The gradient of a correction dx_k also reaches later epochs
    through the linear error propagation Phi (a correction at k shifts the prior
    at k+1 by Phi dx_k), truncated every BACKPROP_WINDOW epochs (A15).
    The covariance P is used only by the EKF gain, the fault detection and the protection
    levels; a network run without fault detection (training, validation) leaves it out and
    returns NaN protection levels.
    """
    last = len(data.fusion_times) - 1 if last is None else last
    times = data.fusion_times
    state, P = truth_state(data, first), initial_covariance()
    i = np.searchsorted(data.imu_times, times[first], side='right')
    previous = {'accel': data.accel[i], 'gyro': data.gyro[i], 'state': state, 'residuals': {},
                'state_residual': np.zeros(STATE_SIZE), 'state_innovation': np.zeros(STATE_SIZE)}
    hidden, link, window_loss, loss_sum = None, torch.zeros(STATE_SIZE, dtype=torch.float64), 0.0, 0.0
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
        model = measurement_model(state, measurements[k], data.lever_arm, times[k], data.klobuchar_alpha,
                                  data.klobuchar_beta, max_measurements if network is None else network.max_measurements)
        dx, faulty_satellite, count = np.zeros(STATE_SIZE), '', 0
        if model is not None:
            nu, H, R = model.innovation, model.H, model.R
            count = len(nu)
            if network is None:
                K = classical_gain(P, H, R)
                dx = K @ nu
            else:
                features, length = build_network_input(previous, model.measurements.sat_ids, nu, accel, gyro,
                                                       network.max_measurements)
                with torch.set_grad_enabled(training):
                    gain, hidden = network(torch.tensor(features, dtype=torch.float32), length, count, hidden)
                    dx_tensor = gain[:, :count].double() @ torch.from_numpy(nu)
                    prior_error = torch.from_numpy(state_difference(truth_state(data, k), state)[:3])
                    error = prior_error - (link - link.detach())[:3] - dx_tensor[:3]   # p_k - (p_k,k-1 + K dy_k)
                    loss = error @ error / (error.numel() * (last - first))   # MSE of Eq. (30), Table III
                    window_loss = window_loss + loss
                    link = link + dx_tensor
                loss_sum += loss.item()
                K, dx = gain.detach().double().numpy()[:, :count], dx_tensor.detach().numpy()
            if covariance:
                P_posterior = joseph_covariance(P, K, H, R)
                if fault_detection:
                    detected, Q, Q_inverse = detect_fault(nu, H, R, P)
                    if detected:
                        index = identify_fault(nu, model.projector, Q_inverse)
                        dx, P_posterior = adapt_to_fault(dx, P_posterior, K, nu, model.projector, index, Q, Q_inverse)
                        faulty_satellite = model.measurements.sat_ids[index]
                P = P_posterior
            gain_square_sum += np.sum(K ** 2, axis=1)
            gain_columns += count

        new_state = apply_correction(state, dx)
        if model is not None:                          # network input of epoch k+1, paper Eq. (11)-(14)
            after = measurement_model(new_state, model.measurements, data.lever_arm, times[k],
                                      data.klobuchar_alpha, data.klobuchar_beta)
            previous = {'accel': accel, 'gyro': gyro, 'state': new_state,
                        'residuals': {} if after is None else dict(zip(after.measurements.sat_ids, after.innovation)),
                        'state_residual': state_difference(new_state, previous['state']),       # Eq. (13)
                        'state_innovation': dx}                                                   # Eq. (12)
        state = new_state

        if training and (k % cfg.BACKPROP_WINDOW == 0 or k == last):
            if torch.is_tensor(window_loss):
                window_loss.backward()
            window_loss, link = 0.0, torch.zeros(STATE_SIZE, dtype=torch.float64)
            hidden = None if hidden is None else tuple(h.detach() for h in hidden)

        antenna, truth = state.position + state.attitude @ data.lever_arm, data.truth_antenna_position[k]
        rows.append((times[k], ecef_to_ned_matrix(*ecef_to_llh(truth)[:2]) @ (antenna - truth),
                     *(protection_levels(P, state.position) if covariance else (np.nan, np.nan)), count,
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
