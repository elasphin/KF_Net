"""Online GNSS/LEO/INS filter of paper Fig. 2, with a classical or a Masked-CLA Kalman gain.

One fusion epoch k: INS propagation -> measurements -> Kalman gain ->
fault detection on the INS-predicted innovation (Eq. (33)) -> state update,
with Eq. (34) adaptation when a fault is identified.
"""
import numpy as np
import torch

import settings as cfg
from fault_detection import adapt_to_fault, detect_fault, identify_fault, protection_levels
from gnss_measurements import merge_measurements, prepare_gnss_measurements
from ins_filter import (STATE_SIZE, apply_correction, classical_gain, error_matrix, initial_covariance, initial_state,
                        joseph_covariance, measurement_model, propagate_ins, state_difference, transition_matrix)
from leo_simulation import real_error_bins, simulate_leo_measurements
from masked_cla_network import build_network_input


def prepare_measurements(data, split):
    """GPS + BDS-3 (real) and LEO (simulated) measurements of every fusion epoch."""
    gnss = prepare_gnss_measurements(data)
    leo = simulate_leo_measurements(data, real_error_bins(data, gnss), cfg.LEO_NOISE_SEED[split])
    return [merge_measurements(g, l) for g, l in zip(gnss, leo)]


def first_epoch_features(data, state):
    """Network input quantities before the first update (zero differences)."""
    i = np.searchsorted(data.imu_times, data.fusion_times[0], side='right')
    return {'accel': data.accel[i], 'gyro': data.gyro[i], 'state': state, 'residuals': {},
            'state_residual': np.zeros(STATE_SIZE), 'state_innovation': np.zeros(STATE_SIZE)}


def next_epoch_features(data, k, previous, new_state, model, dx, accel, gyro):
    """Quantities of epoch k used by the network at epoch k+1 (paper Eq. (11)-(14))."""
    after = measurement_model(new_state, model.measurements, data.lever_arm, data.fusion_times[k],
                              data.klobuchar_alpha, data.klobuchar_beta)
    residuals = {} if after is None else dict(zip(after.measurements.sat_ids, after.innovation))
    return {'accel': accel, 'gyro': gyro, 'state': new_state, 'residuals': residuals,
            'state_residual': state_difference(new_state, previous['state']),       # Eq. (13)
            'state_innovation': dx}                                                   # Eq. (12)


def run_filter(data, measurements, network=None, use_fault_detection=True):
    """Run the filter over all fusion epochs. network=None gives the traditional EKF."""
    state, P = initial_state(data), initial_covariance()
    previous = first_epoch_features(data, state)
    hidden = None
    max_count = None if network is None else network.max_measurements
    times = data.fusion_times
    rows, gain_square_sum, gain_columns = [], np.zeros(STATE_SIZE), 0
    for k in range(1, len(times)):
        state, mean_force, accel, gyro = propagate_ins(state, data, times[k - 1], times[k])
        Phi, Qd = transition_matrix(error_matrix(state, mean_force), times[k] - times[k - 1])
        P = Phi @ P @ Phi.T + Qd
        model = measurement_model(state, measurements[k], data.lever_arm, times[k], data.klobuchar_alpha,
                                  data.klobuchar_beta, max_count)
        dx, faulty_satellite, count = np.zeros(STATE_SIZE), '', 0
        if model is not None:
            nu, H, R = model.innovation, model.H, model.R
            count = len(nu)
            if network is None:
                K = classical_gain(P, H, R)
            else:
                features, length = build_network_input(previous, model.measurements.sat_ids, nu, accel, gyro,
                                                       max_count)
                with torch.no_grad():
                    gain, hidden = network(torch.tensor(features, dtype=torch.float32), length, count, hidden)
                K = gain.double().numpy()[:, :count]
            dx = K @ nu
            P_posterior = joseph_covariance(P, K, H, R)
            if use_fault_detection:
                detected, Q, Q_inverse = detect_fault(nu, H, R, P)
                if detected:
                    index = identify_fault(nu, model.projector, Q_inverse)
                    dx, P_posterior = adapt_to_fault(dx, P_posterior, K, nu, model.projector, index, Q, Q_inverse)
                    faulty_satellite = model.measurements.sat_ids[index]
            P = P_posterior
            gain_square_sum += np.sum(K ** 2, axis=1)
            gain_columns += count
        new_state = apply_correction(state, dx)
        if model is not None:
            previous = next_epoch_features(data, k, previous, new_state, model, dx, accel, gyro)
        state = new_state
        horizontal_pl, vertical_pl = protection_levels(P, state.position)
        rows.append((times[k], state.position + state.attitude @ data.lever_arm, data.truth_antenna_position[k],
                     horizontal_pl, vertical_pl, count, faulty_satellite))
    return {
        'time': np.array([r[0] for r in rows]),
        'estimate': np.array([r[1] for r in rows]),
        'truth': np.array([r[2] for r in rows]),
        'horizontal_pl': np.array([r[3] for r in rows]),
        'vertical_pl': np.array([r[4] for r in rows]),
        'measurement_count': np.array([r[5] for r in rows]),
        'faulty_satellite': [r[6] for r in rows],
        'gain_row_rms': np.sqrt(gain_square_sum / max(gain_columns, 1)),
    }
