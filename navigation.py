"""15-state GNSS/LEO/INS tightly coupled error-state filter in ECEF.

State (paper Eq. (7)): x = [dp, dv, dtheta, b_a, b_g], error = truth - estimate,
C_true = Exp(dtheta) C_estimate. System matrix F from paper Ref. [38] Eq. (3)
(clock rows removed, because Eq. (7) has no clock state).
The *_numba function is a Numba-compiled copy of the Python function that follows it (see measurements.py).

Masked CLA KalmanNet (paper Sec. II-B, Fig. 8, Table III) and its input vector.

GNSS/LEO/INS filter of paper Fig. 2 with a traditional EKF gain or the Masked-CLA gain.

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
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import torch
import torch.nn.functional as F
from numba import njit
from scipy.linalg import expm
from scipy.stats import chi2
from torch import nn

import settings as cfg
from measurements import (EARTH_ROTATION_RATE, EARTH_ROTATION_VECTOR, SYSTEM_ORDER, EpochMeasurements, ecef_to_llh,
                          ecef_to_ned_matrix, gravity, gravity_numba, predict_pseudoranges, rotation_matrix_to_vector,
                          rotation_vector_to_matrix, rotation_vector_to_matrix_numba, skew, skew_numba)


# ===== 15-state INS error-state filter ================================================================================
STATE_SIZE = 15
EARTH_ROTATION_SKEW = skew(EARTH_ROTATION_VECTOR)


@dataclass
class NavigationState:
    position: np.ndarray        # IMU position, ECEF [m]
    velocity: np.ndarray        # ECEF [m/s]
    attitude: np.ndarray        # C_ecef<-body
    accel_bias: np.ndarray      # [m/s^2]
    gyro_bias: np.ndarray       # [rad/s]


def initial_covariance(data):
    """P_0 (A7): truth accuracy, settings for velocity and attitude, bias sigmas of the dataset's IMU error model."""
    return np.diag(np.concatenate([np.full(3, cfg.INITIAL_POSITION_STD), np.full(3, cfg.INITIAL_VELOCITY_STD),
                                   np.full(3, cfg.INITIAL_ATTITUDE_STD), data.accel_bias_std,
                                   data.gyro_bias_std]) ** 2)


def noise_density():
    """Continuous process noise (paper Table II): white noise on v and theta, random walk on the biases (A6)."""
    tau = cfg.BIAS_CORRELATION_TIME
    return np.diag(np.concatenate([np.zeros(3), np.full(3, cfg.ACCEL_WHITE_NOISE ** 2),
                                   np.full(3, cfg.GYRO_WHITE_NOISE ** 2),
                                   np.full(3, 2.0 * cfg.ACCEL_BIAS_INSTABILITY ** 2 / tau),
                                   np.full(3, 2.0 * cfg.GYRO_BIAS_INSTABILITY ** 2 / tau)]))


# --- INS mechanization -------------------------------------------------------
@njit(cache=True)
def mechanize_samples_numba(position, velocity, attitude, accel_bias, gyro_bias, gyro, accel, steps):
    """mechanize for every row of gyro, accel with the time steps 'steps' -> position, velocity, attitude."""
    earth_rotation = np.array([0.0, 0.0, EARTH_ROTATION_RATE])
    earth_skew = skew_numba(earth_rotation)
    r, v, C = position.copy(), velocity.copy(), attitude.copy()
    for i in range(steps.shape[0]):
        dt = steps[i]
        angular_rate = gyro[i] - gyro_bias
        specific_force = accel[i] - accel_bias
        new_C = (rotation_vector_to_matrix_numba(-earth_rotation * dt) @ C
                 @ rotation_vector_to_matrix_numba(angular_rate * dt))
        acceleration = (0.5 * (C + new_C) @ specific_force + gravity_numba(r)
                        - earth_skew @ earth_skew @ r - 2.0 * earth_skew @ v)
        new_v = v + acceleration * dt
        r = r + 0.5 * (v + new_v) * dt
        v, C = new_v, new_C
    return r, v, C


def mechanize(state: NavigationState, gyro, accel, dt) -> NavigationState:
    """ECEF strapdown step with bias compensation (paper Eq. (8), S = M = 0, see A8)."""
    angular_rate = gyro - state.gyro_bias
    specific_force = accel - state.accel_bias
    new_attitude = (rotation_vector_to_matrix(-EARTH_ROTATION_VECTOR * dt) @ state.attitude
                    @ rotation_vector_to_matrix(angular_rate * dt))
    r, v = state.position, state.velocity
    acceleration = (0.5 * (state.attitude + new_attitude) @ specific_force + gravity(r)
                    - EARTH_ROTATION_SKEW @ EARTH_ROTATION_SKEW @ r - 2.0 * EARTH_ROTATION_SKEW @ v)
    new_velocity = v + acceleration * dt
    return NavigationState(r + 0.5 * (v + new_velocity) * dt, new_velocity, new_attitude,
                           state.accel_bias, state.gyro_bias)


def propagate_ins(state, data, start_time, end_time):
    """INS from start_time to end_time with every IMU sample in between (paper Fig. 7).

    Each IMU sample is the mean rate over the interval that ends at its time tag, so the step up to
    end_time inside the next interval uses the next sample (the one whose interval contains end_time).
    Returns the new state, the mean measured specific force over those samples (for F) and the
    last IMU sample at or before end_time (alpha_k, w_k of paper Eq. (14): no later sample is used).
    """
    first = np.searchsorted(data.imu_times, start_time, side='right')
    last = np.searchsorted(data.imu_times, end_time, side='right')      # samples [first, last) are <= end_time
    if last == 0 or (last == len(data.imu_times) and end_time > data.imu_times[-1]):
        raise ValueError(f'IMU samples ({data.imu_times[0]:.3f}..{data.imu_times[-1]:.3f}) do not cover '
                         f'{start_time:.3f}..{end_time:.3f}')
    if cfg.INS_MECHANIZATION == 'python':
        time = start_time
        for i in range(first, last):
            state = mechanize(state, data.gyro[i], data.accel[i], data.imu_times[i] - time)
            time = data.imu_times[i]
        if end_time > time:                                             # partial step up to the GNSS epoch
            state = mechanize(state, data.gyro[last], data.accel[last], end_time - time)
    elif cfg.INS_MECHANIZATION == 'numba':                              # the same steps, compiled
        index, steps = np.arange(first, last), np.diff(data.imu_times[first:last], prepend=start_time)
        time = data.imu_times[last - 1] if last > first else start_time
        if end_time > time:
            index, steps = np.append(index, last), np.append(steps, end_time - time)
        position, velocity, attitude = mechanize_samples_numba(
            state.position, state.velocity, state.attitude, state.accel_bias, state.gyro_bias,
            data.gyro[index], data.accel[index], steps)
        state = NavigationState(position, velocity, attitude, state.accel_bias, state.gyro_bias)
    else:
        raise ValueError(f"INS_MECHANIZATION must be 'python' or 'numba', not {cfg.INS_MECHANIZATION!r}")
    samples = slice(first, min(last + 1, len(data.imu_times)))
    return state, data.accel[samples].mean(axis=0), data.accel[last - 1], data.gyro[last - 1]


def error_matrix(state: NavigationState, specific_force):
    """System matrix F of paper Eq. (6), structure of Ref. [38] Eq. (3)."""
    F = np.zeros((STATE_SIZE, STATE_SIZE))
    r = state.position
    radius = np.linalg.norm(r)
    C = state.attitude
    F[0:3, 3:6] = np.eye(3)
    F[3:6, 0:3] = -2.0 / radius * np.outer(gravity(r), r / radius)
    F[3:6, 3:6] = -2.0 * EARTH_ROTATION_SKEW
    F[3:6, 6:9] = -skew(C @ (specific_force - state.accel_bias))
    F[3:6, 9:12] = -C
    F[6:9, 6:9] = -EARTH_ROTATION_SKEW
    F[6:9, 12:15] = -C
    return F


def transition_matrix(F, dt):
    """Phi = sum F^n dt^n / n! = expm(F dt), Ref. [38] Eq. (4)."""
    return expm(F * dt)


def process_noise(Phi, dt):
    """Qd by the trapezoidal rule (A20)."""
    Q = noise_density()
    return 0.5 * (Phi @ Q @ Phi.T + Q) * dt


# --- Measurement model -------------------------------------------------------
class MeasurementModel(NamedTuple):
    measurements: EpochMeasurements   # rows used (after elevation, capacity and clock checks)
    innovation: np.ndarray            # clock-free y - h(x), paper Eq. (10)
    H: np.ndarray                     # (n, 15)
    R: np.ndarray                     # (n, n)
    projector: np.ndarray             # receiver-clock projector (A5)


CLOCK_SYSTEMS = ('G', 'C')            # one receiver clock each (A5); LEO clocks are ideal (A4)


def clock_projector(systems, variance):
    """Removes one receiver clock per GNSS system: weighted least-squares clock estimate (A5)."""
    P = np.eye(len(systems))
    for system in CLOCK_SYSTEMS:
        rows = np.flatnonzero(systems == system)
        weights = 1.0 / variance[rows]
        P[np.ix_(rows, rows)] -= np.outer(np.ones(len(rows)), weights / weights.sum())
    return P


def measurement_model(state, meas: EpochMeasurements, lever_arm, time, alpha, beta, max_count=None):
    """Tightly coupled pseudorange model (paper Eq. (9)-(10)); None if no usable row.

    GPS/BDS-3 rows at or above GNSS_ELEVATION_MASK_DEG (A10); LEO rows above the horizon (their mask is
    applied when they are simulated, A2).
    """
    lever_ecef = state.attitude @ lever_arm
    antenna = state.position + lever_ecef
    predicted, line_of_sight, elevation, variance = predict_pseudoranges(antenna, meas, time, alpha, beta)
    mask = np.where(meas.systems == 'L', 0.0, np.deg2rad(cfg.GNSS_ELEVATION_MASK_DEG))  # LEO: mask of the simulation
    rows = np.flatnonzero((elevation > 0.0) & (elevation >= mask))
    if max_count is not None and len(rows) > max_count:                  # A16: keep the highest satellites
        rows = np.sort(rows[np.argsort(-elevation[rows])[:max_count]])
    counts = {s: np.sum(meas.systems[rows] == s) for s in SYSTEM_ORDER}
    rows = rows[[s not in CLOCK_SYSTEMS or counts[s] >= 2 for s in meas.systems[rows]]]   # a clock needs two rows
    if len(rows) == 0:
        return None
    H = np.zeros((len(rows), STATE_SIZE))
    H[:, 0:3] = line_of_sight[rows]
    H[:, 6:9] = -line_of_sight[rows] @ skew(lever_ecef)
    projector = clock_projector(meas.systems[rows], variance[rows])
    R = projector @ np.diag(variance[rows]) @ projector.T
    innovation = projector @ (meas.pseudoranges[rows] - predicted[rows])
    return MeasurementModel(meas.subset(rows), innovation, projector @ H, R, projector)


# --- Update ------------------------------------------------------------------
def classical_gain(P, H, R):
    """Kalman gain of the traditional EKF (used as baseline and for the gain scale, A11)."""
    return P @ H.T @ np.linalg.pinv(H @ P @ H.T + R, hermitian=True)


def joseph_covariance(P, K, H, R):
    """Posterior covariance, valid for any gain K (network or classical)."""
    A = np.eye(STATE_SIZE) - K @ H
    return A @ P @ A.T + K @ R @ K.T


def apply_correction(state: NavigationState, dx) -> NavigationState:
    return NavigationState(state.position + dx[0:3], state.velocity + dx[3:6],
                           rotation_vector_to_matrix(dx[6:9]) @ state.attitude,
                           state.accel_bias + dx[9:12], state.gyro_bias + dx[12:15])


def state_difference(a: NavigationState, b: NavigationState):
    """15-state difference a - b."""
    return np.concatenate([a.position - b.position, a.velocity - b.velocity,
                           rotation_matrix_to_vector(a.attitude @ b.attitude.T),
                           a.accel_bias - b.accel_bias, a.gyro_bias - b.gyro_bias])


def truth_state(data, k) -> NavigationState:
    """Post-processed truth at fusion epoch k, zero biases (the truth has no biases; A7).

    Used as the initial state and as the labels of the loss: position, with LOSS = 'pva' also velocity and attitude (A28).
    """
    return NavigationState(data.truth_position[k], data.truth_velocity[k], data.truth_attitude[k],
                           np.zeros(3), np.zeros(3))


# ===== Masked CLA network and its input ===============================================================================
FIXED_FEATURE_SIZE = 6 + 2 * STATE_SIZE      # [d_alpha(3), d_w(3), dx_hat(15), dx_tilde(15)] = 36


def build_network_input(previous, sat_ids, innovation, accel, gyro, max_measurements):
    """Paper Eq. (15)-(16): X_k = [d_alpha, d_w, dx_hat, dx_tilde, dy, dy_tilde] followed by zero padding.

    previous: quantities of epoch k-1 ('accel', 'gyro', 'state_residual' Eq. (13),
    'state_innovation' Eq. (12), 'residuals' Eq. (11) as {sat_id: value}).
    A satellite that was not used at k-1 has lagged residual 0 (docs/ASSUMPTIONS.md A12).
    The state and measurement quantities are float64 tensors; in training they carry the gradient of the
    earlier corrections (navigation.run_filter, A15).
    Returns the padded vector (tensor, length 36 + 2 N_max) and the number of valid entries (mask, Eq. (17)).
    """
    zero = torch.zeros((), dtype=torch.float64)
    lagged_residual = [previous['residuals'].get(s, zero) for s in sat_ids]
    values = torch.cat([torch.from_numpy(accel - previous['accel']), torch.from_numpy(gyro - previous['gyro']),  # Eq. (14)
                        previous['state_residual'], previous['state_innovation'],                            # Eq. (13), (12)
                        torch.stack(lagged_residual), innovation])                                           # Eq. (11), (10)
    return F.pad(values, (0, FIXED_FEATURE_SIZE + 2 * max_measurements - len(values))), len(values)


class MaskedCLANetwork(nn.Module):
    """Masked CNN -> masked LSTM -> masked attention -> masked FC -> Kalman gain (15 x N_max)."""

    def __init__(self, max_measurements):
        super().__init__()
        self.max_measurements = max_measurements
        self.conv = nn.Conv1d(1, cfg.CONV_FILTERS, cfg.CONV_KERNEL_SIZE, stride=1, bias=False)
        self.conv_bias = nn.Parameter(torch.zeros(cfg.CONV_FILTERS))                    # b of Eq. (22)
        self.lstm = nn.LSTM(cfg.CONV_FILTERS, cfg.LSTM_UNITS, num_layers=cfg.LSTM_LAYERS,
                            dropout=cfg.LSTM_DROPOUT, batch_first=True)
        self.attention_hidden = nn.Linear(cfg.LSTM_UNITS, cfg.LSTM_UNITS)                # W_h, b_h of Eq. (26)
        self.attention_vector = nn.Linear(cfg.LSTM_UNITS, 1, bias=False)                 # v of Eq. (26)
        self.fc_hidden = nn.Linear(cfg.LSTM_UNITS, cfg.FC_HIDDEN_UNITS)
        self.fc_output = nn.Linear(cfg.FC_HIDDEN_UNITS, STATE_SIZE * max_measurements)
        nn.init.zeros_(self.fc_output.weight)            # K = 0 before training (A12)
        nn.init.zeros_(self.fc_output.bias)
        # Weights of the errors in the loss Eq. (30): position only (LOSS = 'p', paper) or position, velocity and
        # attitude (LOSS = 'pva', A28); set by train.py (train.loss_weights), kept with the model.
        self.register_buffer('loss_weight', torch.ones(3 if cfg.LOSS == 'p' else 9, dtype=torch.float64))
        self.register_buffer('gain_row_scale', torch.ones(STATE_SIZE))   # K = diag(scale) K_net, set by train.py (A11)

    def forward(self, features, valid_length, measurement_count, hidden=None):
        """features: padded X_k [D]; returns the Kalman gain [15, N_max] and the LSTM state (h, c)."""
        D = features.shape[0]
        mask = (torch.arange(D) < valid_length).to(features.dtype).view(1, 1, D)       # M_k, Eq. (17), (21)
        pad = cfg.CONV_KERNEL_SIZE // 2

        # Masked CNN, Eq. (22): Z / max(N_t, eps) + 1[N_t > 0] b, N_t = valid samples in the window.
        z = self.conv(F.pad(features.view(1, 1, D) * mask, (pad, pad)))
        valid_count = F.conv1d(F.pad(mask, (pad, pad)), torch.ones(1, 1, cfg.CONV_KERNEL_SIZE))
        s = F.relu(z / valid_count.clamp_min(cfg.MASK_EPSILON)
                   + (valid_count > 0).to(z.dtype) * self.conv_bias.view(1, -1, 1)) * mask
        # Pooling (Fig. 8) with stride 1 over valid samples, so that M_out = M_in (Eq. (23)).
        pooled = F.max_pool1d(s.masked_fill(mask == 0, float('-inf')), cfg.POOL_KERNEL_SIZE, stride=1,
                              padding=cfg.POOL_KERNEL_SIZE // 2)
        s = torch.where(mask > 0, pooled, torch.zeros_like(pooled))

        # Masked LSTM, Eq. (24)-(25): steps with M = 0 keep the previous (h, c). They are all at the end
        # of X_k, so running the LSTM on the valid steps only is exactly Eq. (25).
        valid_steps = s[:, :, :valid_length].transpose(1, 2)                            # [1, L, 24]
        outputs, hidden = self.lstm(valid_steps, hidden)

        # Masked attention, Eq. (26)-(29): steps with M = 0 have score -inf, i.e. are left out of the softmax.
        scores = self.attention_vector(torch.tanh(self.attention_hidden(outputs[0]))).squeeze(1)
        weights = torch.softmax(scores, dim=0)
        context = weights @ outputs[0]

        # Masked FC: Kalman gain, columns j >= N_k are zero.
        gain = self.fc_output(F.relu(self.fc_hidden(context))).view(STATE_SIZE, self.max_measurements)
        column_mask = (torch.arange(self.max_measurements) < measurement_count).to(gain.dtype)
        return gain * column_mask * self.gain_row_scale.view(STATE_SIZE, 1), hidden


# ===== Filter of paper Fig. 2, fault detection, integrity =============================================================
PARALLEL_TOLERANCE = 1e-6     # relative: c_i parallel to an identified fault, or already explained by it (above the
                              # rounding of M, ~1e-8, so an explained row cannot be identified again)


# --- Fault detection and integrity (paper Sec. II-D, Ref. [33]) --------------
def find_faults(innovation, H, R, P_prior, projector):
    """Fault detection and identification, repeated after each identified fault (A27).

    Paper Eq. (33) / Ref. [33] Eq. (6): T = nu^T M nu > chi2(1 - alpha, rank Q - q), Q = H P H^T + R,
    with M = Q^+ (I - C C^+) the metric after the q faults already found (columns of C, C^+ of
    Ref. [33] Eq. (39)); M = Q^+ when q = 0. Ref. [33] Eq. (7): the next fault is the row with the
    largest T_i = (c_i^T M nu)^2 / (c_i^T M c_i), c_i = projector column i (pseudorange fault seen in
    the clock-free innovation), among the rows not yet identified and not already explained by the faults
    found (c_i^T M c_i > PARALLEL_TOLERANCE c_i^T Q^+ c_i); the search stops when no such row is left. Rows
    with c_i parallel to that of the identified row are the same hypothesis of Ref. [33] (two rows of one
    GNSS system) and are named with it.
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
        testable[faulty] = False
        if not testable.any():                                             # no row can explain the rest
            break
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

    The antenna error is dp - (C l) x dtheta (navigation.measurement_model), so its covariance is J P J^T.
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
               max_measurements=None):
    """Filter over fusion epochs first..last, starting from the truth at 'first' (A7).

    network=None: traditional EKF gain, with at most max_measurements rows (A16).
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
                K = classical_gain(P, H, R)
                dx = K @ nu
            else:
                with torch.set_grad_enabled(training):
                    innovation = torch.from_numpy(nu) - torch.from_numpy(H) @ shift
                    features, length = build_network_input(previous, model.measurements.sat_ids, innovation, accel,
                                                           gyro, network.max_measurements)
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
                n = network.loss_weight.numel()           # 3: position (paper), 9: position, velocity, attitude (A28)
                prior_error = torch.from_numpy(state_difference(truth_state(data, k), state)[:n])
                error = (prior_error - shift[:n] - dx_tensor[:n]) * network.loss_weight   # x_k - (x_k,k-1 + K dy_k)
                loss = error @ error / (error.numel() * (last - first))   # MSE of Eq. (30), Table III
                window_loss = window_loss + loss
                link = link + dx_tensor
            loss_sum += loss.item()

        new_state = apply_correction(state, dx)
        if network is not None:                        # network input of epoch k+1, paper Eq. (11)-(14)
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
                     faulty_satellite, state_difference(truth_state(data, k), state)[:9]))

    ned_error = np.array([r[1] for r in rows])
    return {
        'time': np.array([r[0] for r in rows]),
        'ned_error': ned_error,                                          # estimate - truth, antenna [m]
        'position_rmse_m': float(np.sqrt(np.mean(np.sum(ned_error ** 2, axis=1)))),
        'horizontal_pl': np.array([r[2] for r in rows]),
        'vertical_pl': np.array([r[3] for r in rows]),
        'measurement_count': np.array([r[4] for r in rows]),
        'faulty_satellite': [r[5] for r in rows],
        'state_error': np.array([r[6] for r in rows]),                  # truth - estimate [dp, dv, dtheta], IMU, ECEF
        'loss': loss_sum,                                                # Eq. (30) mean (network only)
        'gain_row_rms': np.sqrt(gain_square_sum / max(gain_columns, 1)),
    }
