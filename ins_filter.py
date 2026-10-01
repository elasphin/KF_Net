"""15-state GNSS/LEO/INS tightly coupled error-state filter in ECEF.

State (paper Eq. (7)): x = [dp, dv, dtheta, b_a, b_g], error = truth - estimate,
C_true = Exp(dtheta) C_estimate. System matrix F from paper Ref. [38] Eq. (3)
(clock rows removed, because Eq. (7) has no clock state).
The *_numba function is a Numba-compiled copy of the Python function that follows it (see earth_models.py).
"""
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
from numba import njit
from scipy.linalg import expm

import settings as cfg
from earth_models import (EARTH_ROTATION_RATE, EARTH_ROTATION_VECTOR, gravity, gravity_numba, rotation_matrix_to_vector,
                          rotation_vector_to_matrix, rotation_vector_to_matrix_numba, skew, skew_numba)
from gnss_measurements import SYSTEM_ORDER, EpochMeasurements, predict_pseudoranges

STATE_SIZE = 15
EARTH_ROTATION_SKEW = skew(EARTH_ROTATION_VECTOR)


@dataclass
class NavigationState:
    position: np.ndarray        # IMU position, ECEF [m]
    velocity: np.ndarray        # ECEF [m/s]
    attitude: np.ndarray        # C_ecef<-body
    accel_bias: np.ndarray      # [m/s^2]
    gyro_bias: np.ndarray       # [rad/s]


def initial_covariance():
    return np.diag(np.concatenate([np.full(3, cfg.INITIAL_POSITION_STD), np.full(3, cfg.INITIAL_VELOCITY_STD),
                                   np.full(3, cfg.INITIAL_ATTITUDE_STD), np.full(3, cfg.ACCEL_BIAS_INSTABILITY),
                                   np.full(3, cfg.GYRO_BIAS_INSTABILITY)]) ** 2)


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

    Returns the new state, the mean measured specific force (for F) and the raw
    IMU sample at end_time (alpha_k, w_k of paper Eq. (14)).
    """
    first = np.searchsorted(data.imu_times, start_time, side='right')
    last = np.searchsorted(data.imu_times, end_time, side='right')      # samples [first, last) are <= end_time
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
    samples = slice(first, last + 1)
    return state, data.accel[samples].mean(axis=0), data.accel[last], data.gyro[last]


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
    """Tightly coupled pseudorange model (paper Eq. (9)-(10)); None if no usable row."""
    lever_ecef = state.attitude @ lever_arm
    antenna = state.position + lever_ecef
    predicted, line_of_sight, elevation, variance = predict_pseudoranges(antenna, meas, time, alpha, beta)
    rows = np.flatnonzero(elevation > 0.0)
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

    Used as the initial state and its position as the label of the loss.
    """
    return NavigationState(data.truth_position[k], data.truth_velocity[k], data.truth_attitude[k],
                           np.zeros(3), np.zeros(3))
