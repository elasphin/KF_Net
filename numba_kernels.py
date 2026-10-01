"""Numba-compiled versions of the two slowest loops (settings INS_MECHANIZATION and LEO_FORCE_MODEL = 'numba').

Each function repeats the arithmetic of its Python original operation by operation and in the same
order, so both versions give the same numbers; only the Python interpreter overhead is removed:
    propagate_ins_samples    <-  ins_filter.mechanize, applied to every IMU sample of propagate_ins
    leo_equations_of_motion  <-  leo_orbits.equations_of_motion (EGM96 gravity, Sun and Moon)
The first call compiles the functions (a few seconds); cache=True keeps the compiled code in
__pycache__ for the next runs.
"""
import math

import numpy as np
from llvmlite import ir
from numba import njit, types
from numba.core import cgutils
from numba.extending import intrinsic

from earth_models import EARTH_GM, EARTH_J2, EARTH_ROTATION_RATE, GPS_UTC_LEAP_SECONDS, WGS84_A
from leo_orbits import ASTRONOMICAL_UNIT, EGM96_GM, EGM96_RADIUS, GPS_EPOCH_JULIAN_DATE, MOON_GM, SUN_GM


# --- Same operations as numpy ------------------------------------------------
@intrinsic
def power(typingctx, x, y):
    """x ** y with the C library pow(), as Python and numpy compute it.

    Numba would replace x ** 2 by x * x, which differs from pow(x, 2) in the last bit for ~0.1 % of
    the values; 'nobuiltin' keeps the call to pow().
    """
    def codegen(context, builder, signature, args):
        double = ir.DoubleType()
        function = cgutils.get_or_insert_function(builder.module, ir.FunctionType(double, [double, double]), 'pow')
        function.attributes.add('nobuiltin')
        return builder.call(function, args)
    return types.float64(types.float64, types.float64), codegen


@njit(cache=True)
def norm(v):
    """Vector norm as numpy.linalg.norm computes it: sqrt(v . v) with the BLAS dot product."""
    return np.sqrt(np.dot(v, v))


@njit(cache=True)
def skew(v):
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


@njit(cache=True)
def rotation_vector_to_matrix(v):
    """earth_models.rotation_vector_to_matrix."""
    angle = norm(v)
    K = skew(v)
    if angle < 1e-8:
        return np.eye(3) + K + 0.5 * K @ K
    return np.eye(3) + math.sin(angle) / angle * K + (1.0 - math.cos(angle)) / power(angle, 2.0) * K @ K


@njit(cache=True)
def gravity(position):
    """earth_models.gravity."""
    x, y, z = position[0], position[1], position[2]
    r = norm(position)
    j2 = 1.5 * EARTH_J2 * power(WGS84_A / r, 2.0)
    ratio = 5.0 * z * z / (r * r)
    return -EARTH_GM / power(r, 3.0) * np.array([x * (1.0 - j2 * (ratio - 1.0)),
                                             y * (1.0 - j2 * (ratio - 1.0)),
                                             z * (1.0 - j2 * (ratio - 3.0))])


@njit(cache=True)
def propagate_ins_samples(position, velocity, attitude, accel_bias, gyro_bias, gyro, accel, steps):
    """ins_filter.mechanize for every row of gyro, accel with the time steps 'steps' -> position, velocity, attitude."""
    earth_rotation = np.array([0.0, 0.0, EARTH_ROTATION_RATE])
    earth_skew = skew(earth_rotation)
    r, v, C = position.copy(), velocity.copy(), attitude.copy()
    for i in range(steps.shape[0]):
        dt = steps[i]
        angular_rate = gyro[i] - gyro_bias
        specific_force = accel[i] - accel_bias
        new_C = rotation_vector_to_matrix(-earth_rotation * dt) @ C @ rotation_vector_to_matrix(angular_rate * dt)
        acceleration = (0.5 * (C + new_C) @ specific_force + gravity(r)
                        - earth_skew @ earth_skew @ r - 2.0 * earth_skew @ v)
        new_v = v + acceleration * dt
        r = r + 0.5 * (v + new_v) * dt
        v, C = new_v, new_C
    return r, v, C


# --- LEO reference orbit (leo_orbits.py) -------------------------------------
@njit(cache=True)
def earth_rotation_angle(gps_time):
    """earth_models.earth_rotation_angle."""
    julian_date_utc = 2444244.5 + (gps_time - GPS_UTC_LEAP_SECONDS) / 86400.0
    degrees = 280.46061837 + 360.98564736629 * (julian_date_utc - 2451545.0)
    return np.deg2rad(degrees % 360.0)


@njit(cache=True)
def gravity_acceleration(r, C, S):
    """leo_orbits.gravity_acceleration with the coefficients C, S (leo_orbits.GRAVITY_C, GRAVITY_S)."""
    n_max = C.shape[0] - 1
    r2 = r @ r
    rho = power(EGM96_RADIUS, 2.0) / r2
    x0, y0, z0 = EGM96_RADIUS * r[0] / r2, EGM96_RADIUS * r[1] / r2, EGM96_RADIUS * r[2] / r2
    V, W = np.zeros((n_max + 2, n_max + 2)), np.zeros((n_max + 2, n_max + 2))
    V[0, 0] = EGM96_RADIUS / np.sqrt(r2)
    V[1, 0] = z0 * V[0, 0]
    for n in range(2, n_max + 2):
        V[n, 0] = ((2 * n - 1) * z0 * V[n - 1, 0] - (n - 1) * rho * V[n - 2, 0]) / n
    for m in range(1, n_max + 2):
        V[m, m] = (2 * m - 1) * (x0 * V[m - 1, m - 1] - y0 * W[m - 1, m - 1])
        W[m, m] = (2 * m - 1) * (x0 * W[m - 1, m - 1] + y0 * V[m - 1, m - 1])
        if m <= n_max:
            V[m + 1, m] = (2 * m + 1) * z0 * V[m, m]
            W[m + 1, m] = (2 * m + 1) * z0 * W[m, m]
        for n in range(m + 2, n_max + 2):
            V[n, m] = ((2 * n - 1) * z0 * V[n - 1, m] - (n + m - 1) * rho * V[n - 2, m]) / (n - m)
            W[n, m] = ((2 * n - 1) * z0 * W[n - 1, m] - (n + m - 1) * rho * W[n - 2, m]) / (n - m)
    ax = ay = az = 0.0
    for n in range(n_max + 1):
        ax -= C[n, 0] * V[n + 1, 1]
        ay -= C[n, 0] * W[n + 1, 1]
        az -= (n + 1) * C[n, 0] * V[n + 1, 0]
        for m in range(1, n + 1):
            f = 0.5 * (n - m + 1) * (n - m + 2)
            ax += (0.5 * (-C[n, m] * V[n + 1, m + 1] - S[n, m] * W[n + 1, m + 1])
                   + f * (C[n, m] * V[n + 1, m - 1] + S[n, m] * W[n + 1, m - 1]))
            ay += (0.5 * (-C[n, m] * W[n + 1, m + 1] + S[n, m] * V[n + 1, m + 1])
                   + f * (-C[n, m] * W[n + 1, m - 1] + S[n, m] * V[n + 1, m - 1]))
            az += (n - m + 1) * (-C[n, m] * V[n + 1, m] - S[n, m] * W[n + 1, m])
    return EGM96_GM / power(EGM96_RADIUS, 2.0) * np.array([ax, ay, az])


@njit(cache=True)
def sun_moon_positions(gps_time):
    """leo_orbits.sun_moon_positions."""
    days = (gps_time - GPS_UTC_LEAP_SECONDS) / 86400.0
    whole = np.floor(days)
    whole, fraction = GPS_EPOCH_JULIAN_DATE + whole, days - whole
    T = ((whole - 2451545.0) + fraction) / 36525.0
    d = np.deg2rad
    obliquity = d(23.439291 - 0.0130042 * T)
    M = d(357.5291092 + 35999.05034 * T)
    longitude = d(280.460 + 36000.771 * T + 1.914666471 * np.sin(M) + 0.019994643 * np.sin(2 * M))
    distance = (1.000140612 - 0.016708617 * np.cos(M) - 0.000139589 * np.cos(2 * M)) * ASTRONOMICAL_UNIT
    sun = distance * np.array([np.cos(longitude), np.cos(obliquity) * np.sin(longitude),
                               np.sin(obliquity) * np.sin(longitude)])
    longitude = d(218.32 + 481267.8813 * T + 6.29 * np.sin(d(134.9 + 477198.85 * T))
                  - 1.27 * np.sin(d(259.2 - 413335.38 * T)) + 0.66 * np.sin(d(235.7 + 890534.23 * T))
                  + 0.21 * np.sin(d(269.9 + 954397.70 * T)) - 0.19 * np.sin(d(357.5 + 35999.05 * T))
                  - 0.11 * np.sin(d(186.6 + 966404.05 * T)))
    latitude = d(5.13 * np.sin(d(93.3 + 483202.03 * T)) + 0.28 * np.sin(d(228.2 + 960400.87 * T))
                 - 0.28 * np.sin(d(318.3 + 6003.18 * T)) - 0.17 * np.sin(d(217.6 - 407332.20 * T)))
    parallax = d(0.9508 + 0.0518 * np.cos(d(134.9 + 477198.85 * T)) + 0.0095 * np.cos(d(259.2 - 413335.38 * T))
                 + 0.0078 * np.cos(d(235.7 + 890534.23 * T)) + 0.0028 * np.cos(d(269.9 + 954397.70 * T)))
    distance = EGM96_RADIUS / np.sin(parallax)
    moon = distance * np.array([
        np.cos(latitude) * np.cos(longitude),
        np.cos(obliquity) * np.cos(latitude) * np.sin(longitude) - np.sin(obliquity) * np.sin(latitude),
        np.sin(obliquity) * np.cos(latitude) * np.sin(longitude) + np.cos(obliquity) * np.sin(latitude)])
    return sun, moon


@njit(cache=True)
def third_body_acceleration(r, body, gm):
    return gm * ((body - r) / power(norm(body - r), 3.0) - body / power(norm(body), 3.0))


@njit(cache=True)
def leo_equations_of_motion(gps_time, state, C, S):
    """leo_orbits.equations_of_motion: d[r, v]/dt in TEME, gravity coefficients C, S."""
    r = state[:3]
    angle = earth_rotation_angle(gps_time)
    c, s = np.cos(angle), np.sin(angle)
    to_ecef = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])
    sun, moon = sun_moon_positions(gps_time)
    acceleration = (to_ecef.T @ gravity_acceleration(to_ecef @ r, C, S)
                    + third_body_acceleration(r, sun, SUN_GM) + third_body_acceleration(r, moon, MOON_GM))
    return np.concatenate((state[3:], acceleration))
