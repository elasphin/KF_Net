"""LEO signals of opportunity: orbits from TLE files (docs/ASSUMPTIONS.md A1) and the pseudorange
simulation (paper Sec. II-A, Sec. III-A).

Orbits. Per dataset and satellite two TLEs are chosen:
  - predictor TLE: the latest one published at least LEO_TLE_MIN_AGE before
    the dataset starts (what a receiver has when it starts);
  - reference TLE: the newer TLE closest to the middle of the dataset (best
    available estimate of the true orbit).
Reference orbit ("truth", from which the measurements are made): numerical
integration (DOP853) of the SGP4 state of the reference TLE at its epoch, with
EGM96 20x20 gravity and Sun/Moon point masses; this replaces STK HPOP, as in
Kassas & Saroufim, IEEE AESM (HPOP truth, SGP4 in the filter). Drag and solar
radiation pressure are left out (< 5 m in 12 h above 750 km); satellites
with perigee below LEO_MIN_PERIGEE_ALTITUDE are therefore not used.
Filter orbit (A26, settings LEO_TRAIN_ORBIT and LEO_TEST_ORBITS): 'reference'
(the reference orbit itself, i.e. the true orbit), 'tle' (SGP4 of the
predictor TLE) or 'network' (network_orbit(), a neural-network orbit predictor
to be written). All filter orbits share the same satellites and the same
simulated pseudoranges, so their results differ only by the orbit.
Orbits are integrated in TEME and rotated to ECEF with the Greenwich mean
sidereal angle (polar motion ignored, the same rotation for both orbits).

Measurements:
- Orbits (A1, above): the measurement is made with the reference orbit
  (numerical, from the newest TLE); the filter gets its filter orbit at
  t - rho/c.
- Transmit time: iteration of Eq. (5) until the satellite position changes by
  less than epsilon_T.
- Pseudorange: geometric range (with the Sagnac term) + noise; satellite and
  receiver clocks (A4) and ionosphere/troposphere (A24) are ideal, i.e. known
  and removed, so they are in neither the measurement nor the prediction.
- Noise, one independent draw per satellite and epoch (A3): receiver noise,
  Gaussian with the standard deviation of Eq. (3) (formula of Ref. [35]);
  MP/NLOS: Student-t with LEO_NOISE_DOF degrees of freedom (heavy-tailed,
  non-Gaussian, paper Sec. III-A), scaled so that its standard deviation is
  that of Eq. (4) (Ref. [37]). The noise variance is therefore the LEO
  variance of R. The orbit error comes from the two orbits, so URA is not added.
- C/N0: mean C/N0 of the real GPS/BDS-3 observations in the same elevation bin
  (paper Fig. 5; Ref. [35] gives no elevation-C/N0 formula).
- Orbit error variance of the filter (A25): mean square range error of the
  training filter orbit on the training part of the training dataset, saved in
  leo_orbit_error_train.json (zero for the reference orbit), used in R of every
  test orbit.
The *_numba functions are Numba-compiled copies of the Python function that follows each of them
(setting LEO_FORCE_MODEL = 'numba', see measurements.py).
"""
import json
from math import factorial
from pathlib import Path

import numpy as np
from numba import njit
from scipy.integrate import solve_ivp
from sgp4.api import Satrec

import settings as cfg
from measurements import (EARTH_ROTATION_VECTOR, GPS_UTC_LEAP_SECONDS, SPEED_OF_LIGHT, EpochMeasurements,
                          earth_rotation_angle, earth_rotation_angle_numba, elevation_azimuth, geometric_range,
                          multipath_variance, norm, power, receiver_noise_std)
GPS_EPOCH_JULIAN_DATE = 2444244.5               # 1980-01-06 0h
GRAVITY_FILE = Path(__file__).resolve().parent / 'egm96_degree20.txt'
EGM96_GM, EGM96_RADIUS = 3.986004415e14, 6378136.3
SUN_GM, MOON_GM = 1.32712440018e20, 4.902800066e12
ASTRONOMICAL_UNIT = 1.495978707e11
INTEGRATION_TOLERANCE = {'rtol': 1e-10, 'atol': 1e-4}   # relative, absolute [m, m/s]


# --- Time and frames ---------------------------------------------------------
def tle_epoch(tle):
    """TLE epoch in GPST seconds."""
    days = (tle.jdsatepoch - GPS_EPOCH_JULIAN_DATE) + tle.jdsatepochF
    return days * 86400.0 + GPS_UTC_LEAP_SECONDS


def julian_date(gps_time):
    """(whole, fraction) Julian date (UTC) of GPST seconds, split to keep sub-microsecond precision."""
    days = (np.asarray(gps_time, dtype=float) - GPS_UTC_LEAP_SECONDS) / 86400.0
    whole = np.floor(days)
    return GPS_EPOCH_JULIAN_DATE + whole, days - whole


def teme_to_ecef(gps_time, position, velocity):
    angle = earth_rotation_angle(gps_time)
    c, s = np.cos(angle), np.sin(angle)

    def rotate(v):
        return np.stack((c * v[..., 0] + s * v[..., 1], -s * v[..., 0] + c * v[..., 1], v[..., 2]), axis=-1)

    ecef_position = rotate(position)
    return ecef_position, rotate(velocity) - np.cross(EARTH_ROTATION_VECTOR, ecef_position)


# --- Force model of the reference orbit --------------------------------------
def load_gravity_field(degree):
    """Unnormalized C, S of EGM96 up to the given degree and order (C00 = 1)."""
    C, S = np.zeros((degree + 1, degree + 1)), np.zeros((degree + 1, degree + 1))
    C[0, 0] = 1.0
    for n, m, c, s in np.loadtxt(GRAVITY_FILE):
        n, m = int(n), int(m)
        if n <= degree:
            scale = np.sqrt((2 - (m == 0)) * (2 * n + 1) * factorial(n - m) / factorial(n + m))
            C[n, m], S[n, m] = c * scale, s * scale
    return C, S


GRAVITY_C, GRAVITY_S = load_gravity_field(cfg.LEO_GRAVITY_DEGREE)


@njit(cache=True)
def gravity_acceleration_numba(r, C, S):
    """gravity_acceleration with the coefficients C, S (GRAVITY_C, GRAVITY_S)."""
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


def gravity_acceleration(r):
    """Spherical harmonic gravity [m/s^2] at ECEF position r (Montenbruck & Gill, Satellite Orbits, Sec. 3.2)."""
    C, S, n_max = GRAVITY_C, GRAVITY_S, GRAVITY_C.shape[0] - 1
    r2 = r @ r
    rho = EGM96_RADIUS ** 2 / r2
    x0, y0, z0 = EGM96_RADIUS * r / r2
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
    return EGM96_GM / EGM96_RADIUS ** 2 * np.array([ax, ay, az])


@njit(cache=True)
def sun_moon_positions_numba(gps_time):
    """sun_moon_positions."""
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


def sun_moon_positions(gps_time):
    """Low-precision Sun and Moon positions [m] (Vallado, Fundamentals of Astrodynamics, Algorithms 29 and 31)."""
    whole, fraction = julian_date(gps_time)
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
def third_body_acceleration_numba(r, body, gm):
    """third_body_acceleration."""
    return gm * ((body - r) / power(norm(body - r), 3.0) - body / power(norm(body), 3.0))


def third_body_acceleration(r, body, gm):
    return gm * ((body - r) / np.linalg.norm(body - r) ** 3 - body / np.linalg.norm(body) ** 3)


@njit(cache=True)
def equations_of_motion_numba(gps_time, state, C, S):
    """equations_of_motion with the gravity coefficients C, S."""
    r = state[:3]
    angle = earth_rotation_angle_numba(gps_time)
    c, s = np.cos(angle), np.sin(angle)
    to_ecef = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])
    sun, moon = sun_moon_positions_numba(gps_time)
    acceleration = (to_ecef.T @ gravity_acceleration_numba(to_ecef @ r, C, S)
                    + third_body_acceleration_numba(r, sun, SUN_GM) + third_body_acceleration_numba(r, moon, MOON_GM))
    return np.concatenate((state[3:], acceleration))


def equations_of_motion(gps_time, state):
    """d[r, v]/dt in TEME (treated as inertial over the dataset)."""
    r = state[:3]
    angle = earth_rotation_angle(gps_time)
    c, s = np.cos(angle), np.sin(angle)
    to_ecef = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])
    sun, moon = sun_moon_positions(gps_time)
    acceleration = (to_ecef.T @ gravity_acceleration(to_ecef @ r)
                    + third_body_acceleration(r, sun, SUN_GM) + third_body_acceleration(r, moon, MOON_GM))
    return np.concatenate((state[3:], acceleration))


# --- Orbits ------------------------------------------------------------------
def sgp4_orbit(tle, times):
    """SGP4 ECEF position and velocity [m, m/s] at GPST times (NaN where SGP4 fails)."""
    error, r, v = tle.sgp4_array(*julian_date(times))
    position, velocity = teme_to_ecef(times, r * 1e3, v * 1e3)
    position[error != 0] = np.nan
    return position, velocity


def force_model():
    """(equations of motion, extra arguments) of settings.LEO_FORCE_MODEL: the Python function or its compiled copy."""
    if cfg.LEO_FORCE_MODEL == 'python':
        return equations_of_motion, None
    if cfg.LEO_FORCE_MODEL == 'numba':
        return equations_of_motion_numba, (GRAVITY_C, GRAVITY_S)
    raise ValueError(f"LEO_FORCE_MODEL must be 'python' or 'numba', not {cfg.LEO_FORCE_MODEL!r}")


def reference_orbit(tle, times):
    """Numerical ECEF position and velocity at GPST times (ascending), from the SGP4 state at the TLE epoch."""
    epoch = tle_epoch(tle)
    _, r, v = tle.sgp4(tle.jdsatepoch, tle.jdsatepochF)
    states = np.empty((len(times), 6))
    equations, arguments = force_model()
    for t_eval in (times[times < epoch][::-1], times[times >= epoch]):   # backward and forward from the epoch
        if len(t_eval):
            solution = solve_ivp(equations, (epoch, t_eval[-1]), np.array(r + v) * 1e3, method='DOP853',
                                 t_eval=t_eval, args=arguments, **INTEGRATION_TOLERANCE)
            states[np.searchsorted(times, t_eval)] = solution.y.T
    return teme_to_ecef(times, states[:, :3], states[:, 3:])


def network_orbit(history, times):
    """Orbit predicted by a neural network (A26), to be written.

    history: TLEs of the satellite (sgp4 Satrec, ascending epoch) published up to the
    predictor TLE (history[-1]), never the reference TLE. Returns ECEF position and
    velocity [m, m/s] at the GPST times, shape (len(times), 3) each. If the code is put
    in another file, add that file to dataset.DATA_CODE so the cache is renewed.
    """
    raise NotImplementedError("LEO orbit 'network' is not written yet: leo_pseudorange.network_orbit")


def filter_orbit(name, history, reference, times):
    """ECEF position and velocity of the filter orbit 'name' (A26) at the GPST times."""
    if name == 'reference':
        return reference
    if name == 'tle':
        return sgp4_orbit(history[-1], times)
    if name == 'network':
        return network_orbit(history, times)
    raise ValueError(f"LEO orbit must be 'reference', 'tle' or 'network', not {name!r}")


# --- TLE files and satellite selection ---------------------------------------
def read_tle_files(folder):
    """{satellite number: [TLE sorted by epoch]} from all *.txt files (2- or 3-line format), duplicates removed."""
    satellites = {}
    for path in sorted(Path(folder).rglob('*.txt')):
        lines = path.read_text(errors='replace').splitlines()
        for line1, line2 in zip(lines, lines[1:]):
            if line1.startswith('1 ') and line2.startswith('2 '):
                tle = Satrec.twoline2rv(line1.rstrip(), line2.rstrip())
                satellites.setdefault(tle.satnum, {})[tle_epoch(tle)] = tle
    return {number: [tles[epoch] for epoch in sorted(tles)] for number, tles in satellites.items()}


def find_tle_folder():
    folders = sorted(p for p in cfg.DATASET_FOLDER.rglob(cfg.LEO_TLE_FOLDER_NAME) if p.is_dir())
    if not folders:
        raise FileNotFoundError(f'Folder {cfg.LEO_TLE_FOLDER_NAME} (LEO TLE files) not found in {cfg.DATASET_FOLDER}')
    return folders[0]


def above_mask(positions, receivers, margin_deg=2.0):
    """True if the satellite (ECEF, one per epoch) is ever above the elevation mask - margin (geocentric up)."""
    line_of_sight = positions - receivers
    sin_elevation = (np.sum(line_of_sight * receivers, axis=1)
                     / (np.linalg.norm(line_of_sight, axis=1) * np.linalg.norm(receivers, axis=1)))
    return np.nanmax(sin_elevation) >= np.sin(np.deg2rad(cfg.LEO_ELEVATION_MASK_DEG - margin_deg))


def leo_orbits(times, receivers, orbit_names):
    """[(satellite id, reference (position, velocity), {filter orbit name: (position, velocity)})] at the fusion times.

    The satellites are chosen the same way for every filter orbit (they need a predictor TLE), so
    every filter orbit sees the same satellites.
    """
    start, middle = times[0], 0.5 * (times[0] + times[-1])
    orbits = []
    for number, tles in read_tle_files(find_tle_folder()).items():
        epochs = np.array([tle_epoch(tle) for tle in tles])
        older = np.flatnonzero(epochs <= start - cfg.LEO_TLE_MIN_AGE * 3600.0)
        if len(older) == 0 or older[-1] == len(tles) - 1:
            continue
        newer = np.arange(older[-1] + 1, len(tles))
        predictor_tle, reference_tle = tles[older[-1]], tles[newer[np.argmin(np.abs(epochs[newer] - middle))]]
        if reference_tle.altp * reference_tle.radiusearthkm * 1e3 < cfg.LEO_MIN_PERIGEE_ALTITUDE:
            continue

        reference_sgp4 = sgp4_orbit(reference_tle, times)[0]
        if np.isnan(reference_sgp4).any() or not above_mask(reference_sgp4, receivers):
            continue
        at_reference_epoch = np.array([tle_epoch(reference_tle)])
        mismatch = np.linalg.norm(sgp4_orbit(predictor_tle, at_reference_epoch)[0]
                                  - sgp4_orbit(reference_tle, at_reference_epoch)[0])
        if not mismatch <= cfg.LEO_MAX_TLE_MISMATCH:          # manoeuvre between the two TLEs (or SGP4 failure)
            continue
        if np.isnan(sgp4_orbit(predictor_tle, times)[0]).any():
            continue
        reference = reference_orbit(reference_tle, times)
        history = tles[:older[-1] + 1]
        filter_orbits = {name: filter_orbit(name, history, reference, times) for name in orbit_names}
        for name, (position, velocity) in filter_orbits.items():
            if not (np.isfinite(position).all() and np.isfinite(velocity).all()):
                raise ValueError(f'LEO orbit {name!r} of L{number:05d} is not finite over the dataset')
        orbits.append((f'L{number:05d}', reference, filter_orbits))
    return orbits



# --- C/N0 of the LEO satellites from real data -------------------------------
def elevation_bin(elevation, bin_count):
    return np.minimum((np.rad2deg(elevation) // cfg.LEO_ELEVATION_BIN_DEG).astype(int), bin_count - 1)


def mean_cn0_bins(data, gnss_epochs, epochs):
    """Mean C/N0 of the real GPS/BDS-3 observations of the given epochs per elevation bin (A3).

    Elevation at the truth antenna position; only the rows the filter uses (GNSS_ELEVATION_MASK_DEG, A10).
    An empty bin takes the value of the nearest filled one. epochs: the whole dataset (dataset).
    """
    bin_count = int(np.ceil(90.0 / cfg.LEO_ELEVATION_BIN_DEG))
    cn0 = [[] for _ in range(bin_count)]
    for k in epochs:
        meas = gnss_epochs[k]
        elevation, _ = elevation_azimuth(data.truth_antenna_position[k], meas.satellite_positions)
        rows = np.flatnonzero(elevation >= np.deg2rad(cfg.GNSS_ELEVATION_MASK_DEG))
        for b, c in zip(elevation_bin(elevation[rows], bin_count), meas.cn0[rows]):
            cn0[b].append(c)
    filled = [b for b in range(bin_count) if cn0[b]]
    return np.array([np.mean(cn0[min(filled, key=lambda f: abs(f - b))]) for b in range(bin_count)])


# --- Measurement simulation --------------------------------------------------
def transmit_positions(receiver, position, velocity):
    """Paper Eq. (5): satellite positions at T1 = T2 - dT, iterated until they change < epsilon_T.

    The orbit is linear over the transit time (< 15 ms): error < 1 mm.
    """
    transit = np.zeros(len(position))
    for _ in range(20):
        new_transit = geometric_range(receiver, position - transit[:, None] * velocity) / SPEED_OF_LIGHT
        change = np.max(np.abs(new_transit - transit) * np.linalg.norm(velocity, axis=1), initial=0.0)
        transit = new_transit
        if change < cfg.LIGHT_TIME_THRESHOLD:
            break
    return position - transit[:, None] * velocity


def simulate_leo_measurements(data, cn0_bins, seed, orbit_names):
    """Simulated LEO pseudoranges at every fusion epoch, at the truth antenna position.

    Noise (A3): receiver noise ~ Gaussian with the std of Eq. (3); MP/NLOS ~ Student-t with LEO_NOISE_DOF
    degrees of freedom, scaled to the std of Eq. (4). The noise variance is then that of R.

    The pseudoranges (and their noise draws) are made once with the reference orbit; every
    filter orbit of orbit_names (A26) gets the same rows with its own satellite positions.
    Returns {orbit name: epochs} and {orbit name: [range errors of that orbit (filter minus
    true geometric range at the truth antenna) of the rows of each epoch]}.
    """
    rng = np.random.default_rng(seed)
    orbits = leo_orbits(data.fusion_times, data.truth_antenna_position, orbit_names)
    if not orbits:
        raise RuntimeError('No LEO satellite with usable TLEs over the dataset (see LEO_TLE_MIN_AGE in settings.py)')
    sat_ids = np.array([sat_id for sat_id, _, _ in orbits])
    reference_position, reference_velocity = (np.stack([o[1][i] for o in orbits]) for i in (0, 1))
    filter_position = {name: np.stack([o[2][name][0] for o in orbits]) for name in orbit_names}
    filter_velocity = {name: np.stack([o[2][name][1] for o in orbits]) for name in orbit_names}
    epochs, range_errors = {name: [] for name in orbit_names}, {name: [] for name in orbit_names}
    for k in range(len(data.fusion_times)):
        receiver = data.truth_antenna_position[k]
        positions = transmit_positions(receiver, reference_position[:, k], reference_velocity[:, k])
        elevation, _ = elevation_azimuth(receiver, positions)
        visible = np.flatnonzero(elevation >= np.deg2rad(cfg.LEO_ELEVATION_MASK_DEG))
        cn0 = cn0_bins[elevation_bin(elevation[visible], len(cn0_bins))]
        n = len(visible)
        true_range = geometric_range(receiver, positions[visible])
        nu = cfg.LEO_NOISE_DOF
        noise = (receiver_noise_std(cn0, cfg.LEO_CODE_CHIP_RATE) * rng.standard_normal(n)
                 + np.sqrt(multipath_variance(elevation[visible], cn0) * (nu - 2.0) / nu) * rng.standard_t(nu, n))
        pseudoranges = true_range + noise
        for name in orbit_names:
            # Filter side: the filter orbit at the transmit time t - rho/c (clocks are ideal, A4).
            satellites = (filter_position[name][visible, k]
                          - (pseudoranges / SPEED_OF_LIGHT)[:, None] * filter_velocity[name][visible, k])
            range_errors[name].append(geometric_range(receiver, satellites) - true_range)
            epochs[name].append(EpochMeasurements(sat_ids[visible], np.full(n, 'L'), pseudoranges, satellites,
                                                  np.zeros(n), cn0, np.zeros(n)))
    return epochs, range_errors


def range_error_summary(range_errors):
    return {'rms_m': float(np.sqrt(np.mean(range_errors ** 2))) if len(range_errors) else 0.0,
            'mean_m': float(np.mean(range_errors)) if len(range_errors) else 0.0,
            'samples': int(len(range_errors))}


def orbit_error_variance(range_errors, split):
    """sigma^2 of the orbit error in the filter R (A25), the same for every filter orbit (A26).

    range_errors: {filter orbit name: range errors} of the whole dataset. Writes leo_orbit_error_<split>.json
    (train: RMS, mean and samples of LEO_TRAIN_ORBIT; validation, test: the same for each orbit of the split) and
    returns the mean square range error of LEO_TRAIN_ORBIT on the training dataset (written by train.py; zero for
    the reference orbit).
    """
    cfg.OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    train_file = cfg.OUTPUT_FOLDER / 'leo_orbit_error_train.json'
    if split == 'train':
        summary = {'orbit': cfg.LEO_TRAIN_ORBIT, **range_error_summary(range_errors[cfg.LEO_TRAIN_ORBIT])}
    else:
        summary = {name: range_error_summary(errors) for name, errors in range_errors.items()}
    (cfg.OUTPUT_FOLDER / f'leo_orbit_error_{split}.json').write_text(json.dumps(summary, indent=1))
    if not train_file.exists():
        raise FileNotFoundError(f'{train_file} is written by train.py; run train.py first')
    train = json.loads(train_file.read_text())
    if train.get('orbit') != cfg.LEO_TRAIN_ORBIT:
        raise ValueError(f"{train_file} is for LEO orbit {train.get('orbit')!r}, settings.LEO_TRAIN_ORBIT is "
                         f'{cfg.LEO_TRAIN_ORBIT!r}; run train.py again')
    return train['rms_m'] ** 2
