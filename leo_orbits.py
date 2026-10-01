"""LEO satellite orbits from TLE files (ASSUMPTIONS.md A1).

Per dataset and satellite two TLEs are chosen:
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
Predicted orbit (used by the filter): SGP4 of the predictor TLE. To test
another orbit predictor (e.g. a neural network), replace predicted_orbit(); the
orbit error variance of the filter is re-estimated from it (A25).
Orbits are integrated in TEME and rotated to ECEF with the Greenwich mean
sidereal angle (polar motion ignored, the same rotation for both orbits).
"""
from math import factorial
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from sgp4.api import Satrec

import settings as cfg
from earth_models import EARTH_ROTATION_VECTOR, GPS_UTC_LEAP_SECONDS, earth_rotation_angle

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


def third_body_acceleration(r, body, gm):
    return gm * ((body - r) / np.linalg.norm(body - r) ** 3 - body / np.linalg.norm(body) ** 3)


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


def reference_orbit(tle, times):
    """Numerical ECEF position and velocity at GPST times (ascending), from the SGP4 state at the TLE epoch."""
    epoch = tle_epoch(tle)
    _, r, v = tle.sgp4(tle.jdsatepoch, tle.jdsatepochF)
    states = np.empty((len(times), 6))
    for t_eval in (times[times < epoch][::-1], times[times >= epoch]):   # backward and forward from the epoch
        if len(t_eval):
            solution = solve_ivp(equations_of_motion, (epoch, t_eval[-1]), np.array(r + v) * 1e3,
                                 method='DOP853', t_eval=t_eval, **INTEGRATION_TOLERANCE)
            states[np.searchsorted(times, t_eval)] = solution.y.T
    return teme_to_ecef(times, states[:, :3], states[:, 3:])


def predicted_orbit(tle, times):
    """Orbit used by the filter: SGP4 of the predictor TLE."""
    return sgp4_orbit(tle, times)


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


def leo_orbits(times, receivers):
    """[(satellite id, reference (position, velocity), predicted (position, velocity))] at the fusion times."""
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
        predicted = predicted_orbit(predictor_tle, times)
        if np.isnan(predicted[0]).any():
            continue
        orbits.append((f'L{number:05d}', reference_orbit(reference_tle, times), predicted))
    return orbits
