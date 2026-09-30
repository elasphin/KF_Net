"""Constants, rotations, frames, gravity and the GNSS error models of Eq. (3)-(4).

Array functions accept one value or an array of values (vectorized with numpy).
"""
import math
from datetime import datetime, timezone

import numpy as np

import settings as cfg

SPEED_OF_LIGHT = 299792458.0
EARTH_ROTATION_RATE = 7.292115e-5             # rad/s
EARTH_GM = 3.986004418e14                     # m^3/s^2
EARTH_J2 = 1.08262668e-3
WGS84_A = 6378137.0
WGS84_E2 = 6.69437999014e-3
GPS_WEEK_SECONDS = 604800.0
GPS_UTC_LEAP_SECONDS = 18.0                   # valid for the dataset years (2022, 2023)
GPS_EPOCH = datetime(1980, 1, 6, tzinfo=timezone.utc)
GPS_L1_FREQUENCY = 1575.42e6
BDS_B1I_FREQUENCY = 1561.098e6
GPS_CA_CHIP_RATE = 1.023e6
BDS_B1I_CHIP_RATE = 2.046e6
EARTH_ROTATION_VECTOR = np.array([0.0, 0.0, EARTH_ROTATION_RATE])


# --- Time --------------------------------------------------------------------
def gps_seconds(year, month, day, hour, minute, second, time_system='GPS'):
    """Calendar date -> seconds since the GPS epoch (GPST)."""
    whole = int(math.floor(second))
    seconds = (datetime(year, month, day, hour, minute, whole, tzinfo=timezone.utc) - GPS_EPOCH).total_seconds()
    seconds += second - whole
    if time_system.upper() in ('BDT', 'BDS'):
        return seconds + 14.0
    if time_system.upper() in ('UTC', 'GLO'):
        return seconds + GPS_UTC_LEAP_SECONDS
    return seconds


def earth_rotation_angle(gps_time):
    """Greenwich mean sidereal angle [rad] (used to place the simulated LEO orbits in ECEF)."""
    julian_date_utc = 2444244.5 + (np.asarray(gps_time) - GPS_UTC_LEAP_SECONDS) / 86400.0
    degrees = 280.46061837 + 360.98564736629 * (julian_date_utc - 2451545.0)
    return np.deg2rad(degrees % 360.0)


# --- Rotations ---------------------------------------------------------------
def skew(v):
    x, y, z = v
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def rotation_vector_to_matrix(v):
    """Exponential map of SO(3)."""
    v = np.asarray(v, dtype=float)
    angle = np.linalg.norm(v)
    K = skew(v)
    if angle < 1e-8:
        return np.eye(3) + K + 0.5 * K @ K
    return np.eye(3) + math.sin(angle) / angle * K + (1.0 - math.cos(angle)) / angle ** 2 * K @ K


def rotation_matrix_to_vector(R):
    """Logarithm map of SO(3)."""
    vee = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    sin_angle = 0.5 * np.linalg.norm(vee)
    angle = math.atan2(sin_angle, 0.5 * (np.trace(R) - 1.0))
    return (0.5 if sin_angle < 1e-9 else angle / (2.0 * sin_angle)) * vee


# --- Frames ------------------------------------------------------------------
def ecef_to_llh(position):
    """ECEF [..., 3] -> latitude [rad], longitude [rad], height [m] (WGS84)."""
    p = np.asarray(position, dtype=float)
    x, y, z = p[..., 0], p[..., 1], p[..., 2]
    longitude = np.arctan2(y, x)
    horizontal = np.hypot(x, y)
    latitude = np.arctan2(z, horizontal * (1.0 - WGS84_E2))
    for _ in range(6):
        N = WGS84_A / np.sqrt(1.0 - WGS84_E2 * np.sin(latitude) ** 2)
        height = horizontal / np.cos(latitude) - N
        latitude = np.arctan2(z, horizontal * (1.0 - WGS84_E2 * N / (N + height)))
    N = WGS84_A / np.sqrt(1.0 - WGS84_E2 * np.sin(latitude) ** 2)
    return latitude, longitude, horizontal / np.cos(latitude) - N


def ecef_to_ned_matrix(latitude, longitude):
    """Rotation matrix C_ned<-ecef."""
    sa, ca, so, co = math.sin(latitude), math.cos(latitude), math.sin(longitude), math.cos(longitude)
    return np.array([[-sa * co, -sa * so, ca], [-so, co, 0.0], [-ca * co, -ca * so, -sa]])


def elevation_azimuth(receiver, satellites):
    """Elevation and azimuth [rad] of satellites [n, 3] seen from the receiver [3] (ECEF)."""
    latitude, longitude, _ = ecef_to_llh(receiver)
    ned = (np.asarray(satellites) - receiver) @ ecef_to_ned_matrix(latitude, longitude).T
    ned /= np.linalg.norm(ned, axis=1, keepdims=True)
    return np.arcsin(-ned[:, 2]), np.arctan2(ned[:, 1], ned[:, 0]) % (2.0 * np.pi)


def gravity(position):
    """Gravitational acceleration with J2 [m/s^2] in ECEF (centrifugal term excluded)."""
    x, y, z = position
    r = np.linalg.norm(position)
    j2 = 1.5 * EARTH_J2 * (WGS84_A / r) ** 2
    ratio = 5.0 * z * z / (r * r)
    return -EARTH_GM / r ** 3 * np.array([x * (1.0 - j2 * (ratio - 1.0)),
                                           y * (1.0 - j2 * (ratio - 1.0)),
                                           z * (1.0 - j2 * (ratio - 3.0))])


# --- Atmospheric delays ------------------------------------------------------
def klobuchar_delay(gps_time, latitude, longitude, elevation, azimuth, alpha, beta):
    """Klobuchar ionospheric delay on GPS L1 [m] (GPS broadcast coefficients)."""
    e = elevation / np.pi
    psi = 0.0137 / (e + 0.11) - 0.022
    phi = np.clip(latitude / np.pi + psi * np.cos(azimuth), -0.416, 0.416)
    lam = longitude / np.pi + psi * np.sin(azimuth) / np.cos(phi * np.pi)
    phi_m = phi + 0.064 * np.cos((lam - 1.617) * np.pi)
    local_time = (43200.0 * lam + gps_time % GPS_WEEK_SECONDS) % 86400.0
    amplitude = np.maximum(0.0, np.polyval(alpha[::-1], phi_m))
    period = np.maximum(72000.0, np.polyval(beta[::-1], phi_m))
    x = 2.0 * np.pi * (local_time - 50400.0) / period
    slant = 1.0 + 16.0 * (0.53 - e) ** 3
    day = np.where(np.abs(x) < 1.57, amplitude * (1.0 - x ** 2 / 2.0 + x ** 4 / 24.0), 0.0)
    return SPEED_OF_LIGHT * slant * (5e-9 + day)


def saastamoinen_delay(height, elevation):
    """Saastamoinen tropospheric delay [m] with the standard atmosphere (RTKLIB form)."""
    h = np.clip(height, -100.0, 1e4)
    temperature = 15.0 - 0.0065 * h + 273.15
    pressure = 1013.25 * (1.0 - 2.2557e-5 * h) ** 5.2568
    humidity = 6.108 * 0.7 * np.exp((17.15 * temperature - 4684.0) / (temperature - 38.45))
    zenith = np.pi / 2.0 - elevation
    return 0.002277 / np.cos(zenith) * (pressure + (1255.0 / temperature + 0.05) * humidity
                                        - 1.16 * np.tan(zenith) ** 2)


# --- Measurement variance, paper Eq. (3) --------------------------------------
def iono_error_std(elevation, latitude):
    """[ref 35] Eq. (15)-(16)."""
    abs_lat = abs(np.rad2deg(latitude))
    vertical = next(std for limit, std in cfg.IONO_VERTICAL_STD if abs_lat <= limit)
    ratio = cfg.EARTH_MEAN_RADIUS * np.cos(elevation) / (cfg.EARTH_MEAN_RADIUS + cfg.IONO_MEAN_HEIGHT)
    return vertical / np.sqrt(1.0 - ratio ** 2)


def tropo_error_std(elevation):
    """[ref 35] Eq. (17)."""
    return 1.001 / np.sqrt(0.002001 + np.sin(elevation) ** 2) * cfg.TROPO_ZENITH_STD


def multipath_variance(elevation, cn0):
    """MP/NLOS variance, paper Eq. (4) (goGPS model, [ref 37])."""
    A, a, s0, s1 = cfg.GOGPS_A, cfg.GOGPS_a, cfg.GOGPS_S0, cfg.GOGPS_S1
    weak = 1.0 / np.sin(elevation) ** 2 * (10.0 ** (-(cn0 - s1) / a)
                                           * ((A / 10.0 ** (-(s0 - s1) / a) - 1.0) * (cn0 - s1) / (s0 - s1) + 1.0))
    return np.where(cn0 < s1, weak, 1.0)


def receiver_noise_std(cn0, chip_rate):
    """Code tracking noise [m], [ref 35] Eq. (19)-(20)."""
    cn0_linear = 10.0 ** (cn0 / 10.0)
    d = cfg.DLL_CORRELATOR_SPACING
    chips = np.sqrt(cfg.DLL_BANDWIDTH * d / (2.0 * cn0_linear)
                    * (1.0 + 2.0 / ((2.0 - d) * cn0_linear * cfg.DLL_INTEGRATION_TIME)))
    return SPEED_OF_LIGHT / chip_rate * chips


def pseudorange_variance(elevation, latitude, cn0, iono_scale, chip_rate):
    """Paper Eq. (3): URA + ionosphere + troposphere + MP/NLOS + receiver noise [m^2]."""
    return (cfg.URA_STD ** 2 + (iono_scale * iono_error_std(elevation, latitude)) ** 2
            + tropo_error_std(elevation) ** 2 + multipath_variance(elevation, cn0)
            + receiver_noise_std(cn0, chip_rate) ** 2)
