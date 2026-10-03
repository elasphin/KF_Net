"""Constants, rotations, frames, gravity and the GNSS error models of Eq. (3)-(4).

Array functions accept one value or an array of values (vectorized with numpy).

Functions named *_numba are Numba-compiled copies of the Python function that follows them (settings
INS_MECHANIZATION and LEO_FORCE_MODEL = 'numba'). Each repeats the arithmetic of its Python original
operation by operation and in the same order, so both give the same numbers; only the Python interpreter
overhead is removed. They take one value (no vectorization). The first call compiles them (a few seconds);
cache=True keeps the compiled code in __pycache__ for the next runs.

Pseudorange measurements of GPS, BDS-3 and LEO satellites.

Pseudorange model (paper Eq. (1)):
    rho~ = rho + c dt_u - c dt^i + I + T + eps
GPS/BDS-3: the satellite clock dt^i, ionosphere I and troposphere T are
corrected with the models below. LEO: clocks and atmospheric delays are ideal
(known and removed, A4, A24), so only the geometric range remains. The receiver
clock dt_u is handled by navigation.py.
"""
import math
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
from llvmlite import ir
from numba import njit, types
from numba.core import cgutils
from numba.extending import intrinsic

import settings as cfg


# ===== Constants, frames, gravity, error models =======================================================================
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


# --- Numba: the same operations as numpy -------------------------------------
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


# --- Time --------------------------------------------------------------------
def gps_seconds(year, month, day, hour, minute, second, time_system='GPS'):
    """Calendar date -> seconds since the GPS epoch (GPST)."""
    whole = int(math.floor(second))
    seconds = (datetime(year, month, day, hour, minute, whole, tzinfo=timezone.utc) - GPS_EPOCH).total_seconds()
    seconds += second - whole
    system = time_system.upper()
    if system in ('GPS', 'GPST', 'GAL', 'GST', 'QZS', 'QZSST'):
        return seconds
    if system in ('BDT', 'BDS'):
        return seconds + 14.0
    if system in ('UTC', 'GLO'):
        return seconds + GPS_UTC_LEAP_SECONDS
    if system == 'TAI':
        return seconds - 19.0
    raise ValueError(f'unknown time system {time_system!r}')


@njit(cache=True)
def earth_rotation_angle_numba(gps_time):
    """earth_rotation_angle."""
    julian_date_utc = 2444244.5 + (gps_time - GPS_UTC_LEAP_SECONDS) / 86400.0
    degrees = 280.46061837 + 360.98564736629 * (julian_date_utc - 2451545.0)
    return np.deg2rad(degrees % 360.0)


def earth_rotation_angle(gps_time):
    """Greenwich mean sidereal angle [rad] (rotation of the TEME LEO orbits to ECEF, UT1 = UTC)."""
    julian_date_utc = 2444244.5 + (np.asarray(gps_time) - GPS_UTC_LEAP_SECONDS) / 86400.0
    degrees = 280.46061837 + 360.98564736629 * (julian_date_utc - 2451545.0)
    return np.deg2rad(degrees % 360.0)


# --- Rotations ---------------------------------------------------------------
@njit(cache=True)
def skew_numba(v):
    """skew."""
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def skew(v):
    x, y, z = v
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


@njit(cache=True)
def rotation_vector_to_matrix_numba(v):
    """rotation_vector_to_matrix."""
    angle = norm(v)
    K = skew_numba(v)
    if angle < 1e-8:
        return np.eye(3) + K + 0.5 * K @ K
    return np.eye(3) + math.sin(angle) / angle * K + (1.0 - math.cos(angle)) / power(angle, 2.0) * K @ K


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


@njit(cache=True)
def gravity_numba(position):
    """gravity."""
    x, y, z = position[0], position[1], position[2]
    r = norm(position)
    j2 = 1.5 * EARTH_J2 * power(WGS84_A / r, 2.0)
    ratio = 5.0 * z * z / (r * r)
    return -EARTH_GM / power(r, 3.0) * np.array([x * (1.0 - j2 * (ratio - 1.0)),
                                             y * (1.0 - j2 * (ratio - 1.0)),
                                             z * (1.0 - j2 * (ratio - 3.0))])


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


def saastamoinen_delay(latitude, height, elevation):
    """Saastamoinen tropospheric delay [m] with the standard atmosphere, as RTKLIB tropmodel (rtkcmn.c).

    Zero below the horizon or outside -100 m..10 km. The former form with the -1.16 tan^2(z) term is
    valid only well above the horizon (it is negative below ~1.5 deg).
    """
    valid = (elevation > 0.0) & (height >= -100.0) & (height <= 1e4)
    h = np.clip(height, 0.0, 1e4)
    temperature = 15.0 - 6.5e-3 * h + 273.16
    pressure = 1013.25 * (1.0 - 2.2557e-5 * h) ** 5.2568
    humidity = 6.108 * 0.7 * np.exp((17.15 * temperature - 4684.0) / (temperature - 38.45))
    cos_zenith = np.cos(np.pi / 2.0 - np.maximum(elevation, 1e-3))
    hydrostatic = 0.0022768 * pressure / (1.0 - 0.00266 * np.cos(2.0 * latitude) - 0.00028 * h / 1e3) / cos_zenith
    wet = 0.002277 * (1255.0 / temperature + 0.05) * humidity / cos_zenith
    return np.where(valid, hydrostatic + wet, 0.0)


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


# ===== GNSS measurements ==============================================================================================
SYSTEM_ORDER = ('G', 'C', 'L')                      # GPS, BDS-3, LEO
FREQUENCY = {'G': GPS_L1_FREQUENCY, 'C': BDS_B1I_FREQUENCY}
CHIP_RATE = {'G': GPS_CA_CHIP_RATE, 'C': BDS_B1I_CHIP_RATE, 'L': cfg.LEO_CODE_CHIP_RATE}
SP3_INTERPOLATION_POINTS = 10


@dataclass
class EpochMeasurements:
    sat_ids: np.ndarray                 # 'G05', 'C23', 'L43578' (L + NORAD number), ...
    systems: np.ndarray                 # 'G', 'C' or 'L'
    pseudoranges: np.ndarray            # [m]
    satellite_positions: np.ndarray     # ECEF at transmit time [m], shape (n, 3)
    satellite_clocks: np.ndarray        # satellite clock offset [s]
    cn0: np.ndarray                     # C/N0 [dB-Hz]
    orbit_variance: np.ndarray          # LEO: range error variance of the predicted orbit [m^2] (A25); GNSS: 0

    def __len__(self):
        return len(self.sat_ids)

    def subset(self, index):
        return EpochMeasurements(*(getattr(self, f)[index] for f in self.__dataclass_fields__))


def sort_measurements(meas: EpochMeasurements) -> EpochMeasurements:
    """Row order: system (G, C, L), then satellite id."""
    order = sorted(range(len(meas)), key=lambda i: (SYSTEM_ORDER.index(meas.systems[i]), meas.sat_ids[i]))
    return meas.subset(np.array(order, dtype=int))


def merge_measurements(a: EpochMeasurements, b: EpochMeasurements) -> EpochMeasurements:
    return sort_measurements(EpochMeasurements(*(np.concatenate((getattr(a, f), getattr(b, f)))
                                                 for f in a.__dataclass_fields__)))


# --- Predicted pseudorange ---------------------------------------------------
def geometric_range(antenna, satellites):
    """Distance from the satellites (ECEF at transmit time) to the antenna, with the Earth rotation (Sagnac) term."""
    sagnac = EARTH_ROTATION_RATE * (satellites[:, 0] * antenna[1] - satellites[:, 1] * antenna[0]) / SPEED_OF_LIGHT
    return np.linalg.norm(antenna - satellites, axis=1) + sagnac


def predict_pseudoranges(antenna, meas: EpochMeasurements, time, alpha, beta):
    """Eq. (1) without the receiver clock at the given antenna position.

    Returns predicted pseudorange, unit line of sight (satellite -> antenna),
    elevation [rad] and the variance [m^2]: Eq. (3) for GPS/BDS-3; MP/NLOS +
    receiver noise + predicted-orbit error for LEO (A3, A25).
    """
    sat = meas.satellite_positions
    difference = antenna - sat
    distance = np.linalg.norm(difference, axis=1)
    elevation, azimuth = elevation_azimuth(antenna, sat)
    latitude, longitude, height = ecef_to_llh(antenna)

    is_leo = meas.systems == 'L'
    chip_rate = np.array([CHIP_RATE[s] for s in meas.systems])
    iono_scale = np.array([(GPS_L1_FREQUENCY / FREQUENCY.get(s, GPS_L1_FREQUENCY)) ** 2 for s in meas.systems])
    with np.errstate(invalid='ignore', divide='ignore'):
        iono = np.where(is_leo, 0.0, iono_scale * klobuchar_delay(time, latitude, longitude, elevation, azimuth,
                                                                   alpha, beta))
        tropo = np.where(is_leo, 0.0, saastamoinen_delay(latitude, height, elevation))
        leo_variance = (multipath_variance(elevation, meas.cn0) + receiver_noise_std(meas.cn0, chip_rate) ** 2
                        + meas.orbit_variance)
        variance = np.where(is_leo, leo_variance,
                            pseudorange_variance(elevation, latitude, meas.cn0, iono_scale, chip_rate))
    predicted = geometric_range(antenna, sat) - SPEED_OF_LIGHT * meas.satellite_clocks + iono + tropo
    return predicted, difference / distance[:, None], elevation, variance


# --- GNSS satellite positions and clocks from SP3 / CLK -----------------------
def lagrange_interpolate(times, values, query):
    """Lagrange interpolation of values (m, k) at the query times (q,) -> (q, k)."""
    n = SP3_INTERPOLATION_POINTS
    start = np.clip(np.searchsorted(times, query) - n // 2, 0, len(times) - n)
    index = start[:, None] + np.arange(n)
    t = times[index]                                              # (q, n)
    to_query = np.repeat((query[:, None] - t)[:, None, :], n, axis=1)
    between = t[:, :, None] - t[:, None, :]
    diagonal = np.arange(n)
    to_query[:, diagonal, diagonal] = 1.0
    between[:, diagonal, diagonal] = 1.0
    weights = np.prod(to_query, axis=2) / np.prod(between, axis=2)
    return np.einsum('qn,qnk->qk', weights, values[index])


def satellite_position_and_clock(orbits, clocks, sat_id, transmit_times):
    """Position (ECEF at transmit time) and clock offset (with the relativistic term) of one satellite."""
    orbit_times, orbit_positions = orbits[sat_id]
    clock_times, clock_values = clocks[sat_id]
    position = lagrange_interpolate(orbit_times, orbit_positions, transmit_times)
    velocity = (lagrange_interpolate(orbit_times, orbit_positions, transmit_times + 0.5)
                - lagrange_interpolate(orbit_times, orbit_positions, transmit_times - 0.5))
    clock = np.interp(transmit_times, clock_times, clock_values)
    clock -= 2.0 * np.sum(position * velocity, axis=1) / SPEED_OF_LIGHT ** 2
    inside = ((transmit_times >= max(orbit_times[0], clock_times[0]))
              & (transmit_times <= min(orbit_times[-1], clock_times[-1])))
    return position, clock, inside


def prepare_gnss_measurements(data) -> list[EpochMeasurements]:
    """Satellite position and clock at the transmit time of every GPS/BDS-3 observation."""
    rows = [(k, *observation) for k, epoch_obs in enumerate(data.gnss_observations) for observation in epoch_obs]
    epoch, sat_ids, pseudoranges, cn0 = (np.array(column) for column in zip(*rows))
    epoch, pseudoranges, cn0 = epoch.astype(int), pseudoranges.astype(float), cn0.astype(float)
    positions, clocks = np.full((len(rows), 3), np.nan), np.full(len(rows), np.nan)
    for sat in np.unique(sat_ids):
        if sat not in data.orbits or sat not in data.clocks:
            continue
        rows_of_sat = sat_ids == sat
        transmit = data.fusion_times[epoch[rows_of_sat]] - pseudoranges[rows_of_sat] / SPEED_OF_LIGHT
        _, clock, _ = satellite_position_and_clock(data.orbits, data.clocks, sat, transmit)
        position, clock, inside = satellite_position_and_clock(data.orbits, data.clocks, sat, transmit - clock)
        positions[rows_of_sat] = np.where(inside[:, None], position, np.nan)
        clocks[rows_of_sat] = np.where(inside, clock, np.nan)

    valid = np.isfinite(clocks)
    epochs = []
    for k in range(len(data.fusion_times)):
        i = np.flatnonzero((epoch == k) & valid)
        systems = np.array([s[0] for s in sat_ids[i]], dtype='<U1')
        epochs.append(sort_measurements(
            EpochMeasurements(sat_ids[i], systems, pseudoranges[i], positions[i], clocks[i], cn0[i],
                              np.zeros(len(i)))))
    return epochs
