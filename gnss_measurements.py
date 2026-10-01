"""Pseudorange measurements of GPS, BDS-3 and LEO satellites.

Pseudorange model (paper Eq. (1)):
    rho~ = rho + c dt_u - c dt^i + I + T + eps
GPS/BDS-3: the satellite clock dt^i, ionosphere I and troposphere T are
corrected with the models below. LEO: clocks and atmospheric delays are ideal
(known and removed, A4, A24), so only the geometric range remains. The receiver
clock dt_u is handled by ins_filter.py.
"""
from dataclasses import dataclass

import numpy as np

import settings as cfg
from earth_models import (BDS_B1I_CHIP_RATE, BDS_B1I_FREQUENCY, EARTH_ROTATION_RATE, GPS_CA_CHIP_RATE,
                          GPS_L1_FREQUENCY, SPEED_OF_LIGHT, ecef_to_llh, elevation_azimuth, klobuchar_delay,
                          multipath_variance, pseudorange_variance, receiver_noise_std, saastamoinen_delay)

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
        tropo = np.where(is_leo, 0.0, saastamoinen_delay(height, elevation))
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
