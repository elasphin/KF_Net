"""Pseudorange measurements of GPS, BDS-3 and LEO satellites.

Pseudorange model (paper Eq. (1)):
    rho~ = rho + c dt_u - c dt^i + I + T + eps
The satellite clock dt^i, ionosphere I and troposphere T are corrected with
the models below; the receiver clock dt_u is handled by ins_filter.py.
"""
from dataclasses import dataclass

import numpy as np

import settings as cfg
from earth_models import (BDS_B1I_CHIP_RATE, BDS_B1I_FREQUENCY, EARTH_ROTATION_RATE, GPS_CA_CHIP_RATE,
                          GPS_L1_FREQUENCY, SPEED_OF_LIGHT, ecef_to_llh, elevation_azimuth, klobuchar_delay,
                          pseudorange_variance, saastamoinen_delay)

SYSTEM_ORDER = ('G', 'C', 'L')                      # GPS, BDS-3, LEO
FREQUENCY = {'G': GPS_L1_FREQUENCY, 'C': BDS_B1I_FREQUENCY, 'L': GPS_L1_FREQUENCY}
CHIP_RATE = {'G': GPS_CA_CHIP_RATE, 'C': BDS_B1I_CHIP_RATE, 'L': cfg.LEO_CODE_CHIP_RATE}
SP3_INTERPOLATION_POINTS = 10


@dataclass
class EpochMeasurements:
    sat_ids: np.ndarray                 # 'G05', 'C23', 'L017', ...
    systems: np.ndarray                 # 'G', 'C' or 'L'
    pseudoranges: np.ndarray            # [m]
    satellite_positions: np.ndarray     # ECEF at transmit time [m], shape (n, 3)
    satellite_clocks: np.ndarray        # satellite clock offset [s]
    cn0: np.ndarray                     # C/N0 [dB-Hz]

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
def predict_pseudoranges(antenna, meas: EpochMeasurements, time, alpha, beta):
    """Eq. (1) without the receiver clock at the given antenna position.

    Returns predicted pseudorange, unit line of sight (satellite -> antenna),
    elevation [rad] and the Eq. (3) variance [m^2].
    """
    sat = meas.satellite_positions
    difference = antenna - sat
    distance = np.linalg.norm(difference, axis=1)
    sagnac = EARTH_ROTATION_RATE * (sat[:, 0] * antenna[1] - sat[:, 1] * antenna[0]) / SPEED_OF_LIGHT
    elevation, azimuth = elevation_azimuth(antenna, sat)
    latitude, longitude, height = ecef_to_llh(antenna)

    frequency = np.array([FREQUENCY[s] for s in meas.systems])
    chip_rate = np.array([CHIP_RATE[s] for s in meas.systems])
    iono_scale = (GPS_L1_FREQUENCY / frequency) ** 2
    is_leo = meas.systems == 'L'
    if np.any(is_leo):                                  # paper Eq. (2): LEO inside the ionosphere
        satellite_height = ecef_to_llh(sat[is_leo])[2]
        iono_scale[is_leo] *= np.clip((satellite_height - cfg.IONO_LOWER_HEIGHT)
                                      / (cfg.IONO_UPPER_HEIGHT - cfg.IONO_LOWER_HEIGHT), 0.0, 1.0)
    with np.errstate(invalid='ignore', divide='ignore'):
        iono = iono_scale * klobuchar_delay(time, latitude, longitude, elevation, azimuth, alpha, beta)
        tropo = saastamoinen_delay(height, elevation)
        variance = pseudorange_variance(elevation, latitude, meas.cn0, iono_scale, chip_rate)
    predicted = distance + sagnac - SPEED_OF_LIGHT * meas.satellite_clocks + iono + tropo
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
            EpochMeasurements(sat_ids[i], systems, pseudoranges[i], positions[i], clocks[i], cn0[i])))
    return epochs
