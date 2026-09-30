"""LEO downlink pseudorange simulation (paper Sec. II-A, Sec. III-A).

- Constellation: 144 Walker + 81 polar satellites (Table II) on circular
  two-body orbits (substitute for STK HPOP, see ASSUMPTIONS.md A1).
- Transmit time: iteration of Eq. (5) until the satellite position changes by
  less than epsilon_T.
- Pseudorange: Eq. (1) with ionosphere Eq. (2) and troposphere; satellite and
  receiver clocks are zero (A4).
- MP/NLOS + receiver error eps: drawn from the real GPS/BDS-3 pseudorange
  errors of the same dataset in the same elevation bin (Sec. III-A: "inject
  the statistical characteristics of real data"), see A3.
"""
import numpy as np

import settings as cfg
from earth_models import EARTH_GM, EARTH_ROTATION_RATE, SPEED_OF_LIGHT, WGS84_A, earth_rotation_angle, \
    elevation_azimuth
from gnss_measurements import EpochMeasurements, predict_pseudoranges


# --- Constellation -----------------------------------------------------------
def leo_orbit_elements():
    """(orbit radius, inclination, right ascension of node, argument of latitude at start) per satellite."""
    rows = []
    for total, planes, phasing, inclination, altitude in (
            (cfg.WALKER_SATELLITES, cfg.WALKER_PLANES, cfg.WALKER_PHASING, cfg.WALKER_INCLINATION_DEG,
             cfg.WALKER_ALTITUDE),
            (cfg.POLAR_SATELLITES, cfg.POLAR_PLANES, cfg.POLAR_PHASING, cfg.POLAR_INCLINATION_DEG,
             cfg.POLAR_ALTITUDE)):
        per_plane = total // planes
        for plane in range(planes):
            for slot in range(per_plane):
                rows.append((WGS84_A + altitude, np.deg2rad(inclination), 2.0 * np.pi * plane / planes,
                             2.0 * np.pi * slot / per_plane + 2.0 * np.pi * phasing * plane / total))
    return np.array(rows)


def leo_positions(elements, times, start_time):
    """ECEF positions [n, 3] of all satellites; times: one GPST time per satellite."""
    radius, inclination, node, latitude_argument = elements.T
    u = latitude_argument + np.sqrt(EARTH_GM / radius ** 3) * (times - start_time)
    x_orbit, y_orbit = radius * np.cos(u), radius * np.sin(u)
    x = x_orbit * np.cos(node) - y_orbit * np.cos(inclination) * np.sin(node)
    y = x_orbit * np.sin(node) + y_orbit * np.cos(inclination) * np.cos(node)
    z = y_orbit * np.sin(inclination)
    angle = earth_rotation_angle(times)
    return np.stack((np.cos(angle) * x + np.sin(angle) * y, -np.sin(angle) * x + np.cos(angle) * y, z), axis=1)


def transmit_positions(elements, reception_time, start_time, receiver):
    """Paper Eq. (5): satellite positions at T1 = T2 - dT, iterated until they change < epsilon_T."""
    transit = np.zeros(len(elements))
    positions = leo_positions(elements, reception_time - transit, start_time)
    for _ in range(20):
        angle = EARTH_ROTATION_RATE * transit                    # Earth rotation during transit
        rotated = np.stack((np.cos(angle) * positions[:, 0] + np.sin(angle) * positions[:, 1],
                            -np.sin(angle) * positions[:, 0] + np.cos(angle) * positions[:, 1],
                            positions[:, 2]), axis=1)
        transit = np.linalg.norm(rotated - receiver, axis=1) / SPEED_OF_LIGHT
        new_positions = leo_positions(elements, reception_time - transit, start_time)
        change = np.max(np.linalg.norm(new_positions - positions, axis=1))
        positions = new_positions
        if change < cfg.LIGHT_TIME_THRESHOLD:
            break
    return positions


# --- Error model from real data ----------------------------------------------
def real_error_bins(data, gnss_epochs):
    """Real GPS/BDS-3 pseudorange errors and median C/N0 per elevation bin.

    error = pseudorange - Eq. (1) prediction at the truth antenna position; the
    receiver clock of each system is removed with the median of the epoch.
    """
    bin_count = int(np.ceil(90.0 / cfg.LEO_ELEVATION_BIN_DEG))
    errors, cn0 = [[] for _ in range(bin_count)], [[] for _ in range(bin_count)]
    for k, meas in enumerate(gnss_epochs):
        predicted, _, elevation, _ = predict_pseudoranges(data.truth_antenna_position[k], meas, data.fusion_times[k],
                                                          data.klobuchar_alpha, data.klobuchar_beta)
        error = meas.pseudoranges - predicted
        for system in ('G', 'C'):
            rows = np.flatnonzero((meas.systems == system) & (elevation > 0.0))
            if len(rows) < 2:
                continue
            bins = np.minimum((np.rad2deg(elevation[rows]) // cfg.LEO_ELEVATION_BIN_DEG).astype(int), bin_count - 1)
            for b, e, c in zip(bins, error[rows] - np.median(error[rows]), meas.cn0[rows]):
                errors[b].append(e)
                cn0[b].append(c)
    filled = [b for b in range(bin_count) if errors[b]]
    nearest = [min(filled, key=lambda f: abs(f - b)) for b in range(bin_count)]
    return [(np.array(errors[f]), float(np.median(cn0[f]))) for f in nearest]


# --- Simulation --------------------------------------------------------------
def simulate_leo_measurements(data, error_bins, seed) -> list[EpochMeasurements]:
    """Simulated LEO pseudoranges at every fusion epoch, at the truth antenna position."""
    rng = np.random.default_rng(seed)
    elements = leo_orbit_elements()
    sat_ids = np.array([f'L{i + 1:03d}' for i in range(len(elements))])
    start_time = data.fusion_times[0]
    epochs = []
    for k, time in enumerate(data.fusion_times):
        receiver = data.truth_antenna_position[k]
        positions = transmit_positions(elements, time, start_time, receiver)
        elevation, _ = elevation_azimuth(receiver, positions)
        visible = np.flatnonzero(elevation >= np.deg2rad(cfg.LEO_ELEVATION_MASK_DEG))
        bins = np.minimum((np.rad2deg(elevation[visible]) // cfg.LEO_ELEVATION_BIN_DEG).astype(int),
                          len(error_bins) - 1)
        n = len(visible)
        meas = EpochMeasurements(sat_ids[visible], np.full(n, 'L'), np.zeros(n), positions[visible], np.zeros(n),
                                 np.array([error_bins[b][1] for b in bins]))
        predicted, _, _, _ = predict_pseudoranges(receiver, meas, time, data.klobuchar_alpha, data.klobuchar_beta)
        noise = np.array([rng.choice(error_bins[b][0]) for b in bins])
        meas.pseudoranges = predicted + noise
        epochs.append(meas)
    return epochs
