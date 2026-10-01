"""LEO signal-of-opportunity pseudorange simulation (paper Sec. II-A, Sec. III-A).

- Orbits (leo_orbits.py, A1): the measurement is made with the reference orbit
  (numerical, from the newest TLE); the filter gets the predicted orbit (SGP4
  of an older TLE) at t - rho/c.
- Transmit time: iteration of Eq. (5) until the satellite position changes by
  less than epsilon_T.
- Pseudorange: geometric range (with the Sagnac term) + noise; satellite and
  receiver clocks (A4) and ionosphere/troposphere (A24) are ideal, i.e. known
  and removed, so they are in neither the measurement nor the prediction.
- Noise, one draw per satellite and epoch (A3): receiver noise, Gaussian with
  the standard deviation of Eq. (3) (formula of Ref. [35]); MP/NLOS: standard
  deviation of Eq. (4) (Ref. [37]) times a non-Gaussian shape drawn from the
  real GPS/BDS-3 errors of the same dataset in the same elevation bin (paper
  Sec. III-A: statistics of the real data). The orbit error comes from the
  two orbits, so URA is not added.
- C/N0: mean C/N0 of the real GPS/BDS-3 observations in the same elevation bin
  (paper Fig. 5; Ref. [35] gives no elevation-C/N0 formula).
- Orbit error variance of the filter (A25): mean square range error of the
  predicted orbit on the training dataset, saved in leo_orbit_error_train.json.
"""
import json

import numpy as np

import settings as cfg
from earth_models import SPEED_OF_LIGHT, elevation_azimuth, multipath_variance, receiver_noise_std
from gnss_measurements import EpochMeasurements, geometric_range, predict_pseudoranges
from leo_orbits import leo_orbits


# --- Error model from real data ----------------------------------------------
def elevation_bin(elevation, bin_count):
    return np.minimum((np.rad2deg(elevation) // cfg.LEO_ELEVATION_BIN_DEG).astype(int), bin_count - 1)


def real_error_bins(data, gnss_epochs):
    """Per elevation bin of the real GPS/BDS-3 observations: (error shapes, mean C/N0).

    error = pseudorange - Eq. (1) prediction at the truth antenna position, the
    receiver clock of each system removed with the median of the epoch; the
    shape is this error divided by its Eq. (3) standard deviation.
    """
    bin_count = int(np.ceil(90.0 / cfg.LEO_ELEVATION_BIN_DEG))
    shapes, cn0 = [[] for _ in range(bin_count)], [[] for _ in range(bin_count)]
    for k, meas in enumerate(gnss_epochs):
        predicted, _, elevation, variance = predict_pseudoranges(data.truth_antenna_position[k], meas,
                                                                 data.fusion_times[k], data.klobuchar_alpha,
                                                                 data.klobuchar_beta)
        error = meas.pseudoranges - predicted
        for system in ('G', 'C'):
            rows = np.flatnonzero((meas.systems == system) & (elevation > 0.0))
            if len(rows) < 2:
                continue
            for b, e, s, c in zip(elevation_bin(elevation[rows], bin_count), error[rows] - np.median(error[rows]),
                                  np.sqrt(variance[rows]), meas.cn0[rows]):
                shapes[b].append(e / s)
                cn0[b].append(c)
    filled = [b for b in range(bin_count) if shapes[b]]
    nearest = [min(filled, key=lambda f: abs(f - b)) for b in range(bin_count)]
    return [(np.array(shapes[f]), float(np.mean(cn0[f]))) for f in nearest]


# --- Simulation --------------------------------------------------------------
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


def simulate_leo_measurements(data, error_bins, seed):
    """Simulated LEO pseudoranges at every fusion epoch, at the truth antenna position.

    Returns the epochs and the range errors of the predicted orbit (predicted
    minus true geometric range at the truth antenna) of all simulated rows.
    """
    rng = np.random.default_rng(seed)
    orbits = leo_orbits(data.fusion_times, data.truth_antenna_position)
    if not orbits:
        raise RuntimeError('No LEO satellite with usable TLEs over the dataset (see LEO_TLE_MIN_AGE in settings.py)')
    sat_ids = np.array([sat_id for sat_id, _, _ in orbits])
    reference_position, reference_velocity = (np.stack([o[1][i] for o in orbits]) for i in (0, 1))
    predicted_position, predicted_velocity = (np.stack([o[2][i] for o in orbits]) for i in (0, 1))
    epochs, range_errors = [], []
    for k in range(len(data.fusion_times)):
        receiver = data.truth_antenna_position[k]
        positions = transmit_positions(receiver, reference_position[:, k], reference_velocity[:, k])
        elevation, _ = elevation_azimuth(receiver, positions)
        visible = np.flatnonzero(elevation >= np.deg2rad(cfg.LEO_ELEVATION_MASK_DEG))
        bins = elevation_bin(elevation[visible], len(error_bins))
        cn0 = np.array([error_bins[b][1] for b in bins])
        shape = np.array([error_bins[b][0][rng.integers(len(error_bins[b][0]))] for b in bins])
        n = len(visible)
        true_range = geometric_range(receiver, positions[visible])
        noise = (receiver_noise_std(cn0, cfg.LEO_CODE_CHIP_RATE) * rng.standard_normal(n)
                 + np.sqrt(multipath_variance(elevation[visible], cn0)) * shape)
        pseudoranges = true_range + noise
        # Filter side: predicted orbit at the transmit time t - rho/c (clocks are ideal, A4).
        predicted = (predicted_position[visible, k]
                     - (pseudoranges / SPEED_OF_LIGHT)[:, None] * predicted_velocity[visible, k])
        range_errors.append(geometric_range(receiver, predicted) - true_range)
        epochs.append(EpochMeasurements(sat_ids[visible], np.full(n, 'L'), pseudoranges, predicted, np.zeros(n), cn0,
                                        np.zeros(n)))
    return epochs, np.concatenate(range_errors)


def orbit_error_variance(range_errors, split):
    """sigma^2 of the predicted orbit in the filter R (A25).

    Writes leo_orbit_error_<split>.json (RMS, mean, samples) and returns the
    mean square range error of the training dataset (written by train.py).
    """
    summary = {'rms_m': float(np.sqrt(np.mean(range_errors ** 2))) if len(range_errors) else 0.0,
               'mean_m': float(np.mean(range_errors)) if len(range_errors) else 0.0,
               'samples': int(len(range_errors))}
    cfg.OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    (cfg.OUTPUT_FOLDER / f'leo_orbit_error_{split}.json').write_text(json.dumps(summary, indent=1))
    train_file = cfg.OUTPUT_FOLDER / 'leo_orbit_error_train.json'
    if not train_file.exists():
        raise FileNotFoundError(f'{train_file} is written by train.py; run train.py first')
    return json.loads(train_file.read_text())['rms_m'] ** 2
