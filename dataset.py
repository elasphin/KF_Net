"""Read one SmartPNT-POS dataset folder and synchronize GNSS, IMU and truth.

Files used (per folder): README.xml, ROVE*.O (RINEX 3 observations),
ROVE*GroundTruth.txt (antenna truth), <IMU>_GroundTruth.txt (IMU truth),
<IMU>.imr (IMU), and the products *.sp3, *.clk, brdm* (in the folder, in
settings.PRODUCTS_FOLDER or in settings.DATASET_FOLDER itself, searched in this order), and
IMUErrorModel.txt of the dataset (initial bias standard deviations, in the folder or above it).

The dataset folder is searched in settings.DATASET_FOLDER (any depth): the Google
Drive folder in Colab, /kaggle/input on Kaggle (no download), ./Dataset on my
computer. If it is not there, only the files this project needs are downloaded
with kagglehub; this needs a Kaggle API token (~/.kaggle/kaggle.json or
KAGGLE_USERNAME / KAGGLE_KEY).

Read dataset and simulated measurements of 'train' or 'test', kept on disk between runs (settings.DATA_CACHE).

    from dataset import load_dataset
    data, measurements = load_dataset('train')      # measurements: {LEO filter orbit: epochs} (A26)

Reading the RINEX/IMU/truth files, the GNSS satellite orbits and the LEO orbit integration and
simulation give the same result in every run (fixed seeds), so they are done once and kept in
EXPERIMENT_FOLDER/cache/<split>_<key>.pkl (shared by the runs of every learning rate). The key is a hash of everything they depend on:
  - all settings except the network, training, integrity and run-time ones (NOT_DATA_SETTINGS),
  - the LEO filter orbits of the split (LEO_TRAIN_ORBIT or LEO_TEST_ORBITS, A26) and its MAX_FUSION_EPOCHS,
  - the code that makes them (DATA_CODE files),
  - name, size and modification time of the input files (data folder, products, dataset root, TLE files).
A change in any of these makes a new cache file (the old one of that split is removed). The LEO
orbit error variance of the filter R (A25) is not kept: it is set from leo_orbit_error_train.json
in every run, as before.

Statistics taken from the data (LEO C/N0 per elevation, A3; orbit term of R, A25): from the whole dataset (the
whole training dataset trains the network, as in the paper; the testing dataset has its own environment).
"""
import hashlib
import math
import pickle
import struct
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path, PurePosixPath

import numpy as np

import settings as cfg
from leo_pseudorange import find_tle_folder, mean_cn0_bins, orbit_error_variance, simulate_leo_measurements
from measurements import (GPS_UTC_LEAP_SECONDS, GPS_WEEK_SECONDS, ecef_to_llh, ecef_to_ned_matrix, gps_seconds,
                          merge_measurements, prepare_gnss_measurements)


# ===== Reading of the dataset =========================================================================================
IMR_HEADER_SIZE = 512
TRUTH_TAIL_BYTES = 100_000         # end of a truth file read for its last time
TRUTH_MARGIN = 10.0                # s, truth rows kept before the first and after the last fusion epoch (interpolation)
PRODUCT_MARGIN = 3 * 3600.0        # s, SP3/CLK records kept around the fusion epochs (> 10-point Lagrange of 15-min SP3)
VEHICLE_TO_NAVIGATION = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
# Files of one dataset folder used by this project ({imu} = IMU type from README.xml).
NEEDED_FILE_PATTERNS = ('README.xml', 'ROVE*GroundTruth.txt', 'Rove*GroundTruth.txt', '{imu}_GroundTruth.txt',
                        '{imu}.imr', 'ROVE*.*[oO]', '*.[sS][pP]3', '*.[cC][lL][kK]', '[bB][rR][dD][mM]*',
                        cfg.IMU_ERROR_MODEL_FILE_NAME)


@dataclass
class NavigationData:
    name: str
    fusion_times: np.ndarray          # GNSS epochs used for fusion, GPST [s]
    gnss_observations: list           # per fusion epoch: [(sat_id, pseudorange [m], C/N0 [dB-Hz]), ...]
    imu_times: np.ndarray             # GPST [s]
    gyro: np.ndarray                  # angular rate [rad/s], body frame
    accel: np.ndarray                 # specific force [m/s^2], body frame
    truth_position: np.ndarray        # IMU truth at fusion epochs, ECEF [m]
    truth_velocity: np.ndarray        # ECEF [m/s]
    truth_attitude: np.ndarray        # C_ecef<-body
    truth_antenna_position: np.ndarray
    lever_arm: np.ndarray             # IMU -> antenna, body frame [m]
    orbits: dict                      # SP3: sat_id -> (times, positions)
    clocks: dict                      # CLK: sat_id -> (times, clock offsets [s])
    klobuchar_alpha: np.ndarray
    klobuchar_beta: np.ndarray
    accel_bias_std: np.ndarray        # initial standard deviation of the accelerometer biases [m/s^2] (A7)
    gyro_bias_std: np.ndarray         # initial standard deviation of the gyro biases [rad/s] (A7)


# --- Small helpers -----------------------------------------------------------
def find_one_file(folders, patterns, label) -> Path:
    for folder in folders:
        matches = sorted({p for pattern in patterns for p in Path(folder).glob(pattern) if p.is_file()})
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise FileNotFoundError(f'More than one {label} file in {folder}: {[p.name for p in matches]}')
    raise FileNotFoundError(f'No {label} file ({patterns}) in {[str(f) for f in folders]}')


def rinex_number(text):
    text = text.strip()
    return float(text.replace('D', 'E')) if text else None


# --- README.xml --------------------------------------------------------------
def read_rover_info(readme_path: Path):
    """IMU type, mounting angles [deg] and GNSS lever arm (vehicle frame) of rover 01."""
    for rover in ET.parse(readme_path).getroot().iter('ROVE'):
        if (rover.findtext('ID') or '').strip() == '01':
            imu_type = rover.findtext('SINS_IMUType').strip()
            mounting = np.array(rover.findtext('SINS_RotAngle_IMU').split(), dtype=float)
            lever_arm = np.array(rover.findtext('SINS_LeverArm_GNSS').split(), dtype=float)
            return imu_type, mounting, lever_arm
    raise ValueError(f'{readme_path} has no rover with ID 01')


def vehicle_to_body(mounting_deg):
    """SINS_RotAngle_IMU -> C_body<-vehicle."""
    gamma, beta, alpha = np.deg2rad(mounting_deg)
    cb, sb, cg, sg, ca, sa = np.cos(beta), np.sin(beta), np.cos(gamma), np.sin(gamma), np.cos(alpha), np.sin(alpha)
    Ry = np.array([[cb, 0.0, -sb], [0.0, 1.0, 0.0], [sb, 0.0, cb]])
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cg, sg], [0.0, -sg, cg]])
    Rz = np.array([[ca, sa, 0.0], [-sa, ca, 0.0], [0.0, 0.0, 1.0]])
    return Ry @ Rx @ Rz


def truth_attitude_matrix(position, heading_deg, pitch_deg, roll_deg, mounting_deg):
    """Truth heading/pitch/roll (Inertial Explorer) -> C_ecef<-body."""
    latitude, longitude, _ = ecef_to_llh(position)
    h, p, r = np.deg2rad([heading_deg, pitch_deg, roll_deg])
    ch, sh, cp, sp, cr, sr = np.cos(h), np.sin(h), np.cos(p), np.sin(p), np.cos(r), np.sin(r)
    C_nav_vehicle = np.array([[cp * ch, sr * sp * ch - cr * sh, cr * sp * ch + sr * sh],
                              [cp * sh, sr * sp * sh + cr * ch, cr * sp * sh - sr * ch],
                              [-sp, sr * cp, cr * cp]]) @ VEHICLE_TO_NAVIGATION
    C = ecef_to_ned_matrix(latitude, longitude).T @ C_nav_vehicle @ vehicle_to_body(mounting_deg).T
    U, _, Vt = np.linalg.svd(C)
    return U @ Vt


# --- Truth (Inertial Explorer text export) -----------------------------------
def truth_row(line):
    """Columns: week, seconds of week, ECEF position (9:12), ECEF velocity (15:18), heading/pitch/roll (21:24)."""
    v = line.split()
    return [int(v[0]) * GPS_WEEK_SECONDS + float(v[1]), *map(float, v[9:12]), *map(float, v[15:18]),
            *map(float, v[21:24])]


def first_truth_time(lines):
    for line in lines:
        try:
            return truth_row(line)[0]
        except (ValueError, IndexError):
            pass
    raise ValueError('no truth row')


def truth_time_span(path: Path):
    """(first, last) time of the truth table, from the start and the end of the file only."""
    with path.open('rb') as f:
        f.seek(max(0, f.seek(0, 2) - TRUTH_TAIL_BYTES))
        tail = f.read().decode(errors='replace').splitlines()
    with path.open(errors='replace') as f:
        return first_truth_time(f), first_truth_time(reversed(tail))


def read_truth(path: Path, first, last):
    """Truth rows with first <= time <= last (the file is in time order)."""
    rows = []
    with path.open(errors='replace') as f:
        for line in f:
            try:
                row = truth_row(line)
            except (ValueError, IndexError):
                if rows:
                    break
                continue
            if row[0] > last:
                break
            if row[0] >= first:
                rows.append(row)
    return np.array(rows)


def interpolate_truth(truth, times):
    """Linear interpolation of the truth table (heading unwrapped)."""
    columns = truth[:, 1:].copy()
    columns[:, 6] = np.unwrap(np.deg2rad(columns[:, 6]))
    out = np.stack([np.interp(times, truth[:, 0], c) for c in columns.T], axis=1)
    out[:, 6] = np.rad2deg(out[:, 6]) % 360.0
    return out[:, 0:3], out[:, 3:6], out[:, 6], out[:, 7], out[:, 8]


# --- IMU (NovAtel Inertial Explorer .imr) ------------------------------------
def read_imu(path: Path, reference_time: float):
    """-> times (GPST), gyro [rad/s], accel [m/s^2].

    Header flags used: delta theta / delta velocity (increments are multiplied by the rate) and the
    time tag type (UTC tags are moved to GPST). The time tag bias is taken in ms.
    """
    with path.open('rb') as f:
        header = f.read(IMR_HEADER_SIZE)
    endian = '<' if header[8] == 0 else '>'
    fields = struct.unpack(endian + '8scdiidddiid32s?BBB32s6h?iii354s', header)
    delta_theta, delta_velocity, rate, gyro_scale, accel_scale, time_type, time_bias_ms = (
        fields[3], fields[4], fields[5], fields[6], fields[7], fields[8], fields[10])
    record = np.dtype([('tow', endian + 'f8'), ('counts', endian + 'i4', (6,))])
    data = np.fromfile(path, dtype=record, offset=IMR_HEADER_SIZE)
    tow = data['tow'] - time_bias_ms * 1e-3 + (GPS_UTC_LEAP_SECONDS if time_type == 1 else 0.0)   # 1 = UTC, 2 = GPS
    tow[tow > GPS_WEEK_SECONDS] -= GPS_WEEK_SECONDS
    week_start = math.floor(reference_time / GPS_WEEK_SECONDS) * GPS_WEEK_SECONDS
    times = week_start + tow
    times[times - reference_time > GPS_WEEK_SECONDS / 2] -= GPS_WEEK_SECONDS
    times[times - reference_time < -GPS_WEEK_SECONDS / 2] += GPS_WEEK_SECONDS
    gyro = np.deg2rad(data['counts'][:, 0:3] * gyro_scale * (rate if delta_theta else 1.0))   # increments -> rates
    accel = data['counts'][:, 3:6] * accel_scale * (rate if delta_velocity else 1.0)
    return times, gyro, accel


# --- RINEX 3 observations ----------------------------------------------------
def read_rinex_observations(path: Path, start, stop, max_epochs=None):
    """GPS L1 C/A (C1C) and BDS-3 B1I (C2I, or C1I in RINEX < 3.03) pseudoranges with C/N0.

    Returns [(time GPST, [(sat_id, pseudorange, cn0), ...]), ...] of the first max_epochs epochs
    with observations and start <= time <= stop (None = all of them).
    """
    with path.open(errors='replace') as f:
        version = float(f.readline()[:9])
        types, system, time_system = {}, None, 'GPS'
        for line in f:
            label = line[60:].strip()
            if label == 'END OF HEADER':
                break
            if label == 'TIME OF FIRST OBS':
                time_system = line[48:51].strip() or 'GPS'
            elif label == 'SYS / # / OBS TYPES':
                if line[0] != ' ':
                    system = line[0]
                    types[system] = []
                types[system] += line[7:60].split()
        codes = {'G': ('C1C', 'S1C')}
        if 'C2I' in types.get('C', []):
            codes['C'] = ('C2I', 'S2I')
        elif 'C1I' in types.get('C', []) and version < 3.03:
            codes['C'] = ('C1I', 'S1I')

        epochs = []
        for line in f:
            if not line.startswith('>'):
                continue
            parts = line[1:].split()
            satellite_lines = [f.readline() for _ in range(int(parts[7]))]
            if int(parts[6]) > 1:                          # event records, not observations
                continue
            time = gps_seconds(*map(int, parts[:5]), float(parts[5]), time_system)
            if time < start:
                continue
            if time > stop:
                break
            observations = []
            for text in satellite_lines:
                sat_id = text[:3].replace(' ', '0')
                if sat_id[0] not in codes or (sat_id[0] == 'C' and int(sat_id[1:]) < cfg.BDS3_FIRST_PRN):
                    continue
                code_name, snr_name = codes[sat_id[0]]
                if code_name not in types[sat_id[0]] or snr_name not in types[sat_id[0]]:
                    continue
                code, snr = types[sat_id[0]].index(code_name), types[sat_id[0]].index(snr_name)
                pseudorange = rinex_number(text[3 + 16 * code: 17 + 16 * code])
                cn0 = rinex_number(text[3 + 16 * snr: 17 + 16 * snr])
                if pseudorange and cn0:
                    observations.append((sat_id, pseudorange, cn0))
            if observations:
                epochs.append((time, observations))
                if len(epochs) == max_epochs:
                    break
    return epochs


# --- Products ----------------------------------------------------------------
def read_sp3(path: Path, first, last):
    """SP3 precise orbits with first <= time <= last (the file is in time order) -> {sat_id: (times, positions [m])}."""
    orbits, time, time_system, first_c_line_read = {}, None, 'GPS', False
    for line in path.read_text(errors='replace').splitlines():
        if line.startswith('%c'):
            if not first_c_line_read and line[9:12].strip():      # SP3-c/d: only the first %c line has it
                time_system = line[9:12].strip()
            first_c_line_read = True
        elif line.startswith('*'):
            v = line[1:].split()
            time = gps_seconds(*map(int, v[:5]), float(v[5]), time_system)
            if time > last:
                break
        elif line.startswith('P') and time is not None and time >= first:
            xyz = np.array(line[4:46].split(), dtype=float)
            if np.any(xyz != 0.0) and np.all(np.abs(xyz) < 999999.0):
                orbits.setdefault(line[1:4].replace(' ', '0'), []).append((time, *(xyz * 1000.0)))
    return {sat: (np.array(v)[:, 0], np.array(v)[:, 1:]) for sat, v in orbits.items()}


def read_clock(path: Path, first, last):
    """RINEX clock with first <= time <= last (the file is in time order) -> {sat_id: (times, clock offsets [s])}."""
    clocks, time_system = {}, 'GPS'
    with path.open(errors='replace') as f:
        for line in f:
            label = line[60:].strip()
            if label == 'TIME SYSTEM ID':
                time_system = line[:10].split()[0]
            if label == 'END OF HEADER':
                break
        for line in f:
            if line.startswith('AS '):
                v = line.split()
                time = gps_seconds(*map(int, v[2:7]), float(v[7]), time_system)
                if time > last:
                    break
                if time >= first:
                    clocks.setdefault(v[1], []).append((time, float(v[9].replace('D', 'E'))))
    return {sat: (np.array(v)[:, 0], np.array(v)[:, 1]) for sat, v in clocks.items()}


def read_klobuchar(path: Path):
    """GPS Klobuchar alpha and beta from the header of a RINEX 3 navigation file."""
    values = {}
    with path.open(errors='replace') as f:
        for line in f:
            if line[60:].strip() == 'END OF HEADER':
                break
            if line[60:].strip() == 'IONOSPHERIC CORR' and line[:4] in ('GPSA', 'GPSB'):
                values[line[:4]] = np.array(line[5:60].replace('D', 'E').split(), dtype=float)
    return values['GPSA'], values['GPSB']


# --- IMU error model of the dataset (IMUErrorModel.txt) ---------------------
def find_imu_error_model(folder: Path) -> Path:
    """settings.IMU_ERROR_MODEL_FILE_NAME in the data folder or a folder above it, else anywhere in DATASET_FOLDER."""
    for parent in (folder, *folder.parents):
        if (parent / cfg.IMU_ERROR_MODEL_FILE_NAME).is_file():
            return parent / cfg.IMU_ERROR_MODEL_FILE_NAME
    matches = sorted(cfg.DATASET_FOLDER.rglob(cfg.IMU_ERROR_MODEL_FILE_NAME)) if cfg.DATASET_FOLDER.is_dir() else []
    if matches:
        return matches[0]
    raise FileNotFoundError(f'{cfg.IMU_ERROR_MODEL_FILE_NAME} (SmartPNT-POS IMU error model) was not found in '
                            f'{folder}, the folders above it or {cfg.DATASET_FOLDER}')


def read_imu_error_model(path: Path, imu_type: str):
    """Initial standard deviations of the biases of imu_type (A7) -> accelerometer [m/s^2], gyro [rad/s].

    The file has one 'IMU { ... }' block per IMU type with ISDV_AccelBias [m/s^2] and ISDV_GyrosBias [deg/s].
    """
    blocks, block = {}, None
    for line in path.read_text(errors='replace').splitlines():
        line = line.split('#')[0].strip()
        if line.startswith('IMU') and line.endswith('{'):
            block = {}
        elif line == '}' and block is not None:
            blocks[block.get('IMU_Type', '')], block = block, None
        elif '=' in line and block is not None:
            key, value = line.split('=', 1)
            block[key.strip()] = value.strip().strip('"')
    if imu_type not in blocks:
        raise ValueError(f'{path} has no IMU_Type "{imu_type}" (types: {", ".join(blocks)})')
    values = blocks[imu_type]
    return (np.array(values['ISDV_AccelBias'].split(), dtype=float),
            np.deg2rad(np.array(values['ISDV_GyrosBias'].split(), dtype=float)))


# --- Find or download the dataset folder -------------------------------------
def find_dataset_folder(folder_name: str) -> Path:
    if cfg.DATASET_FOLDER.is_dir():
        matches = sorted(p for p in cfg.DATASET_FOLDER.rglob(folder_name) if p.is_dir())
        if matches:
            return matches[0]
    return download_dataset_folder(folder_name)


def list_kaggle_files() -> list[str]:
    from kagglehub.clients import build_kaggle_client
    from kagglesdk.datasets.types.dataset_api_service import ApiListDatasetFilesRequest

    owner, dataset = cfg.KAGGLE_DATASET.split('/')
    names, page_token = [], None
    with build_kaggle_client() as client:
        while True:
            request = ApiListDatasetFilesRequest()
            request.owner_slug, request.dataset_slug, request.page_size = owner, dataset, 200
            if page_token:
                request.page_token = page_token
            response = client.datasets.dataset_api_client.list_dataset_files(request)
            names += [f.name for f in response.dataset_files]
            page_token = response.next_page_token
            if not page_token:
                return names


def download_dataset_folder(folder_name: str) -> Path:
    import kagglehub

    names = list_kaggle_files()
    files = [n for n in names if folder_name in PurePosixPath(n).parts]
    if not files:
        raise FileNotFoundError(f'{folder_name} was not found in the Kaggle dataset {cfg.KAGGLE_DATASET}. '
                                f'Set TRAIN_FOLDER_NAME / TEST_FOLDER_NAME in settings.py.')

    def download(names):
        return [Path(kagglehub.dataset_download(cfg.KAGGLE_DATASET, path=n)) for n in names]

    readme = download([n for n in files if PurePosixPath(n).name == 'README.xml'])
    if not readme:
        raise FileNotFoundError(f'README.xml is missing in {folder_name} on Kaggle')
    imu_type = read_rover_info(readme[0])[0]
    patterns = [p.format(imu=imu_type) for p in NEEDED_FILE_PATTERNS]
    download([n for n in files if any(fnmatch(PurePosixPath(n).name, p) for p in patterns)])
    if not any(PurePosixPath(n).name == cfg.IMU_ERROR_MODEL_FILE_NAME for n in files):   # e.g. at the dataset root
        download([n for n in names if PurePosixPath(n).name == cfg.IMU_ERROR_MODEL_FILE_NAME][:1])
    return readme[0].parent


# --- Dataset -----------------------------------------------------------------
def load_navigation_data(split: str) -> NavigationData:
    """Load the 'train' or 'test' dataset (paper Sec. III) on the GNSS epochs.

    Only the span of the fusion epochs (settings.MAX_FUSION_EPOCHS[split]) is read from the RINEX,
    truth, SP3 and CLK files.
    """
    folder = find_dataset_folder(cfg.TRAIN_FOLDER_NAME if split == 'train' else cfg.TEST_FOLDER_NAME)
    product_folders = (folder, cfg.PRODUCTS_FOLDER, cfg.DATASET_FOLDER)
    imu_type, mounting, lever_arm_vehicle = read_rover_info(folder / 'README.xml')

    antenna_file = find_one_file([folder], ('ROVE*GroundTruth.txt', 'Rove*GroundTruth.txt'), 'antenna truth')
    imu_truth_file = folder / f'{imu_type}_GroundTruth.txt'
    antenna_span, imu_truth_span = truth_time_span(antenna_file), truth_time_span(imu_truth_file)
    imu_times, gyro, accel = read_imu(folder / f'{imu_type}.imr', antenna_span[0])

    # Fusion epochs: the first MAX_FUSION_EPOCHS[split] GNSS epochs inside the IMU and both truth spans.
    start = max(imu_times[0], antenna_span[0], imu_truth_span[0])
    stop = min(imu_times[-1], antenna_span[1], imu_truth_span[1])
    gnss = read_rinex_observations(find_one_file([folder], ('ROVE*.*[oO]',), 'RINEX observation'), start, stop,
                                   cfg.MAX_FUSION_EPOCHS[split])
    fusion_times = np.array([t for t, _ in gnss])
    keep = (imu_times >= fusion_times[0] - 1.0) & (imu_times <= fusion_times[-1] + 1.0)
    first, last = fusion_times[0] - TRUTH_MARGIN, fusion_times[-1] + TRUTH_MARGIN
    imu_truth, antenna_truth = read_truth(imu_truth_file, first, last), read_truth(antenna_file, first, last)
    products = (fusion_times[0] - PRODUCT_MARGIN, fusion_times[-1] + PRODUCT_MARGIN)

    position, velocity, heading, pitch, roll = interpolate_truth(imu_truth, fusion_times)
    attitude = np.stack([truth_attitude_matrix(*row, mounting) for row in zip(position, heading, pitch, roll)])
    antenna_position = interpolate_truth(antenna_truth, fusion_times)[0]
    orbit_file = find_one_file(product_folders, ('*.[sS][pP]3',), 'SP3 orbit')
    clock_file = find_one_file(product_folders, ('*.[cC][lL][kK]',), 'RINEX clock')
    orbits, clocks = read_sp3(orbit_file, *products), read_clock(clock_file, *products)
    if not orbits or not clocks:
        raise FileNotFoundError(f'{orbit_file} or {clock_file} does not cover {folder.name}; '
                                f'put the SP3 and CLK of its day in {folder}')
    alpha, beta = read_klobuchar(find_one_file(product_folders, ('[bB][rR][dD][mM]*',), 'broadcast navigation'))
    accel_bias_std, gyro_bias_std = read_imu_error_model(find_imu_error_model(folder), imu_type)

    return NavigationData(
        name=folder.name, fusion_times=fusion_times, gnss_observations=[obs for _, obs in gnss],
        imu_times=imu_times[keep], gyro=gyro[keep], accel=accel[keep],
        truth_position=position, truth_velocity=velocity, truth_attitude=attitude,
        truth_antenna_position=antenna_position, lever_arm=vehicle_to_body(mounting) @ lever_arm_vehicle,
        orbits=orbits, clocks=clocks,
        klobuchar_alpha=alpha, klobuchar_beta=beta, accel_bias_std=accel_bias_std, gyro_bias_std=gyro_bias_std)


# ===== Dataset and simulated measurements kept between runs ===========================================================
DATA_CODE = ('dataset.py', 'measurements.py', 'leo_pseudorange.py', 'egm96_degree20.txt')
NOT_DATA_SETTINGS = {
    'CONV_FILTERS', 'CONV_KERNEL_SIZE', 'POOL_KERNEL_SIZE', 'LSTM_UNITS', 'LSTM_LAYERS', 'LSTM_DROPOUT',
    'FC_HIDDEN_UNITS', 'MASK_EPSILON', 'RANDOM_SEED', 'LEARNING_RATE', 'TRAINING_EPOCHS', 'L2_WEIGHT',
    'BACKPROP_WINDOW',
    'FALSE_ALARM_PROBABILITY', 'HORIZONTAL_PL_FACTOR', 'VERTICAL_PL_FACTOR', 'ALERT_LIMIT',
    'INS_MECHANIZATION', 'DATA_CACHE',
    'LEO_TRAIN_ORBIT', 'LEO_TEST_ORBITS',          # only the orbits of the split are in the key (split_orbits)
    'MAX_FUSION_EPOCHS',                           # only the limit of the split is in the key
    'TRAIN_FOLDER_NAME', 'TEST_FOLDER_NAME',      # the folder of the split is in the key itself
    'PROJECT_FOLDER', 'COLAB_FOLDER', 'COLAB_OUTPUT_FOLDER', 'KAGGLE_FOLDER', 'KAGGLE_OUTPUT_FOLDER', 'LOCAL_FOLDER',
    'LOCAL_OUTPUT_FOLDER',                        # candidates of DATASET_FOLDER / OUTPUT_ROOT (these are in the key)
    'OUTPUT_FOLDER',                              # the folder of one learning rate; EXPERIMENT_FOLDER is in the key
}


def split_orbits(split):
    """LEO filter orbits of the split (A26): the training orbit, or every test orbit."""
    return (cfg.LEO_TRAIN_ORBIT,) if split == 'train' else tuple(cfg.LEO_TEST_ORBITS)


def simulate_measurements(data, split):
    """GPS + BDS-3 (real) and LEO (simulated) measurements of every fusion epoch, LEO orbit variance not yet set.

    Returns (GNSS epochs, {LEO filter orbit: LEO epochs}, {LEO filter orbit: range errors}).
    """
    start = time.time()
    gnss = prepare_gnss_measurements(data)
    print(f'GNSS measurements: {time.time() - start:.1f} s')
    start = time.time()
    cn0_bins = mean_cn0_bins(data, gnss, range(len(data.fusion_times)))          # whole dataset (A3)
    leo, range_errors = simulate_leo_measurements(data, cn0_bins, cfg.LEO_NOISE_SEED[split], split_orbits(split))
    print(f'LEO orbits and measurements: {time.time() - start:.1f} s')
    return gnss, leo, range_errors


def prepare_measurements(split, gnss, leo, range_errors):
    """{LEO filter orbit: merged measurements of every fusion epoch}, with the LEO orbit error variance of R.

    The variance is the same for every filter orbit: that of LEO_TRAIN_ORBIT on the training dataset (A25, A26).
    """
    variance = orbit_error_variance({name: np.concatenate(errors) for name, errors in range_errors.items()}, split)
    measurements = {}
    for name, epochs in leo.items():
        for meas in epochs:
            meas.orbit_variance[:] = variance
        measurements[name] = [merge_measurements(g, l) for g, l in zip(gnss, epochs)]
    return measurements


def input_files(folder):
    """Files read for one dataset folder: the folder itself, the products folders, the IMU error model and the
    LEO TLE files."""
    files = [p for p in folder.iterdir() if p.is_file()]
    for products in (cfg.PRODUCTS_FOLDER, cfg.DATASET_FOLDER):
        if products.is_dir():
            files += [p for p in products.iterdir() if p.is_file()]
    return sorted(set(files) | {find_imu_error_model(folder)}) + sorted(find_tle_folder().rglob('*.txt'))


def cache_key(split):
    folder_name = cfg.TRAIN_FOLDER_NAME if split == 'train' else cfg.TEST_FOLDER_NAME
    folder = find_dataset_folder(folder_name)
    digest = hashlib.sha256(f'{split} {folder_name} {split_orbits(split)} {cfg.MAX_FUSION_EPOCHS[split]}'.encode())
    for name, value in sorted(vars(cfg).items()):
        if name.isupper() and name not in NOT_DATA_SETTINGS:
            digest.update(f'{name}={value!r}\n'.encode())
    for name in DATA_CODE:
        digest.update((cfg.PROJECT_FOLDER / name).read_bytes())
    for path in input_files(folder):
        stat = path.stat()
        digest.update(f'{path} {stat.st_size} {stat.st_mtime_ns}\n'.encode())
    return digest.hexdigest()[:16]


def read_and_simulate_now(split):
    """(data, GNSS epochs, LEO epochs, LEO range errors), computed, with the time of each stage printed."""
    start = time.time()
    data = load_navigation_data(split)
    print(f'{split} dataset read ({len(data.fusion_times)} fusion epochs): {time.time() - start:.1f} s')
    return (data, *simulate_measurements(data, split))


def read_and_simulate(split):
    """(data, GNSS epochs, LEO epochs, LEO range errors): from the cache, or computed and then kept."""
    if not cfg.DATA_CACHE:
        return read_and_simulate_now(split)
    folder = cfg.EXPERIMENT_FOLDER / 'cache'
    path = folder / f'{split}_{cache_key(split)}.pkl'
    if path.exists():
        print(f'dataset and measurements read from {path}')
        with path.open('rb') as f:
            return pickle.load(f)
    result = read_and_simulate_now(split)
    folder.mkdir(parents=True, exist_ok=True)
    for old in folder.glob(f'{split}_*.pkl'):
        old.unlink()
    temporary = path.with_suffix('.tmp')
    with temporary.open('wb') as f:
        pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)
    print(f'dataset and measurements kept in {path}')
    return result


def load_dataset(split):
    """Dataset and {LEO filter orbit: merged GNSS + LEO measurements of every fusion epoch} of 'train' or 'test'."""
    data, gnss, leo, range_errors = read_and_simulate(split)
    return data, prepare_measurements(split, gnss, leo, range_errors)
