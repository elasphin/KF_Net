"""Read one SmartPNT-POS dataset folder and synchronize GNSS, IMU and truth.

Files used (per folder): README.xml, ROVE*.O (RINEX 3 observations),
ROVE*GroundTruth.txt (antenna truth), <IMU>_GroundTruth.txt (IMU truth),
<IMU>.imr (IMU), and the products *.sp3, *.clk, brdm* (in the folder or in
settings.PRODUCTS_FOLDER).

The dataset folder is searched in settings.DATASET_FOLDER (any depth): the Google
Drive folder in Colab, /kaggle/input on Kaggle (no download), ./Dataset on my
computer. If it is not there, only the files this project needs are downloaded
with kagglehub; this needs a Kaggle API token (~/.kaggle/kaggle.json or
KAGGLE_USERNAME / KAGGLE_KEY).
"""
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path, PurePosixPath
import math
import struct
import xml.etree.ElementTree as ET

import numpy as np

import settings as cfg
from earth_models import GPS_WEEK_SECONDS, ecef_to_llh, ecef_to_ned_matrix, gps_seconds

IMR_HEADER_SIZE = 512
VEHICLE_TO_NAVIGATION = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
# Files of one dataset folder used by this project ({imu} = IMU type from README.xml).
NEEDED_FILE_PATTERNS = ('README.xml', 'ROVE*GroundTruth.txt', 'Rove*GroundTruth.txt', '{imu}_GroundTruth.txt',
                        '{imu}.imr', 'ROVE*.*[oO]', '*.[sS][pP]3', '*.[cC][lL][kK]', '[bB][rR][dD][mM]*')


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
def read_truth(path: Path):
    """Columns: week, seconds of week, ECEF position (9:12), ECEF velocity (15:18), heading/pitch/roll (21:24)."""
    rows = []
    for line in path.read_text(errors='replace').splitlines():
        v = line.split()
        try:
            rows.append([int(v[0]) * GPS_WEEK_SECONDS + float(v[1]), *map(float, v[9:12]), *map(float, v[15:18]),
                         *map(float, v[21:24])])
        except (ValueError, IndexError):
            if rows:
                break
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
    """-> times (GPST), gyro [rad/s], accel [m/s^2]."""
    with path.open('rb') as f:
        header = f.read(IMR_HEADER_SIZE)
    endian = '<' if header[8] == 0 else '>'
    fields = struct.unpack(endian + '8scdiidddiid32s?BBB32s6h?iii354s', header)
    rate, gyro_scale, accel_scale, time_bias_ms = fields[5], fields[6], fields[7], fields[10]
    record = np.dtype([('tow', endian + 'f8'), ('counts', endian + 'i4', (6,))])
    data = np.fromfile(path, dtype=record, offset=IMR_HEADER_SIZE)
    tow = data['tow'] - time_bias_ms * 1e-3
    tow[tow > GPS_WEEK_SECONDS] -= GPS_WEEK_SECONDS
    week_start = math.floor(reference_time / GPS_WEEK_SECONDS) * GPS_WEEK_SECONDS
    times = week_start + tow
    times[times - reference_time > GPS_WEEK_SECONDS / 2] -= GPS_WEEK_SECONDS
    times[times - reference_time < -GPS_WEEK_SECONDS / 2] += GPS_WEEK_SECONDS
    gyro = np.deg2rad(data['counts'][:, 0:3] * gyro_scale * rate)
    accel = data['counts'][:, 3:6] * accel_scale * rate
    return times, gyro, accel


# --- RINEX 3 observations ----------------------------------------------------
def read_rinex_observations(path: Path):
    """GPS L1 C/A (C1C) and BDS-3 B1I (C2I, or C1I in RINEX < 3.03) pseudoranges with C/N0.

    Returns [(time GPST, [(sat_id, pseudorange, cn0), ...]), ...].
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
            time = gps_seconds(*map(int, parts[:5]), float(parts[5]), time_system)
            epochs.append((time, observations))
    return epochs


# --- Products ----------------------------------------------------------------
def read_sp3(path: Path):
    """SP3 precise orbits -> {sat_id: (times, positions [m])}."""
    orbits, time, time_system = {}, None, 'GPS'
    for line in path.read_text(errors='replace').splitlines():
        if line.startswith('%c') and line[9:12].strip():
            time_system = line[9:12].strip()
        elif line.startswith('*'):
            v = line[1:].split()
            time = gps_seconds(*map(int, v[:5]), float(v[5]), time_system)
        elif line.startswith('P') and time is not None:
            xyz = np.array(line[4:46].split(), dtype=float)
            if np.any(xyz != 0.0) and np.all(np.abs(xyz) < 999999.0):
                orbits.setdefault(line[1:4].replace(' ', '0'), []).append((time, *(xyz * 1000.0)))
    return {sat: (np.array(v)[:, 0], np.array(v)[:, 1:]) for sat, v in orbits.items()}


def read_clock(path: Path):
    """RINEX clock -> {sat_id: (times, clock offsets [s])}."""
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

    files = [n for n in list_kaggle_files() if folder_name in PurePosixPath(n).parts]
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
    return readme[0].parent


# --- Dataset -----------------------------------------------------------------
def load_navigation_data(split: str) -> NavigationData:
    """Load the 'train' or 'test' dataset (paper Sec. III) on the GNSS epochs."""
    folder = find_dataset_folder(cfg.TRAIN_FOLDER_NAME if split == 'train' else cfg.TEST_FOLDER_NAME)
    product_folders = (folder, cfg.PRODUCTS_FOLDER)
    imu_type, mounting, lever_arm_vehicle = read_rover_info(folder / 'README.xml')

    antenna_truth = read_truth(find_one_file([folder], ('ROVE*GroundTruth.txt', 'Rove*GroundTruth.txt'),
                                             'antenna truth'))
    imu_truth = read_truth(folder / f'{imu_type}_GroundTruth.txt')
    gnss = read_rinex_observations(find_one_file([folder], ('ROVE*.*[oO]',), 'RINEX observation'))
    imu_times, gyro, accel = read_imu(folder / f'{imu_type}.imr', gnss[0][0])

    # Fusion epochs: GNSS epochs inside the IMU and both truth spans.
    start = max(imu_times[0], antenna_truth[0, 0], imu_truth[0, 0])
    stop = min(imu_times[-1], antenna_truth[-1, 0], imu_truth[-1, 0])
    gnss = [(t, obs) for t, obs in gnss if start <= t <= stop and obs][:cfg.MAX_FUSION_EPOCHS]
    fusion_times = np.array([t for t, _ in gnss])
    keep = (imu_times >= fusion_times[0] - 1.0) & (imu_times <= fusion_times[-1] + 1.0)

    position, velocity, heading, pitch, roll = interpolate_truth(imu_truth, fusion_times)
    attitude = np.stack([truth_attitude_matrix(*row, mounting) for row in zip(position, heading, pitch, roll)])
    antenna_position = interpolate_truth(antenna_truth, fusion_times)[0]
    alpha, beta = read_klobuchar(find_one_file(product_folders, ('[bB][rR][dD][mM]*',), 'broadcast navigation'))

    return NavigationData(
        name=folder.name, fusion_times=fusion_times, gnss_observations=[obs for _, obs in gnss],
        imu_times=imu_times[keep], gyro=gyro[keep], accel=accel[keep],
        truth_position=position, truth_velocity=velocity, truth_attitude=attitude,
        truth_antenna_position=antenna_position, lever_arm=vehicle_to_body(mounting) @ lever_arm_vehicle,
        orbits=read_sp3(find_one_file(product_folders, ('*.[sS][pP]3',), 'SP3 orbit')),
        clocks=read_clock(find_one_file(product_folders, ('*.[cC][lL][kK]',), 'RINEX clock')),
        klobuchar_alpha=alpha, klobuchar_beta=beta)
