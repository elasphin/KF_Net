"""Read and synchronize one SmartPNT dataset (RINEX, IMR, truth, products)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple
import math
import re
import struct
import xml.etree.ElementTree as ET

import numpy as np

import config as cfg
from navigation_models import GPS_WEEK_S, GNSSProcessor, NavState, body_to_ecef, to_gpst, vehicle_to_body

# RINEX code preference per constellation and carrier frequency by band digit.
SIGNAL_PREFS = {
    'G': ('1C', '1W', '1P', '2W', '2L', '2X', '5Q', '5X', '5I'),
    'C': ('2I', '1I', '2X', '1X', '1P', '1D', '5X', '5P', '5D', '7I', '7X', '6I', '6X'),
}
FREQ_HZ = {('G', '1'): 1575420000.0, ('G', '2'): 1227600000.0, ('G', '5'): 1176450000.0,
           ('C', '1'): 1575420000.0, ('C', '2'): 1561098000.0, ('C', '5'): 1176450000.0,
           ('C', '7'): 1207140000.0, ('C', '8'): 1191795000.0, ('C', '6'): 1268520000.0}
IMU_MODEL_KEYS = ('ISDV_Pos', 'ISDV_Vel', 'ISDV_Att', 'PNSD_Pos', 'PNSD_Vel', 'PNSD_Att')
IMR_HEADER_BYTES = 512


# --- RINEX observations ------------------------------------------------------
class SatMeasurement(NamedTuple):
    sat_id: str
    constellation: str
    signal_suffix: str
    frequency_hz: float
    pseudorange_m: float
    cn0_dbhz: float | None = None


class ObsEpoch(NamedTuple):
    time_gpst_s: float
    measurements: tuple[SatMeasurement, ...]


def _rinex_float(field: str):
    raw = field.ljust(16)[:14].strip()
    if not raw:
        return None
    try:
        return float(raw.replace('D', 'E'))
    except ValueError:
        return float('nan')


class RINEXObs:
    """RINEX 3 observation file: one code pseudorange (+ C/N0) per satellite."""

    def __init__(self, path):
        self.path = Path(path)
        self.time_scale = 'GPS'
        types: dict[str, list[str]] = {}
        system, count = None, 0
        with self.path.open('r', encoding='ascii', errors='replace') as f:
            self.version = float(f.readline()[:9])
            for line in f:
                label = line[60:80].strip() if len(line) >= 60 else ''
                if label == 'END OF HEADER':
                    break
                if label == 'TIME OF FIRST OBS':
                    self.time_scale = line[48:51].strip() or 'GPS'
                elif label == 'SYS / # / OBS TYPES':
                    if line[0:1].strip():
                        system, count = line[0], int(line[3:6])
                        types[system] = []
                    if system:
                        types[system] = (types[system] + line[7:60].split())[:count]
        self.n_types = {s: len(v) for s, v in types.items()}
        self.type_index = {s: {t: i for i, t in enumerate(v)} for s, v in types.items()}

    def _decode(self, sat_id, fields) -> SatMeasurement | None:
        """First preferred signal with a valid pseudorange."""
        system = sat_id[0]
        index = self.type_index.get(system, {})
        for suffix in SIGNAL_PREFS.get(system, ()):
            i = index.get('C' + suffix)
            if i is None or i >= len(fields):
                continue
            pr = _rinex_float(fields[i])
            if pr is None or not math.isfinite(pr) or pr <= 0.0:
                continue
            if system == 'C' and self.version < 3.04 and suffix in {'1I', '1Q', '1X'}:
                freq = 1561098000.0                    # pre-3.04 B1I labelled as band 1
            else:
                freq = FREQ_HZ.get((system, suffix[0]))
            if freq is None:
                continue
            j = index.get('S' + suffix)
            cn0 = _rinex_float(fields[j]) if j is not None and j < len(fields) else None
            if cn0 is not None and not math.isfinite(cn0):
                cn0 = None
            return SatMeasurement(sat_id, system, suffix, float(freq), float(pr), cn0)
        return None

    def iter_epochs(self, systems=None, *, t_start=None, t_end=None, max_epochs=None, need_meas=False):
        yielded = 0
        with self.path.open('r', encoding='ascii', errors='replace') as f:
            for line in f:
                if len(line) >= 60 and line[60:80].strip() == 'END OF HEADER':
                    break
            for line in f:
                if not line.startswith('>'):
                    continue
                parts = line[1:].split()
                if len(parts) < 8:
                    continue
                flag, n_sat = int(parts[6]), int(parts[7])
                if flag not in (0, 1):
                    for _ in range(n_sat):
                        f.readline()
                    continue
                t = to_gpst(*map(int, parts[:5]), float(parts[5]), self.time_scale)
                if t_end is not None and t > float(t_end):
                    break
                meas = []
                for _ in range(n_sat):
                    sat_line = f.readline()
                    if not sat_line:
                        break
                    sat_id = sat_line[:3].strip()
                    if not sat_id:
                        continue
                    n_fields = self.n_types.get(sat_id[0], 0)
                    payload = sat_line[3:].rstrip('\n')
                    fields = [payload[i:i + 16] for i in range(0, len(payload), 16)]
                    while len(fields) < n_fields:
                        cont = f.readline()
                        if not cont:
                            break
                        payload = cont[3:].rstrip('\n')
                        fields.extend(payload[i:i + 16] for i in range(0, len(payload), 16))
                    if systems and sat_id[0] not in systems:
                        continue
                    m = self._decode(sat_id, fields[:n_fields])
                    if m is not None:
                        meas.append(m)
                if t_start is not None and t < float(t_start):
                    continue
                if need_meas and not meas:
                    continue
                yield ObsEpoch(t, tuple(meas))
                yielded += 1
                if max_epochs is not None and yielded >= int(max_epochs):
                    break


# --- Truth (Inertial Explorer) -----------------------------------------------
@dataclass
class Truth:
    time: np.ndarray            # GPST [s]
    pos: np.ndarray             # ECEF [m]
    vel: np.ndarray             # ECEF [m/s]
    heading: np.ndarray         # [deg]
    pitch: np.ndarray
    roll: np.ndarray


def load_truth(path) -> Truth:
    rows, started = [], False
    for line in Path(path).read_text(encoding='utf-8', errors='replace').splitlines():
        v = line.split()
        try:
            if len(v) < 24:
                raise ValueError
            row = (int(v[0]), float(v[1]), *map(float, v[9:12]), *map(float, v[15:18]), *map(float, v[21:24]))
        except (ValueError, IndexError):
            if started:
                break
            continue
        started = True
        rows.append(row)
    a = np.asarray(rows, dtype=float)
    return Truth(a[:, 0].astype(int).astype(float) * GPS_WEEK_S + a[:, 1], a[:, 2:5], a[:, 5:8], a[:, 8], a[:, 9], a[:, 10])


def interp_truth(truth: Truth, t_query, max_gap_s, *, position_only=False):
    """Linear interpolation (heading unwrapped). Returns pos or (pos, vel, hdg, pitch, roll)."""
    tt, q = truth.time, np.asarray(t_query, dtype=float)
    hi = np.clip(np.searchsorted(tt, q, side='left'), 0, len(tt) - 1)
    lo = np.maximum(hi - 1, 0)
    exact = tt[hi] == q
    lo[exact] = hi[exact]
    gap = tt[hi] - tt[lo]
    if np.any(gap > float(max_gap_s)) or np.any(q < tt[0]) or np.any(q > tt[-1]):
        raise ValueError('Invalid ground-truth interpolation span')
    w = np.zeros_like(q)
    nz = gap > 0.0
    w[nz] = (q[nz] - tt[lo[nz]]) / gap[nz]
    w0 = 1.0 - w
    pos = w0[:, None] * truth.pos[lo] + w[:, None] * truth.pos[hi]
    if position_only:
        return pos
    vel = w0[:, None] * truth.vel[lo] + w[:, None] * truth.vel[hi]
    hdg_u = np.unwrap(np.deg2rad(truth.heading))
    hdg = np.rad2deg(w0 * hdg_u[lo] + w * hdg_u[hi]) % 360.0
    pitch = w0 * truth.pitch[lo] + w * truth.pitch[hi]
    roll = w0 * truth.roll[lo] + w * truth.roll[hi]
    return pos, vel, hdg, pitch, roll


# --- Small file helpers ------------------------------------------------------
def check_time_axis(name, t) -> np.ndarray:
    t = np.asarray(t, dtype=float).reshape(-1)
    if t.size == 0 or not np.all(np.isfinite(t)) or (t.size > 1 and np.any(np.diff(t) <= 0.0)):
        raise ValueError(f'{name} time axis is empty, non-finite, or not strictly increasing')
    return t


def unique_file(directory: Path, patterns, label) -> Path:
    matches = sorted({p for pat in patterns for p in directory.glob(pat) if p.is_file()})
    if len(matches) != 1:
        names = ', '.join(p.name for p in matches) or 'none'
        raise FileNotFoundError(f'Expected exactly one {label} file in {directory}, found {len(matches)}: {names}')
    return matches[0]


def required_file(path: Path, label) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f'{label} is missing: {path}')
    return path


def anchor_imr_gpst(tow, t_anchor) -> np.ndarray:
    """IMR time-of-week -> continuous GPST, week chosen nearest t_anchor, rollovers handled."""
    tow = np.asarray(tow, dtype=float).reshape(-1)
    if tow.size == 0:
        return tow.copy()
    week = int(math.floor(float(t_anchor) / GPS_WEEK_S))
    cand = np.array([(week + k) * GPS_WEEK_S + tow[0] for k in (-1, 0, 1)])
    first = float(cand[np.argmin(np.abs(cand - float(t_anchor)))])
    if tow.size == 1:
        return np.asarray([first], dtype=float)
    d = np.diff(tow)
    step = np.zeros(d.size, dtype=np.int64)
    step[d < -0.5 * GPS_WEEK_S] = 1
    step[d > 0.5 * GPS_WEEK_S] = -1
    offset = np.empty(tow.size, dtype=np.int64)
    offset[0] = 0
    np.cumsum(step, out=offset[1:])
    t = (int(math.floor(first / GPS_WEEK_S)) + offset).astype(float) * GPS_WEEK_S + tow
    t[0] = first
    return t


def read_imr_header(path: Path) -> dict:
    """NovAtel Inertial Explorer .imr header -> scales and record layout."""
    with path.open('rb') as f:
        buf = f.read(IMR_HEADER_BYTES)
    endian = '<' if buf[8] == 0 else '>'
    v = struct.unpack(endian + '8scdiidddiid32s?BBB32s6h?iii354s', buf)
    rec_bytes = struct.calcsize(endian + 'd6i')
    return {'rate_hz': float(v[5]), 'gyro_scale': float(v[6]), 'accel_scale': float(v[7]),
            'time_bias_ms': float(v[10]), 'rec_bytes': rec_bytes,
            'n_rec': max(0, path.stat().st_size - IMR_HEADER_BYTES) // rec_bytes,
            'dtype': np.dtype([('tow', endian + 'f8'), ('counts', endian + 'i4', (6,))], align=False)}


def read_imu_model(text, imu_type) -> dict[str, np.ndarray]:
    for block in re.finditer(r'IMU\s*\{(.*?)\}', text, flags=re.S):
        block = block.group(1)
        name = re.search(r'IMU_Type\s*=\s*"([^"]+)"', block)
        if not name or name.group(1) != imu_type:
            continue
        values = {}
        for key in IMU_MODEL_KEYS:
            m = re.search(rf'{key}\s*=\s*([^\r\n]+)', block)
            if m:
                values[key] = np.fromstring(m.group(1), sep=' ', dtype=float)
        if len(values) == len(IMU_MODEL_KEYS):
            return values
    raise KeyError(f'No complete IMU noise model for IMU type {imu_type!r}')


def read_iono_coeffs(nav_path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Klobuchar alpha/beta for GPS and BDS from the broadcast NAV header."""
    header = {}
    with nav_path.open('r', encoding='ascii', errors='replace') as f:
        f.readline()
        for line in f:
            label = line[60:80].strip() if len(line) >= 60 else ''
            if label == 'END OF HEADER':
                break
            fields = line[:60].split()
            if label == 'IONOSPHERIC CORR' and len(fields) >= 5:
                header[fields[0]] = tuple(float(x.replace('D', 'E')) for x in fields[1:5])
    missing = [k for k in ('GPSA', 'GPSB', 'BDSA', 'BDSB') if k not in header]
    if missing:
        raise ValueError(f'{nav_path} is missing ionosphere coefficients: {missing}')
    return {'G': (np.asarray(header['GPSA']), np.asarray(header['GPSB'])),
            'C': (np.asarray(header['BDSA']), np.asarray(header['BDSB']))}


# --- Dataset -----------------------------------------------------------------
@dataclass
class Dataset:
    root: Path
    split: str                   # 'train' or 'test'
    gnss_epochs: tuple           # ObsEpoch per fusion epoch
    t_fusion: np.ndarray         # fusion epoch times (GPST)
    t_imu: np.ndarray            # IMU sample times
    gyro: np.ndarray             # body angular rate [rad/s]
    accel: np.ndarray            # body specific force [m/s^2]
    truth_ant_pos: np.ndarray    # antenna truth at fusion epochs
    truth_pos: np.ndarray        # IMU truth position at fusion epochs
    truth_vel: np.ndarray
    truth_dcm: np.ndarray
    truth_span: tuple            # (first, last) antenna-truth time (TLE window)
    nav0: NavState               # truth-initialized state at first IMU sample
    lever_b: np.ndarray          # IMU -> antenna lever arm, body frame
    P0: np.ndarray
    Qc: np.ndarray
    gnss: GNSSProcessor

    @property
    def name(self) -> str:
        return self.root.name


def load_dataset(split: str) -> Dataset:
    """Load and synchronize the 'train' (Data01) or 'test' (Data02) dataset."""
    root = cfg.TRAIN_DIR if split == 'train' else cfg.TEST_DIR
    max_epochs = cfg.TRAIN_FUSION_EPOCHS if split == 'train' else cfg.TEST_FUSION_EPOCHS
    label = 'training' if split == 'train' else 'test'

    # README: rover 01 IMU type, mounting angles and lever arm.
    readme = required_file(root / 'README.xml', 'README.xml')
    rover = next((r for r in ET.fromstring(readme.read_text(encoding='utf-8', errors='replace')).findall('ROVE')
                  if (r.findtext('ID') or '').strip() == '01'), None)
    if rover is None:
        raise ValueError(f'{readme} has no ROVE entry with ID 01')
    imu_type = (rover.findtext('SINS_IMUType') or '').strip()
    if not imu_type:
        raise ValueError(f'{readme} has no SINS_IMUType for ROVE 01')
    mounting = np.fromstring(rover.findtext('SINS_RotAngle_IMU') or '', sep=' ')
    if mounting.shape != (3,):
        raise ValueError(f'{readme} has an invalid SINS_RotAngle_IMU')
    lever_v = np.fromstring(rover.findtext('SINS_LeverArm_GNSS') or '', sep=' ')
    if lever_v.shape != (3,):
        raise ValueError(f'{readme} has an invalid SINS_LeverArm_GNSS')

    ant_truth = load_truth(unique_file(root, ('ROVE_GroundTruth.txt', 'ROVE_01_GroundTruth.txt',
                                              'Rove_01_GroundTruth.txt'), 'rover 01 ground truth'))
    imu_truth = load_truth(required_file(root / f'{imu_type}_GroundTruth.txt', f'{imu_type} ground truth'))
    rinex = RINEXObs(unique_file(root, ('ROVE.*O', 'ROVE.*o', 'ROVE_01.*O', 'ROVE_01.*o'),
                                 'rover 01 RINEX observation'))
    imr_path = required_file(root / f'{imu_type}.imr', f'{imu_type} IMR')
    sp3_path = unique_file(root, ('*.SP3', '*.sp3'), 'SP3 precise-orbit')
    clk_path = unique_file(root, ('*.CLK', '*.clk'), 'RINEX clock')
    nav_path = unique_file(root, ('brdm*.*p', 'brdm*.*P', 'brdm*.rnx', 'BRDM*.RNX'), 'broadcast navigation')
    imu_model = read_imu_model(required_file(cfg.IMU_MODEL_FILE, 'IMU error model')
                               .read_text(encoding='utf-8', errors='replace'), imu_type)

    first_obs = next(rinex.iter_epochs({'G', 'C'}), None)
    if first_obs is None:
        raise ValueError(f'{rinex.path} contains no usable GPS/BDS observation epoch')

    # IMU time axis (all records), anchored to the first RINEX epoch.
    imr = read_imr_header(imr_path)
    records = np.memmap(imr_path, dtype=imr['dtype'], mode='r', offset=IMR_HEADER_BYTES, shape=(imr['n_rec'],))
    tow = np.asarray(records['tow'], dtype=np.float64).copy()
    del records
    tow[tow > GPS_WEEK_S] -= GPS_WEEK_S
    tow -= imr['time_bias_ms'] * 1e-3
    t_imu_all = anchor_imr_gpst(tow, float(first_obs.time_gpst_s))

    # Common span of IMU and both truths; GNSS epochs inside it.
    t_ant = check_time_axis(f'{label} antenna truth', ant_truth.time)
    t_imu_truth = check_time_axis(f'{label} IMU truth', imu_truth.time)
    t_end = min(float(t_imu_all[-1]), float(t_ant[-1]), float(t_imu_truth[-1]))
    start = int(np.searchsorted(t_imu_all, max(float(t_imu_all[0]), float(t_ant[0]), float(t_imu_truth[0])),
                                side='left'))
    gnss_epochs = tuple(rinex.iter_epochs({'G', 'C'}, t_start=float(t_imu_all[start]), t_end=t_end,
                                          max_epochs=max_epochs, need_meas=True))
    t_fusion = check_time_axis(f'{label} fusion', np.asarray([e.time_gpst_s for e in gnss_epochs], dtype=float))
    if len(t_fusion) < 3:
        raise ValueError(f'Fewer than three synchronized {label} fusion epochs remain')

    # IMU samples up to the last fusion epoch.
    stop = min(int(np.searchsorted(t_imu_all, t_fusion[-1], side='left')) + 1,
               int(np.searchsorted(t_imu_all, t_end, side='left')) + 1, len(t_imu_all))
    rec = np.fromfile(imr_path, dtype=imr['dtype'], count=stop - start,
                      offset=IMR_HEADER_BYTES + start * imr['rec_bytes'])
    gyro = np.deg2rad(rec['counts'][:, :3].astype(float) * imr['gyro_scale'] * imr['rate_hz'])
    accel = rec['counts'][:, 3:6].astype(float) * imr['accel_scale'] * imr['rate_hz']
    t_imu = t_imu_all[start:stop].copy()

    # Truth at [first IMU sample, fusion epochs...].
    t_query = np.concatenate(([t_imu[0]], t_fusion))
    ant_pos = interp_truth(ant_truth, t_query, cfg.MAX_TRUTH_GAP_S, position_only=True)
    pos, vel, hdg, pitch, roll = interp_truth(imu_truth, t_query, cfg.MAX_TRUTH_GAP_S)
    dcm = [body_to_ecef(p, h, q, r, mounting) for p, h, q, r in zip(pos, hdg, pitch, roll)]

    deg = np.pi / 180.0
    P0 = np.diag(np.concatenate((imu_model['ISDV_Pos'], imu_model['ISDV_Vel'], imu_model['ISDV_Att'] * deg)) ** 2)
    Qc = np.diag(np.concatenate((imu_model['PNSD_Pos'], imu_model['PNSD_Vel'], imu_model['PNSD_Att'] * deg)) ** 2)

    return Dataset(
        root=root, split=split, gnss_epochs=gnss_epochs, t_fusion=t_fusion, t_imu=t_imu, gyro=gyro, accel=accel,
        truth_ant_pos=ant_pos[1:], truth_pos=pos[1:], truth_vel=vel[1:], truth_dcm=np.stack(dcm[1:]),
        truth_span=(float(t_ant[0]), float(t_ant[-1])),
        nav0=NavState(pos[0].copy(), vel[0].copy(), dcm[0], np.zeros(3), np.zeros(3)),
        lever_b=vehicle_to_body(*mounting) @ lever_v, P0=P0, Qc=Qc,
        gnss=GNSSProcessor(sp3_path, clk_path, read_iono_coeffs(nav_path), cfg.GNSS_MIN_ELEV_DEG),
    )
