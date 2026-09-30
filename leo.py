"""LEO pseudorange simulation (Yan Eq. (1)) with TLE-differencing ephemeris error,
and the offline cache used by training/test.

Model: rho = |r_truth(t_tx) - r_rx| + eps_MP, eps_MP ~ N(0, sigma_mp(elev)^2).
The filter receives the *estimator* (older, published) TLE orbit, so the
ephemeris error enters the innovation like a broadcast-orbit error.  Satellite
clock, ionosphere and troposphere are taken as exactly known (zero); the
receiver clock is removed by the 'L' clock group of the measurement model.
Variance (Yan Eq. (3)): sigma_URA(family)^2 + sigma_mp^2, sigma_URA from the
training cache only.

Run `python leo.py` once to (re)build both caches.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import timedelta, timezone
from pathlib import Path
from typing import NamedTuple
import hashlib
import json
import math
import os
import tempfile

import numpy as np
from astropy import units as u
from astropy.coordinates import CartesianDifferential, CartesianRepresentation, ITRS, TEME
from astropy.time import Time
from astropy.utils import iers
from sgp4.api import SGP4_ERRORS, Satrec, WGS72
from sgp4.io import verify_checksum

import config as cfg
from navigation_models import (C_LIGHT, GM, GPS_EPOCH, GPS_UTC_LEAP_S, PseudoObs, ecef_llh, ecef_to_ned,
                               elev_az, range_los)
from simulation_data import Dataset, load_dataset

# MP/NLOS sigma of LEO: base + amplitude * exp(-elev_deg / scale).
MP_BASE_M, MP_AMP_M, MP_SCALE_DEG = 0.13, 0.53, 10.0

# Cache identity (stored in and checked against every cache file).
CACHE_VERSION = 2
MODEL_NAME = 'YAN_EQ1_TLE_DIFFERENCING_EPHEMERIS_PLUS_MP_NOISE'
TRUTH_ORBIT_SOURCE = 'TLE_SGP4_WGS72_ASTROPY_ITRS_NEAREST_WINDOW_CENTRE'
ESTIMATOR_ORBIT_SOURCE = 'TLE_SGP4_WGS72_ASTROPY_ITRS_LATEST_PUBLISHED_BEFORE_WINDOW'
CORRECTION_POLICY = 'CLOCK_IONO_TROPO_EXACTLY_KNOWN_ZERO'
RX_CLOCK_POLICY = 'RECEIVER_CLOCK_PROJECTED_AS_CLOCK_GROUP_L'
URA_POLICY = 'EQ3_URA_FAMILY_RMS_OF_TRAINING_TLE_DIFFERENCING_RANGE_ERROR'
URA_DEFAULT_FAMILY = 'ALL'
DETAIL_FIELDS = ('family', 'ephemeris_error_rsw_m', 'ephemeris_range_error_m', 'mp_nlos_error_m',
                 'mp_sigma_m', 'truth_tle_epoch_gpst_s', 'estimator_tle_epoch_gpst_s')


def leo_mp_sigma(elev) -> float:
    return float(MP_BASE_M + MP_AMP_M * math.exp(-float(np.rad2deg(float(elev))) / MP_SCALE_DEG))


# --- TLE catalogue -----------------------------------------------------------
class TLERecord(NamedTuple):
    epoch: float                 # GPST [s]
    set_number: int
    satrec: object
    line1: str
    line2: str


class TLEPair(NamedTuple):
    sat_id: str
    name: str
    truth: TLERecord             # nearest the window centre
    estimator: TLERecord         # newest published before the window


def _sgp4_teme(satrec, t):
    """SGP4 at GPST t -> (r [km], v [km/s], utc Time) in TEME."""
    utc = Time(GPS_EPOCH + timedelta(seconds=float(t) - GPS_UTC_LEAP_S), scale='utc')
    err, r, v = satrec.sgp4(float(utc.jd1), float(utc.jd2))
    if err:
        raise ValueError(SGP4_ERRORS.get(err, f'SGP4 error {err}'))
    return np.asarray(r, dtype=float), np.asarray(v, dtype=float), utc


def _sma_m(satrec) -> float:
    n = float(satrec.no_kozai) / 60.0
    return float((GM / n ** 2) ** (1.0 / 3.0))


def rsw_error(err, r, v) -> np.ndarray:
    """Project an orbit error on radial / along-track / cross-track axes."""
    r, v, err = (np.asarray(x, dtype=float).reshape(3) for x in (r, v, err))
    radial = r / np.linalg.norm(r)
    cross = np.cross(r, v)
    cross = cross / np.linalg.norm(cross)
    return np.array([err @ radial, err @ np.cross(cross, radial), err @ cross])


class TLEProvider:
    """TLE files -> de-duplicated catalogue -> one truth/estimator pair per satellite."""

    def __init__(self, window_start, window_end):
        self.max_age_s = cfg.TLE_MAX_AGE_DAYS * 86400.0
        self.t0, self.t1 = float(window_start), float(window_end)
        if not self.t1 > self.t0:
            raise ValueError('TLE selection window must have positive length')
        by_id: dict[str, dict[float, TLERecord]] = defaultdict(dict)
        self.names: dict[str, str] = {}
        self.duplicates = self.conflicting_duplicates = 0
        for src in sorted(Path(cfg.TLE_DIR).iterdir()):
            if not src.is_file():
                continue
            lines = src.read_text(encoding='ascii', errors='replace').splitlines()
            found = False
            for i in range(len(lines) - 1):
                l1, l2 = lines[i].strip(), lines[i + 1].strip()
                if not (l1.startswith('1 ') and l2.startswith('2 ')):
                    continue
                try:
                    verify_checksum(l1, l2)
                except ValueError:
                    if cfg.TLE_ALLOW_NON_TLE_FILES:
                        continue
                    raise
                satrec = Satrec.twoline2rv(l1, l2, WGS72)
                sat_id = f'NORAD-{str(satrec.satnum_str).strip()}'
                prev_line = lines[i - 1].strip() if i > 0 else ''
                if prev_line and not prev_line.startswith(('1 ', '2 ')):
                    self.names.setdefault(sat_id, prev_line[2:].strip() if prev_line.startswith('0 ') else prev_line)
                epoch_utc = Time(satrec.jdsatepoch, satrec.jdsatepochF, format='jd', scale='utc')
                epoch = (epoch_utc.to_datetime(timezone=timezone.utc) - GPS_EPOCH).total_seconds() + GPS_UTC_LEAP_S
                try:
                    set_number = int(l1[64:68])
                except ValueError:
                    set_number = -1
                rec = TLERecord(float(epoch), set_number, satrec, l1, l2)
                key = round(float(epoch), 3)             # duplicate = same sat, same epoch (1 ms)
                old = by_id[sat_id].get(key)
                found = True
                if old is None:
                    by_id[sat_id][key] = rec
                    continue
                # Re-issued copies differ only in rev number/checksum; genuinely
                # different elements keep the higher element-set number.
                self.duplicates += 1
                if (old.line1[:63], old.line2[:63]) != (l1[:63], l2[:63]):
                    self.conflicting_duplicates += 1
                    if rec.set_number > old.set_number:
                        by_id[sat_id][key] = rec
            if not found and not cfg.TLE_ALLOW_NON_TLE_FILES:
                raise ValueError(f'No valid TLE in {src}')

        self.records = {s: tuple(sorted(r.values(), key=lambda x: x.epoch)) for s, r in by_id.items()}
        self.pairs: dict[str, TLEPair] = {}
        self.report: dict[str, str] = {}
        for sat_id in sorted(self.records, key=lambda s: int(s.split('-')[1])):
            if sat_id in set(cfg.LEO_EXCLUDED_SAT_IDS):
                self.report[sat_id] = 'excluded by configuration'
                continue
            pair, reason = self._select_pair(sat_id)
            self.report[sat_id] = reason
            if pair is not None:
                self.pairs[sat_id] = pair
        self.sat_ids = tuple(self.pairs)

    def _select_pair(self, sat_id):
        recs = self.records[sat_id]
        center = 0.5 * (self.t0 + self.t1)
        if len(recs) < 2:
            return None, 'fewer than two distinct TLEs'
        truth = min(recs, key=lambda r: abs(r.epoch - center))
        if abs(truth.epoch - center) > cfg.TLE_TRUTH_MAX_OFFSET_DAYS * 86400.0:
            return None, 'no truth TLE close to the data window'
        older = [r for r in recs if r.epoch <= self.t0 - cfg.TLE_PUBLICATION_LATENCY_S
                 and r.epoch <= truth.epoch - cfg.TLE_MIN_PAIR_SEP_S]
        if not older:
            return None, 'no published estimator TLE before the data window'
        est = max(older, key=lambda r: r.epoch)
        if self.t1 - est.epoch > self.max_age_s:
            return None, 'estimator TLE older than TLE_MAX_AGE_DAYS inside the window'
        # Maneuver screening on consecutive pairs overlapping [est, max(truth, t1)].
        span_end = max(truth.epoch, self.t1)
        for a, b in zip(recs[:-1], recs[1:]):
            if b.epoch <= est.epoch or a.epoch >= span_end or b.epoch - a.epoch > self.max_age_s:
                continue
            jump = abs(_sma_m(b.satrec) - _sma_m(a.satrec))
            if jump > cfg.TLE_MANEUVER_SMA_JUMP_M:
                return None, f'maneuver: mean semi-major-axis jump {jump:.0f} m'
            try:
                r_b, _, _ = _sgp4_teme(b.satrec, b.epoch)
                r_a, _, _ = _sgp4_teme(a.satrec, b.epoch)
            except ValueError as exc:
                return None, f'SGP4 failure during maneuver screening: {exc}'
            gap = float(np.linalg.norm(r_a - r_b) * 1000.0)
            if gap > cfg.TLE_MANEUVER_CONSISTENCY_M:
                return None, f'maneuver: consecutive-TLE discrepancy {gap / 1000.0:.1f} km'
        return TLEPair(sat_id, self.names.get(sat_id, ''), truth, est), 'ok'

    @staticmethod
    def _to_itrs(r_km, v_kms, utc):
        pos = CartesianRepresentation(np.stack(r_km, axis=1) * u.km)
        vel = CartesianDifferential(np.stack(v_kms, axis=1) * u.km / u.s)
        teme = TEME(pos.with_differentials(vel), obstime=utc)
        degraded = 'warn' if cfg.TLE_ALLOW_DEGRADED_EOP else 'error'
        with iers.conf.set_temp('auto_download', False), iers.conf.set_temp('iers_degraded_accuracy', degraded):
            itrs = teme.transform_to(ITRS(obstime=utc))
        xyz = np.asarray(itrs.cartesian.xyz.to_value(u.m)).reshape(3, -1)
        dxyz = np.asarray(itrs.cartesian.differentials['s'].d_xyz.to_value(u.m / u.s)).reshape(3, -1)
        return [(xyz[:, k].copy(), dxyz[:, k].copy()) for k in range(xyz.shape[1])]

    def truth_pos(self, t, sat_id) -> np.ndarray:
        r, v, utc = _sgp4_teme(self.pairs[sat_id].truth.satrec, t)
        return self._to_itrs([r], [v], utc)[0][0]

    def truth_and_estimator(self, t, sat_id):
        """ITRS truth and estimator positions + RSW orbit error (estimator - truth)."""
        pair = self.pairs[sat_id]
        if float(t) - pair.estimator.epoch > self.max_age_s:
            raise ValueError(f'TLE too old for {sat_id}')
        r_t, v_t, utc = _sgp4_teme(pair.truth.satrec, t)
        r_e, v_e, _ = _sgp4_teme(pair.estimator.satrec, t)
        (p_t, _), (p_e, _) = self._to_itrs([r_t, r_e], [v_t, v_e], utc)
        return p_t, p_e, rsw_error((r_e - r_t) * 1000.0, r_t * 1000.0, v_t * 1000.0)

    def family(self, sat_id) -> str:
        """Constellation family for sigma_URA (first name token)."""
        token = self.names.get(sat_id, '').upper().replace('-', ' ').split()
        return token[0] if token else 'UNKNOWN'


# --- Simulator ---------------------------------------------------------------
class LightTimeError(RuntimeError):
    pass


class LEOSimulator:
    def __init__(self, tle: TLEProvider, seed: int):
        self.tle = tle
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.min_elev = np.deg2rad(cfg.LEO_MIN_ELEV_DEG)
        self.guard = np.deg2rad(float(cfg.LEO_PREFILTER_GUARD_DEG))

    def _light_time(self, sat_id, t_rx, rx, pos0, transit):
        """Yan Eq. (5) transmit-time iteration on the truth orbit."""
        prev, delta, resid = pos0, float('inf'), float('inf')
        for _ in range(int(cfg.LEO_TX_MAX_ITER)):
            t_tx = t_rx - transit
            pos = np.asarray(self.tle.truth_pos(t_tx, sat_id), dtype=float).reshape(3)
            rho, _, _ = range_los(rx, pos, transit)
            nxt = float(rho / C_LIGHT)
            delta = float(np.linalg.norm(pos - prev))
            resid = float(abs(nxt - transit) * C_LIGHT)
            if delta < float(cfg.LEO_TX_EPS_M):
                return t_tx, transit
            prev, transit = pos, nxt
        raise LightTimeError(f'Transmit-time iteration did not converge for {sat_id}: '
                             f'iterations={cfg.LEO_TX_MAX_ITER}, last_satellite_position_delta_m={delta:.6g}, '
                             f'last_light_time_residual_m={resid:.6g}, last_transit_s={transit:.12g}')

    def _simulate_sat(self, t_rx, rx, sat_id, C_ned):
        pos0 = self.tle.truth_pos(t_rx, sat_id)
        rho0, los0, _ = range_los(rx, pos0, 0.0)
        if elev_az(C_ned, los0)[0] < self.min_elev - self.guard:
            return None
        t_tx, transit = self._light_time(sat_id, float(t_rx), rx, np.asarray(pos0, dtype=float).reshape(3),
                                         float(rho0 / C_LIGHT))
        truth_tx, est_tx, err_rsw = self.tle.truth_and_estimator(t_tx, sat_id)
        rho_true, los, _ = range_los(rx, truth_tx, transit)
        elev, _ = elev_az(C_ned, los)
        if elev < self.min_elev:
            return None
        # Receiver uses the estimator orbit at the same transmit time.
        rho_est, _, est_rx = range_los(rx, est_tx, transit)
        sigma = leo_mp_sigma(elev)
        mp = float(self.rng.normal(0.0, sigma))
        pair = self.tle.pairs[sat_id]
        obs = PseudoObs(sat_id, 'L', float(rho_true + mp), est_rx, 0.0, 0.0, 0.0, float(sigma ** 2))
        detail = {'family': self.tle.family(sat_id), 'ephemeris_error_rsw_m': np.asarray(err_rsw, dtype=float).copy(),
                  'ephemeris_range_error_m': float(rho_est - rho_true), 'mp_nlos_error_m': mp,
                  'truth_tle_epoch_gpst_s': float(pair.truth.epoch),
                  'estimator_tle_epoch_gpst_s': float(pair.estimator.epoch)}
        return obs, detail

    def simulate_epoch(self, t_rx, rx_pos):
        """All visible LEO satellites at one epoch -> [(PseudoObs, detail), ...]."""
        rx = np.asarray(rx_pos, dtype=float).reshape(3)
        lat, lon, _ = ecef_llh(rx_pos)
        C_ned = ecef_to_ned(lat, lon)
        out = []
        for sat_id in self.tle.sat_ids:
            try:
                res = self._simulate_sat(t_rx, rx, sat_id, C_ned)
            except (ValueError, LightTimeError):
                continue
            if res is not None:
                out.append(res)
        return out


# --- Cache -------------------------------------------------------------------
@dataclass(frozen=True)
class LEOCache:
    t_fusion: np.ndarray
    meas: tuple                  # tuple[tuple[PseudoObs, ...]] per fusion epoch
    elev: tuple                  # elevation [rad] per measurement, per epoch
    details: tuple               # simulation-side diagnostics per epoch (never used by the filter)
    ura: dict                    # sigma_URA [m] per family
    ura_source: str              # dataset the sigma_URA values came from


def ura_sigma(details) -> dict[str, float]:
    """Yan Eq. (3) sigma_URA per family: RMS of the LOS TLE-differencing range error."""
    fam = np.concatenate([np.asarray(d['family'], dtype=str) for d in details]) if details else np.empty(0, str)
    err = np.concatenate([np.asarray(d['ephemeris_range_error_m'], dtype=float) for d in details]) \
        if details else np.empty(0)
    if err.size == 0:
        raise ValueError('cannot derive sigma_URA from a cache without LEO measurements')
    out = {URA_DEFAULT_FAMILY: float(np.sqrt(np.mean(err ** 2)))}
    for f in sorted(set(fam.tolist())):
        out[str(f)] = float(np.sqrt(np.mean(err[fam == f] ** 2)))
    return out


def ura_for(family, ura) -> float:
    return float(ura.get(str(family), ura[URA_DEFAULT_FAMILY]))


def _elevation(rx, sat_pos) -> float:
    d = np.asarray(rx, dtype=float).reshape(3) - np.asarray(sat_pos, dtype=float).reshape(3)
    rho = float(np.linalg.norm(d))
    if not np.isfinite(rho) or rho <= 0.0:
        raise ValueError('invalid cached LEO geometry')
    lat, lon, _ = ecef_llh(rx)
    return float(elev_az(ecef_to_ned(lat, lon), d / rho)[0])


def build_cache(ds: Dataset, sim: LEOSimulator, ura=None, ura_source=None) -> LEOCache:
    """Simulate every fusion epoch at the antenna truth.  ura=None: derive sigma_URA here (training)."""
    t_fusion = np.asarray(ds.t_fusion, dtype=np.float64).reshape(-1)
    raw, elevs, details = [], [], []
    for t, rx in zip(t_fusion, np.asarray(ds.truth_ant_pos, dtype=np.float64)):
        res = sim.simulate_epoch(float(t), rx)
        obs = [o for o, _ in res]
        raw.append(obs)
        elevs.append(np.asarray([_elevation(rx, o.sat_pos) for o in obs], dtype=np.float64))
        d = [x for _, x in res]
        details.append({
            'family': np.asarray([x['family'] for x in d], dtype=str),
            'ephemeris_error_rsw_m': np.asarray([x['ephemeris_error_rsw_m'] for x in d], dtype=np.float64).reshape(-1, 3),
            'ephemeris_range_error_m': np.asarray([x['ephemeris_range_error_m'] for x in d], dtype=np.float64),
            'mp_nlos_error_m': np.asarray([x['mp_nlos_error_m'] for x in d], dtype=np.float64),
            'mp_sigma_m': np.sqrt(np.asarray([o.variance_m2 for o in obs], dtype=np.float64)),
            'truth_tle_epoch_gpst_s': np.asarray([x['truth_tle_epoch_gpst_s'] for x in d], dtype=np.float64),
            'estimator_tle_epoch_gpst_s': np.asarray([x['estimator_tle_epoch_gpst_s'] for x in d], dtype=np.float64),
        })
    if ura is None:
        ura, ura_source = ura_sigma(details), ds.name
    elif ura_source is None:
        raise ValueError('ura_source is required with given sigma_URA values')
    meas = tuple(
        tuple(o._replace(variance_m2=float(ura_for(f, ura) ** 2 + s ** 2))
              for o, f, s in zip(obs, d['family'], d['mp_sigma_m']))
        for obs, d in zip(raw, details))
    return LEOCache(t_fusion.copy(), meas, tuple(elevs), tuple(details),
                    {str(k): float(v) for k, v in ura.items()}, str(ura_source))


def _fingerprint(a) -> str:
    a = np.ascontiguousarray(a)
    h = hashlib.sha256()
    h.update(a.dtype.str.encode('ascii'))
    h.update(repr(a.shape).encode('ascii'))
    h.update(a.tobytes(order='C'))
    return h.hexdigest()


def _tle_fingerprint() -> str:
    root = Path(cfg.TLE_DIR)
    files = sorted(p for p in root.rglob('*') if p.is_file())
    if not files:
        raise FileNotFoundError(f'LEO TLE directory has no input files: {root}')
    h = hashlib.sha256()
    for p in files:
        rel = p.relative_to(root).as_posix().encode('utf-8')
        data = p.read_bytes()
        h.update(len(rel).to_bytes(8, 'little'))
        h.update(rel)
        h.update(len(data).to_bytes(8, 'little'))
        h.update(data)
    return h.hexdigest()


def _tle_config() -> str:
    return json.dumps({
        'max_tle_age_days': float(cfg.TLE_MAX_AGE_DAYS),
        'publication_latency_s': float(cfg.TLE_PUBLICATION_LATENCY_S),
        'truth_max_offset_days': float(cfg.TLE_TRUTH_MAX_OFFSET_DAYS),
        'min_pair_separation_s': float(cfg.TLE_MIN_PAIR_SEP_S),
        'maneuver_sma_jump_m': float(cfg.TLE_MANEUVER_SMA_JUMP_M),
        'maneuver_consistency_m': float(cfg.TLE_MANEUVER_CONSISTENCY_M),
        'excluded_sat_ids': sorted(cfg.LEO_EXCLUDED_SAT_IDS),
    }, sort_keys=True)


def _expected_metadata(ds: Dataset) -> dict:
    """Every scalar a cache must carry to be valid for the current code, config and data."""
    return {
        'format_version': CACHE_VERSION, 'model_name': MODEL_NAME, 'dataset_id': ds.name,
        'seed': int(cfg.LEO_SEED[ds.split]), 'minimum_elevation_deg': float(cfg.LEO_MIN_ELEV_DEG),
        'prefilter_guard_deg': float(cfg.LEO_PREFILTER_GUARD_DEG), 'tx_epsilon_position_m': float(cfg.LEO_TX_EPS_M),
        'tx_max_iterations': int(cfg.LEO_TX_MAX_ITER), 'truth_orbit_source': TRUTH_ORBIT_SOURCE,
        'estimator_orbit_source': ESTIMATOR_ORBIT_SOURCE, 'truth_and_estimator_orbit_same': False,
        'correction_policy': CORRECTION_POLICY, 'receiver_clock_policy': RX_CLOCK_POLICY,
        'ura_policy': URA_POLICY, 'tle_selection_config': _tle_config(),
        'fusion_time_fingerprint': _fingerprint(np.asarray(ds.t_fusion, dtype=np.float64).reshape(-1)),
        'truth_position_fingerprint': _fingerprint(np.asarray(ds.truth_ant_pos, dtype=np.float64)),
        'tle_fingerprint': _tle_fingerprint(),
    }


def save_cache(path, cache: LEOCache, ds: Dataset) -> None:
    flat = [o for epoch in cache.meas for o in epoch]
    offsets = np.concatenate(([0], np.cumsum([len(e) for e in cache.meas]))).astype(np.int64)
    arrays = {k: np.asarray(v) for k, v in _expected_metadata(ds).items()}
    arrays.update({
        'mp_sigma_coefficients': np.asarray([MP_BASE_M, MP_AMP_M, MP_SCALE_DEG], dtype=np.float64),
        'ura_family_names': np.asarray(sorted(cache.ura), dtype=str),
        'ura_sigma_m': np.asarray([cache.ura[k] for k in sorted(cache.ura)], dtype=np.float64),
        'ura_source_dataset_id': np.asarray(cache.ura_source),
        'fusion_time_gpst_s': np.asarray(cache.t_fusion, dtype=np.float64),
        'epoch_offsets': offsets,
        'sat_ids': np.asarray([o.sat_id for o in flat], dtype=str),
        'constellations': np.asarray([o.constellation for o in flat], dtype=str),
        'pseudorange_m': np.asarray([o.pseudorange_m for o in flat], dtype=np.float64),
        'satellite_position_reception_ecef_m': np.asarray([o.sat_pos for o in flat], dtype=np.float64).reshape(-1, 3),
        'variance_m2': np.asarray([o.variance_m2 for o in flat], dtype=np.float64),
        'elevation_rad': np.concatenate([np.asarray(e, dtype=np.float64).reshape(-1) for e in cache.elev]),
    })
    for name in DETAIL_FIELDS:
        parts = np.concatenate([np.asarray(d[name]) for d in cache.details])
        arrays['detail_' + name] = (parts.astype(str) if name == 'family' else
                                    parts.reshape(-1, 3) if name == 'ephemeris_error_rsw_m' else
                                    parts.astype(np.float64))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f'.{path.stem}-', suffix='.npz', dir=path.parent)
    os.close(fd)
    try:
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp, path)
    finally:
        if Path(tmp).exists():
            Path(tmp).unlink()


def load_cache(path, ds: Dataset) -> LEOCache:
    """Load a cache and reject it if code, config, TLEs or dataset changed since it was built."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'LEO cache is missing: {path}. Run leo.py first.')
    try:
        with np.load(path, allow_pickle=False) as z:
            a = {k: z[k] for k in z.files}
        cache = _check_and_unpack(a, ds)
    except (OSError, EOFError, ValueError, KeyError) as exc:
        raise ValueError(f'invalid LEO cache {path}: {exc}. Run leo.py to regenerate it.') from exc
    return cache


def _check_and_unpack(a: dict, ds: Dataset) -> LEOCache:
    def require(ok, msg):
        if not ok:
            raise ValueError(msg)

    for key, expected in _expected_metadata(ds).items():
        value = np.asarray(a[key])
        require(value.shape == (), f'LEO cache {key} must be a scalar')
        require(type(expected)(value.item()) == expected, f'LEO cache {key} is incompatible')
    require(np.array_equal(a['mp_sigma_coefficients'], [MP_BASE_M, MP_AMP_M, MP_SCALE_DEG]),
            'LEO cache MP/NLOS model coefficients are incompatible')

    t = np.asarray(a['fusion_time_gpst_s'], dtype=np.float64)
    off = np.asarray(a['epoch_offsets'])
    sat_ids, groups = np.asarray(a['sat_ids']), np.asarray(a['constellations'])
    pr = np.asarray(a['pseudorange_m'], dtype=np.float64)
    pos = np.asarray(a['satellite_position_reception_ecef_m'], dtype=np.float64)
    var = np.asarray(a['variance_m2'], dtype=np.float64)
    elev = np.asarray(a['elevation_rad'], dtype=np.float64)
    n = pr.size
    require(t.ndim == 1 and t.size > 0 and np.all(np.isfinite(t)) and np.all(np.diff(t) > 0.0),
            'LEO cache fusion times are invalid')
    require(off.ndim == 1 and off.size == t.size + 1 and off.dtype.kind in 'iu', 'LEO cache epoch offsets have the wrong shape')
    off = off.astype(np.int64)
    require(off[0] == 0 and off[-1] == n and np.all(np.diff(off) >= 0), 'LEO cache epoch offsets are invalid')
    require(all(x.ndim == 1 and x.size == n for x in (sat_ids, groups, var, elev)) and pos.shape == (n, 3),
            'LEO cache flat measurement arrays have inconsistent shapes')
    require(n == 0 or (np.all(np.char.str_len(sat_ids.astype(str)) > 0) and np.all(groups.astype(str) == 'L')),
            'LEO cache satellite identities are invalid')
    require(all(np.all(np.isfinite(x)) for x in (pr, pos, var, elev)) and np.all(var > 0.0),
            'LEO cache measurement arrays must be finite with positive variance')
    require(np.all(np.abs(elev) <= np.pi / 2.0 + 1e-12), 'LEO cache elevations are outside the physical range')

    names = np.asarray(a['ura_family_names']).astype(str)
    values = np.asarray(a['ura_sigma_m'], dtype=np.float64)
    require(names.ndim == 1 and names.shape == values.shape and np.all(np.isfinite(values)) and np.all(values >= 0.0),
            'LEO cache sigma_URA table is invalid')
    ura = {str(k): float(v) for k, v in zip(names, values)}
    require(URA_DEFAULT_FAMILY in ura, 'LEO cache sigma_URA table lacks the default family')
    fam = np.asarray(a['detail_family']).astype(str)
    mp = np.asarray(a['detail_mp_sigma_m'], dtype=np.float64)
    require(fam.shape == (n,) and mp.shape == (n,), 'LEO cache detail arrays have inconsistent shapes')
    # sigma_mp uses truth geometry, stored elevation the estimator orbit (< 0.1 deg apart).
    require(np.allclose(mp, [leo_mp_sigma(e) for e in elev], rtol=0.0, atol=1e-3),
            'LEO cache MP/NLOS sigma disagrees with elevation')
    require(np.allclose(var, [ura_for(f, ura) ** 2 + s ** 2 for f, s in zip(fam, mp)], rtol=1e-12, atol=1e-12),
            'LEO cache Eq. (3) variance disagrees with sigma_URA and sigma_mp')

    detail = {k: np.asarray(a['detail_' + k]) for k in DETAIL_FIELDS}
    spans = list(zip(off[:-1].tolist(), off[1:].tolist()))
    meas = tuple(tuple(PseudoObs(str(sat_ids[i]), str(groups[i]), float(pr[i]), pos[i].copy(), 0.0, 0.0, 0.0,
                                 float(var[i])) for i in range(s, e)) for s, e in spans)
    cache = LEOCache(t.copy(), meas, tuple(elev[s:e].copy() for s, e in spans),
                     tuple({k: v[s:e].copy() for k, v in detail.items()} for s, e in spans),
                     ura, str(np.asarray(a['ura_source_dataset_id']).item()))
    if not np.array_equal(cache.t_fusion, ds.t_fusion):
        raise ValueError('LEO cache fusion times do not match the current dataset')
    return cache


# --- Entry point -------------------------------------------------------------
def make_simulator(ds: Dataset) -> LEOSimulator:
    tle = TLEProvider(*ds.truth_span)       # full truth span: pairs do not depend on epoch limits
    print(f'[{ds.name}] TLE selection: {len(tle.sat_ids)} satellites, {tle.duplicates} duplicate TLEs removed '
          f'({tle.conflicting_duplicates} with different elements)')
    for sat_id, reason in tle.report.items():
        if reason != 'ok':
            print(f"  skipped {sat_id} {tle.names.get(sat_id, '')}: {reason}")
    return LEOSimulator(tle, cfg.LEO_SEED[ds.split])


def main() -> None:
    train = load_dataset('train')
    train_cache = build_cache(train, make_simulator(train))
    save_cache(cfg.TRAIN_LEO_CACHE, train_cache, train)
    print(f'sigma_URA (m) from training TLE differencing: {train_cache.ura}')

    test = load_dataset('test')
    test_cache = build_cache(test, make_simulator(test), train_cache.ura, train_cache.ura_source)
    save_cache(cfg.TEST_LEO_CACHE, test_cache, test)


if __name__ == '__main__':
    main()
