"""Physical models: time, frames, INS, GNSS pseudoranges, measurement model, KF.

State order of the 9-state error model: [dp(3), dv(3), dtheta(3)] in ECEF.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple
import math

import numpy as np
from scipy.interpolate import BarycentricInterpolator
from scipy.linalg import expm

# --- Constants ---------------------------------------------------------------
WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_B = WGS84_A * (1.0 - WGS84_F)
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)
OMEGA_E = 7.292115e-5                    # Earth rotation rate [rad/s]
GM = 3.986004418e14                      # [m^3/s^2]
J2 = 1.08262668e-3
C_LIGHT = 299792458.0

GPS_EPOCH = datetime(1980, 1, 6, tzinfo=timezone.utc)
GPS_WEEK_S = 604800.0
GPS_UTC_LEAP_S = 18.0

F_L1 = 1575420000.0                      # GPS L1 / BDS B1C [Hz]
F_B1I = 1561098000.0                     # BDS B1I [Hz]

N_STATE = 9
ATT_SIGN = -1.0                          # C+ = Exp(ATT_SIGN * dtheta) C
OMEGA_IE = np.array([0.0, 0.0, OMEGA_E])
OMEGA_IE_SKEW = np.array([[0.0, -OMEGA_E, 0.0], [OMEGA_E, 0.0, 0.0], [0.0, 0.0, 0.0]])
C_F_V = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])  # vehicle -> IE frame

# GNSS noise model (Yan Eq. (3)).
URA_SIGMA_M = 1.5
GOGPS_A, GOGPS_S0, GOGPS_S1, GOGPS_A_DB = 30.0, 10.0, 50.0, 20.0     # MP/NLOS vs C/N0
DLL_BL_HZ, DLL_SPACING_CHIP, DLL_T_S, CHIP_RATE_HZ = 2.0, 0.1, 0.02, 1.023e6
DEFAULT_CN0_DBHZ = 45.0
SP3_INTERP_POINTS = 9

# Van Loan discretization: Taylor series below this 1-norm, expm above.
VAN_LOAN_TAYLOR_ORDER = 10
VAN_LOAN_TAYLOR_MAX_NORM = 0.20

# Receiver-clock groups: one clock / inter-system-bias mode each.  Yan Eq. (1)
# carries c*dt_u in the LEO pseudorange too, so LEO is its own group.
CLOCK_GROUPS = ('G', 'C', 'L')


# --- Time --------------------------------------------------------------------
def to_gpst(year, month, day, hour, minute, second, time_system='GPS') -> float:
    """Calendar epoch in the given time system -> seconds of GPST."""
    sec_int = int(math.floor(second))
    dt = datetime(year, month, day, hour, minute, tzinfo=timezone.utc)
    base = (dt - GPS_EPOCH).total_seconds() + sec_int + (second - sec_int)
    system = time_system.upper()
    if system in {'GPS', 'GPST', 'GAL', 'GST', 'QZS', 'QZSST', 'IRN'}:
        return float(base)
    if system in {'BDT', 'BDS'}:
        return float(base + 14.0)
    if system in {'UTC', 'GLO'}:
        return float(base + GPS_UTC_LEAP_S)
    raise ValueError(f'Unsupported time system: {time_system}')


# --- Rotations and frames ----------------------------------------------------
def skew(v) -> np.ndarray:
    x, y, z = v
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def so3_exp(rotvec) -> np.ndarray:
    v = np.asarray(rotvec, dtype=float).reshape(3)
    theta2 = float(v @ v)
    K = skew(v)
    if theta2 < 1e-12:
        a = 1.0 - theta2 / 6.0 + theta2 * theta2 / 120.0
        b = 0.5 - theta2 / 24.0 + theta2 * theta2 / 720.0
    else:
        theta = math.sqrt(theta2)
        a = math.sin(theta) / theta
        b = (1.0 - math.cos(theta)) / theta2
    return np.eye(3) + a * K + b * (K @ K)


def so3_log(R) -> np.ndarray:
    R = np.asarray(R, dtype=float).reshape(3, 3)
    vee = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    sin_t = 0.5 * float(np.linalg.norm(vee))
    theta = math.atan2(sin_t, 0.5 * (float(np.trace(R)) - 1.0))
    factor = 0.5 + theta * theta / 12.0 if sin_t < 1e-7 else theta / (2.0 * sin_t)
    return factor * vee


def project_rotation(M) -> np.ndarray:
    """Nearest proper rotation matrix (SVD)."""
    U, _, Vt = np.linalg.svd(np.asarray(M, dtype=float).reshape(3, 3))
    R = U @ Vt
    if np.linalg.det(R) < 0.0:
        U[:, -1] *= -1.0
        R = U @ Vt
    return R


def ecef_llh(pos) -> tuple[float, float, float]:
    """ECEF -> (lat [rad], lon [rad], h [m]) on WGS84."""
    x, y, z = np.asarray(pos, dtype=float).reshape(3)
    lon = float(np.arctan2(y, x))
    p = float(np.hypot(x, y))
    if p < 1e-08:
        lat = np.pi / 2.0 if z >= 0.0 else -np.pi / 2.0
        return (float(lat), lon, float(abs(z) - WGS84_B))
    lat = float(np.arctan2(z, p * (1.0 - WGS84_E2)))
    for _ in range(15):
        N = WGS84_A / np.sqrt(1.0 - WGS84_E2 * np.sin(lat) * np.sin(lat))
        h = p / np.cos(lat) - N
        new_lat = float(np.arctan2(z, p * (1.0 - WGS84_E2 * N / (N + h))))
        converged = abs(new_lat - lat) < 1e-13
        lat = new_lat
        if converged:
            break
    N = WGS84_A / np.sqrt(1.0 - WGS84_E2 * np.sin(lat) * np.sin(lat))
    return (float(lat), lon, float(p / np.cos(lat) - N))


def ecef_to_ned(lat, lon) -> np.ndarray:
    """Rotation matrix C_ned<-ecef."""
    slat, clat, slon, clon = np.sin(lat), np.cos(lat), np.sin(lon), np.cos(lon)
    return np.array([[-slat * clon, -slat * slon, clat],
                     [-slon, clon, 0.0],
                     [-clat * clon, -clat * slon, -slat]])


def vehicle_to_body(x_deg, y_deg, z_deg) -> np.ndarray:
    """SmartPNT SINS_RotAngle_IMU -> C_body<-vehicle."""
    gamma, beta, alpha = np.deg2rad([x_deg, y_deg, z_deg])
    cb, sb, cg, sg, ca, sa = np.cos(beta), np.sin(beta), np.cos(gamma), np.sin(gamma), np.cos(alpha), np.sin(alpha)
    Ry = np.array([[cb, 0.0, -sb], [0.0, 1.0, 0.0], [sb, 0.0, cb]])
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cg, sg], [0.0, -sg, cg]])
    Rz = np.array([[ca, sa, 0.0], [-sa, ca, 0.0], [0.0, 0.0, 1.0]])
    return Ry @ Rx @ Rz


def body_to_ecef(pos, heading_deg, pitch_deg, roll_deg, mounting_deg) -> np.ndarray:
    """Truth attitude (Inertial Explorer) -> C_ecef<-body."""
    lat, lon, _ = ecef_llh(pos)
    C_e_n = ecef_to_ned(lat, lon).T
    h, p, r = np.deg2rad([heading_deg, pitch_deg, roll_deg])
    ch, sh, cp, sp, cr, sr = np.cos(h), np.sin(h), np.cos(p), np.sin(p), np.cos(r), np.sin(r)
    C_n_f = np.array([[cp * ch, sr * sp * ch - cr * sh, cr * sp * ch + sr * sh],
                      [cp * sh, sr * sp * sh + cr * ch, cr * sp * sh - sr * ch],
                      [-sp, sr * cp, cr * cp]])
    C_b_v = vehicle_to_body(*mounting_deg)
    return project_rotation(C_e_n @ (C_n_f @ C_F_V) @ C_b_v.T)


def range_los(rx_pos, sat_pos_tx, transit_s) -> tuple[float, np.ndarray, np.ndarray]:
    """Sagnac-rotated range. Returns (rho, unit LOS sat->rx, sat pos in rx frame)."""
    angle = OMEGA_E * float(transit_s)
    c, s = np.cos(angle), np.sin(angle)
    rot = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])
    sat_rx = rot @ np.asarray(sat_pos_tx, dtype=float).reshape(3)
    los = np.asarray(rx_pos, dtype=float).reshape(3) - sat_rx
    rho = float(np.linalg.norm(los))
    return rho, los / rho, sat_rx


def elev_az(C_ned_ecef, los) -> tuple[float, float]:
    """Elevation/azimuth from unit LOS (sat -> receiver)."""
    ned = np.asarray(C_ned_ecef, dtype=float) @ -np.asarray(los, dtype=float)
    elev = float(np.arcsin(np.clip(-ned[2], -1.0, 1.0)))
    az = float(np.arctan2(ned[1], ned[0]) % (2.0 * np.pi))
    return elev, az


# --- Atmosphere and noise ----------------------------------------------------
def tropo_delay(height_m, elev) -> float:
    """Saastamoinen with standard atmosphere (Kelvin-form water vapour)."""
    if elev <= 0.0:
        return float('inf')
    h = max(-100.0, min(float(height_m), 10000.0))
    T = 15.0 - 0.0065 * h + 273.15
    p = 1013.25 * (1.0 - 2.2557e-05 * h) ** 5.2568
    e = 6.108 * 0.7 * math.exp((17.15 * T - 4684.0) / (T - 38.45))
    z = math.pi / 2.0 - elev
    return 0.002277 / math.cos(z) * (p + (1255.0 / T + 0.05) * e - 1.16 * math.tan(z) ** 2)


def tropo_sigma(elev) -> float:
    return 1.001 * 0.12 / math.sqrt(0.002001 + math.sin(float(elev)) ** 2)


def iono_delay(tow, lat, lon, elev, az, alpha, beta) -> float:
    """Klobuchar L1 delay [m]."""
    alpha = np.asarray(alpha, dtype=float).reshape(4)
    beta = np.asarray(beta, dtype=float).reshape(4)
    el_sc = elev / math.pi
    psi = 0.0137 / (el_sc + 0.11) - 0.022
    phi_i = np.clip(lat / math.pi + psi * math.cos(az), -0.416, 0.416)
    lam_i = lon / math.pi + psi * math.sin(az) / math.cos(phi_i * math.pi)
    phi_m = phi_i + 0.064 * math.cos((lam_i - 1.617) * math.pi)
    local_time = (43200.0 * lam_i + tow) % 86400.0
    basis = np.array([1.0, phi_m, phi_m ** 2, phi_m ** 3])
    amp = max(0.0, float(alpha @ basis))
    period = max(72000.0, float(beta @ basis))
    x = 2.0 * math.pi * (local_time - 50400.0) / period
    F = 1.0 + 16.0 * (0.53 - el_sc) ** 3
    delay_s = F * (5e-09 + amp * (1.0 - x ** 2 / 2.0 + x ** 4 / 24.0)) if abs(x) < 1.57 else F * 5e-09
    return C_LIGHT * delay_s


def iono_sigma(elev, lat) -> float:
    """Residual Klobuchar error: vertical sigma by latitude band x obliquity."""
    lat_deg = abs(float(np.rad2deg(lat)))
    sigma_v = 9.0 if lat_deg <= 20.0 else (4.5 if lat_deg <= 55.0 else 6.0)
    R, h_i = 6_378_140.0, 350_000.0
    denom = 1.0 - (R * math.cos(float(elev)) / (R + h_i)) ** 2
    return float(sigma_v / math.sqrt(max(denom, 1e-15)))


def mp_nlos_sigma(elev, cn0) -> float:
    """goGPS C/N0-elevation multipath/NLOS model."""
    e = max(float(elev), 1e-06)
    cn0 = float(cn0)
    if cn0 >= GOGPS_S1:
        var = 1.0
    else:
        ratio = (cn0 - GOGPS_S1) / (GOGPS_S0 - GOGPS_S1)
        term = 10.0 ** (-(cn0 - GOGPS_S1) / GOGPS_A_DB) * (
            (GOGPS_A / 10.0 ** (-(GOGPS_S0 - GOGPS_S1) / GOGPS_A_DB) - 1.0) * ratio + 1.0)
        var = 1.0 / max(math.sin(e) ** 2, 1e-06) * term
    return float(math.sqrt(max(var, 1e-12)))


def dll_sigma(cn0) -> float:
    """Code-tracking (DLL) thermal noise [m]."""
    cn0_lin = 10.0 ** (float(cn0) / 10.0)
    d = DLL_SPACING_CHIP
    chips = math.sqrt(DLL_BL_HZ * d / (2.0 * cn0_lin) * (1.0 + 2.0 / ((2.0 - d) * cn0_lin * DLL_T_S)))
    return float(C_LIGHT / CHIP_RATE_HZ * chips)


# --- INS ---------------------------------------------------------------------
@dataclass
class NavState:
    pos: np.ndarray        # IMU position, ECEF [m]
    vel: np.ndarray        # ECEF [m/s]
    dcm: np.ndarray        # C_ecef<-body
    ba: np.ndarray         # fixed accel bias (not an error state)
    bg: np.ndarray         # fixed gyro bias (not an error state)

    def copy(self) -> 'NavState':
        return NavState(*(np.asarray(v, dtype=float).copy() for v in (self.pos, self.vel, self.dcm, self.ba, self.bg)))

    def antenna(self, lever_b) -> np.ndarray:
        return self.pos + self.dcm @ np.asarray(lever_b).reshape(3)


def state_diff(later: NavState, earlier) -> np.ndarray:
    """9-state difference 'later - earlier' (earlier: NavState or (pos, vel, dcm))."""
    e_pos, e_vel, e_dcm = (earlier.pos, earlier.vel, earlier.dcm) if isinstance(earlier, NavState) else earlier
    return np.concatenate((
        np.asarray(later.pos, dtype=float).reshape(3) - np.asarray(e_pos, dtype=float).reshape(3),
        np.asarray(later.vel, dtype=float).reshape(3) - np.asarray(e_vel, dtype=float).reshape(3),
        so3_log(np.asarray(later.dcm, dtype=float).reshape(3, 3)
                @ np.asarray(e_dcm, dtype=float).reshape(3, 3).T) / ATT_SIGN,
    ))


def gravity_j2(pos) -> np.ndarray:
    x, y, z = np.asarray(pos, dtype=float).reshape(3)
    r = float(np.linalg.norm([x, y, z]))
    z2_r2 = z * z / (r * r)
    j2 = 1.5 * J2 * (WGS84_A / r) ** 2
    xy = 1.0 - j2 * (5.0 * z2_r2 - 1.0)
    zf = 1.0 - j2 * (5.0 * z2_r2 - 3.0)
    return -GM / r ** 3 * np.array([x * xy, y * xy, z * zf])


@lru_cache(maxsize=128)
def _earth_rotation(dt) -> np.ndarray:
    m = so3_exp(-OMEGA_IE * float(dt))
    m.setflags(write=False)
    return m


def mechanize(nav: NavState, gyro, accel, dt) -> NavState:
    """ECEF strapdown step (Yan Eq. (8)); fixed biases are subtracted."""
    dt = float(dt)
    w = OMEGA_E
    f = np.asarray(accel, dtype=float).reshape(3) - nav.ba
    C0 = nav.dcm
    C1 = project_rotation(_earth_rotation(dt) @ C0 @ so3_exp((np.asarray(gyro, dtype=float).reshape(3) - nav.bg) * dt))
    r, v = np.asarray(nav.pos, dtype=float).reshape(3), nav.vel
    centripetal = np.array([-w * (w * r[0]), w * (-w * r[1]), 0.0])
    coriolis = np.array([-w * v[1], w * v[0], 0.0])
    acc = 0.5 * (C0 + C1) @ f + gravity_j2(r) - centripetal - 2.0 * coriolis
    vel = v + acc * dt
    pos = nav.pos + 0.5 * (v + vel) * dt
    return NavState(pos, vel, C1, nav.ba.copy(), nav.bg.copy())


def error_dynamics(nav: NavState, accel) -> np.ndarray:
    """Continuous-time 9-state error matrix F."""
    F = np.zeros((N_STATE, N_STATE))
    r = nav.pos
    radius = float(np.linalg.norm(r))
    F[0:3, 3:6] = np.eye(3)
    F[3:6, 0:3] = -(2.0 / radius) * np.outer(gravity_j2(r), r / radius)
    F[3:6, 3:6] = -2.0 * OMEGA_IE_SKEW
    F[3:6, 6:9] = skew(nav.dcm @ (np.asarray(accel, dtype=float).reshape(3) - nav.ba))
    F[6:9, 6:9] = -OMEGA_IE_SKEW
    return F


def van_loan(F, Qc, dt) -> tuple[np.ndarray, np.ndarray]:
    """Discrete (Phi, Qd) from (F, Qc) over dt."""
    n = N_STATE
    A = np.zeros((2 * n, 2 * n))
    A[:n, :n], A[:n, n:], A[n:, n:] = F, Qc, -F.T
    B = A * float(dt)
    if float(np.linalg.norm(B, 1)) <= VAN_LOAN_TAYLOR_MAX_NORM:
        E = np.eye(2 * n)
        term = np.eye(2 * n)
        for k in range(1, VAN_LOAN_TAYLOR_ORDER + 1):
            term = (term @ B) / float(k)
            E += term
    else:
        E = expm(B)
    Phi = E[:n, :n]
    Qd = E[:n, n:] @ Phi.T
    return Phi, 0.5 * (Qd + Qd.T)


def propagate(nav: NavState, P, segments, gyro, accel, Qc):
    """INS + covariance over IMU segments [(imu_index, dt), ...]."""
    for i, dt in segments:
        nav = mechanize(nav, gyro[i], accel[i], dt)
        Phi, Qd = van_loan(error_dynamics(nav, accel[i]), Qc, dt)
        P = Phi @ P @ Phi.T + Qd
        P = 0.5 * (P + P.T)
    return nav, P


# --- GNSS pseudorange preprocessing ------------------------------------------
class PseudoObs(NamedTuple):
    sat_id: str
    constellation: str              # 'G', 'C' or 'L'
    pseudorange_m: float
    sat_pos: np.ndarray             # satellite ECEF at reception frame
    sat_clock_s: float
    iono_m: float
    tropo_m: float
    variance_m2: float


def _read_sp3(path):
    raw = defaultdict(lambda: [[], [], []])
    t, scale = None, 'GPS'
    with Path(path).open('r', encoding='ascii', errors='replace') as f:
        for line in f:
            if line.startswith('%c') and len(line) >= 12:
                cand = line[9:12].strip()
                if cand.upper() in {'GPS', 'GPST', 'GAL', 'GST', 'QZS', 'IRN', 'BDT', 'BDS', 'UTC', 'GLO'}:
                    scale = cand
            elif line.startswith('*'):
                v = line[1:].split()
                t = to_gpst(int(v[0]), int(v[1]), int(v[2]), int(v[3]), int(v[4]), float(v[5]), scale)
            elif t is not None and line.startswith('P'):
                v = line[4:].split()
                if len(v) < 4:
                    continue
                pos_km = np.asarray(v[:3], dtype=float)
                if np.any(np.abs(pos_km) >= 999999.0) or np.allclose(pos_km, 0.0):
                    continue
                clk_us = float(v[3])
                sat = raw[line[1:4].strip()]
                sat[0].append(t)
                sat[1].append(pos_km * 1000.0)
                sat[2].append(np.nan if abs(clk_us) >= 999999.0 else clk_us * 1e-06)
    orbits = {}
    for sat_id, (times, pos, clk) in raw.items():
        order = np.argsort(np.asarray(times, dtype=float))
        times = np.asarray(times, dtype=float)[order]
        clk = np.asarray(clk, dtype=float)[order]
        ok = np.isfinite(clk)
        orbits[sat_id] = {'time': times, 'pos': np.asarray(pos, dtype=float)[order],
                          'clk_time': times[ok], 'clk': clk[ok]}
    return orbits


def _read_clk(path):
    raw = defaultdict(lambda: [[], []])
    scale = 'GPS'
    with Path(path).open('r', encoding='ascii', errors='replace') as f:
        f.readline()
        for line in f:
            label = line[60:80].strip() if len(line) >= 60 else ''
            if label == 'END OF HEADER':
                break
            if label == 'TIME SYSTEM ID' and line[:10].split():
                scale = line[:10].split()[0]
        for line in f:
            if line[:2] != 'AS':
                continue
            v = line.split()
            if len(v) < 10:
                continue
            values = [float(x.replace('D', 'E')) for x in v[9:]]
            while len(values) < int(v[8]):
                values.extend(float(x.replace('D', 'E')) for x in f.readline().split())
            y, mo, d, h, mi = map(int, v[2:7])
            raw[v[1]][0].append(to_gpst(y, mo, d, h, mi, float(v[7]), scale))
            raw[v[1]][1].append(values[0])
    out = {}
    for sat_id, (times, bias) in raw.items():
        order = np.argsort(times)
        out[sat_id] = (np.asarray(times)[order], np.asarray(bias)[order])
    return out


class GNSSProcessor:
    """Precise orbit/clock + broadcast Klobuchar -> corrected GPS/BDS PseudoObs."""

    def __init__(self, sp3_path, clk_path, iono_coeffs, min_elev_deg):
        self.orbits = _read_sp3(sp3_path)
        self.clocks = _read_clk(clk_path)
        self.iono = iono_coeffs                     # {'G': (alpha, beta), 'C': (alpha, beta)}
        self.min_elev = np.deg2rad(min_elev_deg)

    def sat_pos(self, sat_id, t) -> np.ndarray:
        """Lagrange (barycentric) interpolation of SP3 positions."""
        s = self.orbits[sat_id]
        times = s['time']
        if t < times[0] or t > times[-1]:
            raise ValueError(f'SP3 time outside product span for {sat_id}')
        n = min(SP3_INTERP_POINTS, len(times))
        i0 = max(0, min(len(times) - n, int(np.searchsorted(times, t)) - n // 2))
        wt, wp = times[i0:i0 + n], s['pos'][i0:i0 + n]
        scale = max(float(np.max(np.abs(wt - t))), 1.0)
        x = (wt - t) / scale
        return np.asarray([float(BarycentricInterpolator(x, wp[:, k], rng=0)(0.0)) for k in range(wp.shape[1])])

    def prepare_epoch(self, epoch, nav: NavState, lever_b) -> tuple[PseudoObs, ...]:
        ant = nav.antenna(lever_b)
        lat, lon, height = ecef_llh(ant)
        C_ned = ecef_to_ned(lat, lon)
        t_rx = float(epoch.time_gpst_s)
        out = []
        for raw in epoch.measurements:
            # Light time: two fixed iterations.
            try:
                rho0, _, _ = range_los(ant, self.sat_pos(raw.sat_id, t_rx), 0.0)
                tau0 = float(rho0 / C_LIGHT)
                rho1, _, _ = range_los(ant, self.sat_pos(raw.sat_id, t_rx - tau0), tau0)
                transit = float(rho1 / C_LIGHT)
                t_tx = t_rx - transit
                pos_tx = self.sat_pos(raw.sat_id, t_tx)
            except (KeyError, ValueError):
                continue

            if raw.sat_id in self.clocks:
                ct, cb = self.clocks[raw.sat_id]
                if t_tx < ct[0] or t_tx > ct[-1]:
                    raise ValueError(f'CLK time outside product span for {raw.sat_id}')
                sat_clk = float(np.interp(t_tx, ct, cb))
            else:
                orb = self.orbits[raw.sat_id]
                if t_tx < orb['time'][0] or t_tx > orb['time'][-1]:
                    raise ValueError(f'SP3 time outside product span for {raw.sat_id}')
                sat_clk = float(np.interp(t_tx, orb['clk_time'], orb['clk']))

            _, los, pos_rx = range_los(ant, pos_tx, transit)
            elev, az = elev_az(C_ned, los)
            if elev < self.min_elev:
                continue
            tow = t_rx - int(math.floor(t_rx / GPS_WEEK_S)) * GPS_WEEK_S
            if raw.constellation == 'G':
                f_ref = F_L1
            else:
                if raw.signal_suffix not in {'2I', '2X', '1I', '1X'}:
                    continue
                tow = (tow - 14.0) % 604800.0          # BDT
                f_ref = F_B1I
            f_scale = (f_ref / raw.frequency_hz) ** 2
            alpha, beta = self.iono[raw.constellation]
            iono = iono_delay(tow, lat, lon, elev, az, alpha, beta) * f_scale
            tropo = tropo_delay(height, elev)
            cn0 = float(raw.cn0_dbhz) if raw.cn0_dbhz is not None else DEFAULT_CN0_DBHZ
            # Yan Eq. (3): URA + iono + tropo + MP/NLOS + receiver noise.
            variance = (URA_SIGMA_M ** 2 + (iono_sigma(elev, lat) * f_scale) ** 2 + tropo_sigma(elev) ** 2
                        + mp_nlos_sigma(elev, cn0) ** 2 + dll_sigma(cn0) ** 2)
            out.append(PseudoObs(raw.sat_id, raw.constellation, float(raw.pseudorange_m), pos_rx,
                                 sat_clk, float(iono), float(tropo), float(variance)))
        return tuple(out)


# --- Measurement set and model -----------------------------------------------
def clock_observable(meas) -> tuple:
    """Drop clock groups with a single member (clock-free residual unobservable)."""
    meas = tuple(meas)
    counts = Counter(m.constellation for m in meas if m.constellation in CLOCK_GROUPS)
    return tuple(m for m in meas if m.constellation not in CLOCK_GROUPS or counts[m.constellation] >= 2)


def clock_projector(meas) -> np.ndarray:
    """Weighted residual-maker that removes one clock mode per CLOCK_GROUPS group."""
    meas = tuple(meas)
    var = np.asarray([float(m.variance_m2) for m in meas], dtype=float)
    if np.any(~np.isfinite(var)) or np.any(var <= 0.0):
        raise ValueError('measurement variances must be finite and strictly positive')
    P = np.eye(len(meas))
    for group in CLOCK_GROUPS:
        idx = np.asarray([i for i, m in enumerate(meas) if m.constellation == group], dtype=int)
        if idx.size == 0:
            continue
        if idx.size < 2:
            raise ValueError(f'{group} receiver-clock projection requires at least two measurements')
        w = 1.0 / var[idx]
        P[np.ix_(idx, idx)] = np.eye(idx.size) - np.ones((idx.size, 1)) @ (w / np.sum(w))[None, :]
    return P


def select_measurements(meas, capacity: int) -> tuple:
    """Keep at most `capacity` rows: lowest variance first, clock groups as pairs."""
    meas = clock_observable(meas)
    if len(meas) <= capacity:
        return meas
    ranked = sorted(enumerate(meas), key=lambda it: (float(it[1].variance_m2), str(it[1].sat_id), int(it[0])))
    chosen: list[int] = []
    group_count: Counter = Counter()
    for rank, (i, m) in enumerate(ranked):
        if len(chosen) >= capacity:
            break
        if i in chosen:
            continue
        g = m.constellation
        if g in CLOCK_GROUPS and group_count[g] == 0:
            # A new clock group needs a partner to be observable.
            if capacity - len(chosen) < 2:
                continue
            partner = next((j for j, c in ranked[rank + 1:] if j not in chosen and c.constellation == g), None)
            if partner is None:
                continue
            chosen += [i, partner]
            group_count[g] += 2
            continue
        chosen.append(i)
        group_count[g] += 1
    return clock_observable(tuple(meas[i] for i in sorted(chosen[:capacity])))


def prepare_measurements(gnss: GNSSProcessor, gnss_epoch, leo_meas, nav: NavState, lever_b, *, capacity):
    """State-dependent measurement set of one epoch, in Yan Eq. (16) row order.

    Rows: by constellation, then descending elevation (sat_id breaks ties).
    """
    meas = clock_observable(tuple(gnss.prepare_epoch(gnss_epoch, nav, lever_b)) + tuple(leo_meas))
    ant = nav.antenna(lever_b)
    lat, lon, _ = ecef_llh(ant)
    C_ned = ecef_to_ned(lat, lon)

    def order(m):
        los = ant - np.asarray(m.sat_pos, dtype=float).reshape(3)
        elev, _ = elev_az(C_ned, los / np.linalg.norm(los))
        return (str(m.constellation), -round(float(elev), 9), str(m.sat_id))

    meas = tuple(sorted(meas, key=order))
    if capacity is not None and len(meas) > int(capacity):
        meas = select_measurements(meas, int(capacity))
    return tuple(meas)


class TCModel(NamedTuple):
    innovation: np.ndarray          # clock-projected y - h(x)
    H: np.ndarray                   # projected Jacobian (n, 9)
    R: np.ndarray                   # projected covariance
    sat_ids: tuple
    constellations: tuple
    projector: np.ndarray           # clock projector used above


def measurement_model(nav: NavState, meas, lever_b) -> TCModel:
    """Tightly coupled pseudorange model, clock modes projected out."""
    meas = clock_observable(meas)
    n = len(meas)
    if n == 0:
        return TCModel(np.empty(0), np.zeros((0, N_STATE)), np.zeros((0, 0)), (), (), np.zeros((0, 0)))
    lever_e = nav.dcm @ np.asarray(lever_b).reshape(3)
    ant = nav.pos + lever_e
    lever_skew = skew(lever_e)
    y = np.empty(n)
    H = np.zeros((n, N_STATE))
    for i, m in enumerate(meas):
        d = np.asarray(ant) - m.sat_pos
        rho = float(np.linalg.norm(d))
        los = d / rho
        predicted = rho + 0.0 - C_LIGHT * m.sat_clock_s + m.iono_m + m.tropo_m
        y[i] = m.pseudorange_m - predicted
        # dh/dp = u^T, dh/dv = 0, dh/dtheta = -s_theta u^T [C l]_x
        H[i, 0:3] = los
        H[i, 6:9] = -ATT_SIGN * (los @ lever_skew)
    Pc = clock_projector(meas)
    R = Pc @ np.diag([float(m.variance_m2) for m in meas]) @ Pc.T
    return TCModel(Pc @ y, Pc @ H, 0.5 * (R + R.T), tuple(m.sat_id for m in meas),
                   tuple(m.constellation for m in meas), Pc.copy())


# --- Kalman update and error-state reset -------------------------------------
def joseph_reset(P, K, H, R, dx) -> np.ndarray:
    """Joseph-form posterior covariance, then attitude reset Jacobian."""
    A = np.eye(N_STATE) - K @ H
    return reset_covariance(A @ P @ A.T + K @ R @ K.T, dx)


def kf_update(P, innovation, H, R):
    """Classical EKF update. Returns (dx, P_reset, K)."""
    PHt = P @ H.T
    S = H @ PHt + R
    K = PHt @ np.linalg.pinv(0.5 * (S + S.T), rcond=1e-12)
    if not np.all(np.isfinite(K)):
        raise FloatingPointError('non-finite 9-state classical Kalman gain')
    dx = K @ innovation
    return dx, joseph_reset(P, K, H, R, dx), K


def reset_covariance(P, dx) -> np.ndarray:
    """P <- G P G^T with the SO(3) left-Jacobian reset for dtheta."""
    phi = np.asarray(ATT_SIGN * np.asarray(dx, dtype=float).reshape(N_STATE)[6:9], dtype=float).reshape(3)
    t2 = float(phi @ phi)
    K = skew(phi)
    if t2 < 1e-12:
        a = 0.5 - t2 / 24.0 + t2 ** 2 / 720.0
        b = 1.0 / 6.0 - t2 / 120.0 + t2 ** 2 / 5040.0
    else:
        t = math.sqrt(t2)
        a = (1.0 - math.cos(t)) / t2
        b = (t - math.sin(t)) / (t2 * t)
    G = np.eye(N_STATE)
    G[6:9, 6:9] = np.eye(3) + a * K + b * (K @ K)
    out = G @ np.asarray(P, dtype=float) @ G.T
    out = 0.5 * (out + out.T)
    if not np.all(np.isfinite(out)):
        raise FloatingPointError('non-finite covariance after error-state reset')
    return out


def inject_error(nav: NavState, dx) -> NavState:
    dx = np.asarray(dx, dtype=float).reshape(N_STATE)
    return NavState(nav.pos + dx[0:3], nav.vel + dx[3:6],
                    project_rotation(so3_exp(ATT_SIGN * dx[6:9]) @ nav.dcm), nav.ba.copy(), nav.bg.copy())
