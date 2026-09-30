"""Fault detection, identification and adaptation (paper Sec. II-D, Ref. [33]) and integrity.

The test runs on the innovation predicted by the INS, before the network gain
is applied (paper Sec. II-D). All vectors are receiver-clock free, so a
pseudorange fault e_i appears in the innovation as c_i = projector @ e_i.
"""
import numpy as np
from scipy.stats import chi2

import settings as cfg
from earth_models import ecef_to_llh, ecef_to_ned_matrix


def detect_fault(innovation, H, R, P_prior):
    """Paper Eq. (33): T = nu^T Q^+ nu with Q = H P H^T + R; fault if T > chi2(1 - alpha, rank Q)."""
    Q = H @ P_prior @ H.T + R
    Q_inverse = np.linalg.pinv(Q, hermitian=True)
    statistic = innovation @ Q_inverse @ innovation
    threshold = chi2.ppf(1.0 - cfg.FALSE_ALARM_PROBABILITY, np.linalg.matrix_rank(Q, hermitian=True))
    return statistic > threshold, Q, Q_inverse


def identify_fault(innovation, projector, Q_inverse):
    """Ref. [33] Eq. (6)-(7): the pseudorange with the largest T_i = (c_i^T Q^+ nu)^2 / (c_i^T Q^+ c_i)."""
    c = projector
    return int(np.argmax((c.T @ Q_inverse @ innovation) ** 2 / np.einsum('ji,jk,ki->i', c, Q_inverse, c)))


def adapt_to_fault(dx, P_posterior, K, innovation, projector, index, Q, Q_inverse):
    """Paper Eq. (34): x_i = x_0 - L_i nu, P_i = P_0 + L_i Q L_i^T, L_i = K c_i c_i^+ (Ref. [33] Eq. (39))."""
    c = projector[:, [index]]
    c_plus = (c.T @ Q_inverse) / (c.T @ Q_inverse @ c)
    L = K @ c @ c_plus
    return dx - L @ innovation, P_posterior + L @ Q @ L.T


def protection_levels(P, position):
    """Horizontal and vertical protection levels K * sigma from the position covariance (A17)."""
    latitude, longitude, _ = ecef_to_llh(position)
    C = ecef_to_ned_matrix(latitude, longitude)
    P_ned = C @ P[0:3, 0:3] @ C.T
    half_sum, half_difference = 0.5 * (P_ned[0, 0] + P_ned[1, 1]), 0.5 * (P_ned[0, 0] - P_ned[1, 1])
    horizontal_major = half_sum + np.hypot(half_difference, P_ned[0, 1])
    return cfg.HORIZONTAL_PL_FACTOR * np.sqrt(horizontal_major), cfg.VERTICAL_PL_FACTOR * np.sqrt(P_ned[2, 2])


def stanford_percentages(error, protection_level):
    """Share of epochs in each region of the Stanford diagram (paper Fig. 20)."""
    al = cfg.ALERT_LIMIT
    regions = {
        'NO': (error <= protection_level) & (protection_level <= al),      # nominal operation
        'MI': (protection_level < error) & (error <= al),                  # misleading information
        'HO': (protection_level <= al) & (al < error),                     # hazardous operation
        'SU': (al < protection_level) & (error <= protection_level),       # system unavailable
        'SU&MI': (al < protection_level) & (protection_level < error),
    }
    return {name: 100.0 * float(np.mean(region)) for name, region in regions.items()}
