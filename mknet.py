"""Masked CLA network of Yan et al. (2026), Fig. 8, its Eq. (15)-(17) input
builder, checkpoint I/O and the differentiable INS used during training.
"""
from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import torch
from torch import nn
import torch.nn.functional as F

import config as cfg
from navigation_models import ATT_SIGN, GM, J2, N_STATE, OMEGA_IE, WGS84_A

DTYPE = torch.float32

# Architecture (Fig. 8) and input layout (Eq. (15)-(17)).
FIXED_DIM = 6 + 2 * N_STATE      # [d_accel, d_gyro, state residual, state innovation] = 24
OBS_DIM = 2                      # per measurement: [lagged residual, innovation]
CONV_CH, KERNEL, POOL = 24, 3, 3
HIDDEN, LAYERS = 32, 2
MASK_EPS = 1e-6
# Fixed-feature groups (3 components each), in network_input order:
# d_accel, d_gyro, dx_resid[p, v, theta], dx_innov[p, v, theta].
FIXED_GROUPS = ('d_accel', 'd_gyro', 'resid_p', 'resid_v', 'resid_theta',
                'innov_p', 'innov_v', 'innov_theta')

# Checkpoint identity (must match to load).
STATE_ORDER = '[delta_p,delta_v,delta_theta]'
FEATURE_VERSION = 10             # 10: fixed / group RMS (train, classical EKF); dy, dy~ / sigma_j (Eq. (3))
# 9: LSTM holds its state at missing lagged residuals (Eq. (25))
ARCHITECTURE = 'Yan_masked_compact_CNN24_k3_LSTM2x32_seq_attention_FC_KG_9state'
# Output scale of the gain (config.GAIN_SCALE_MODE).  Checkpoints written before
# this field existed are read as 'empirical' (their K = diag(g) K_n semantics).
GAIN_SCALE_MODES = ('empirical', 'none', 'analytic', 'covariance')
# 'covariance' (not in Yan; documented deviation): the head outputs COV_OUT numbers,
# not a slot-indexed K.  K = P~ H^T (H P~ H^T + lambda R)^-1 with
# P~ = diag(s) P^- diag(s),  s_i = 10^(COV_LOG10_RANGE_P tanh(a_i)),
# lambda = 10^(COV_LOG10_RANGE_R tanh(b)).  Geometry enters through H, so the gain
# does not depend on which slot a satellite occupies.  Zero head output = classical EKF.
COV_OUT = N_STATE + 1
COV_LOG10_RANGE_P = 1.0          # prior std scale in [0.1, 10]
COV_LOG10_RANGE_R = 2.0          # measurement variance scale in [0.01, 100]


def gain_head_out(mode: str, nmax: int) -> int:
    return COV_OUT if mode == 'covariance' else N_STATE * int(nmax)


def pad_measurement_model(H, R, nmax: int):
    """Pad H (n, 9) with zero rows and R (n, n) with an identity block to nmax (float64 torch)."""
    H = torch.as_tensor(H, dtype=torch.float64)
    R = torch.as_tensor(R, dtype=torch.float64, device=H.device)
    n, nmax = int(H.shape[0]), int(nmax)
    if n > nmax:
        raise ValueError(f'{n} measurement rows exceed nmax={nmax}')
    Hp = H.new_zeros((nmax, N_STATE))
    Hp[:n] = H
    Rp = torch.eye(nmax, dtype=torch.float64, device=H.device)
    Rp[:n, :n] = R
    return Hp, Rp


# --- Differentiable INS (torch twin of navigation_models) --------------------
class TorchNav(NamedTuple):
    pos: torch.Tensor
    vel: torch.Tensor
    dcm: torch.Tensor
    ba: torch.Tensor
    bg: torch.Tensor


def so3_exp(v: torch.Tensor) -> torch.Tensor:
    v = v.reshape(3)
    t2 = torch.dot(v, v)
    t2_safe = t2.clamp_min(1e-30)
    t = torch.sqrt(t2_safe)
    x, y, z = v.unbind()
    zero = v.new_zeros(())
    K = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero)).reshape(3, 3)
    small = t2 < 1e-10
    a = torch.where(small, 1.0 - t2 / 6.0 + t2 * t2 / 120.0, torch.sin(t) / t)
    b = torch.where(small, 0.5 - t2 / 24.0 + t2 * t2 / 720.0, (1.0 - torch.cos(t)) / t2_safe)
    return torch.eye(3, dtype=v.dtype, device=v.device) + a * K + b * (K @ K)


def so3_log(R: torch.Tensor) -> torch.Tensor:
    R = R.reshape(3, 3)
    vee = torch.stack((R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]))
    sin_t = 0.5 * torch.linalg.vector_norm(vee)
    t = torch.atan2(sin_t, 0.5 * (torch.trace(R) - 1.0))
    factor = torch.where(sin_t < 1e-07, 0.5 + t * t / 12.0, t / (2.0 * sin_t).clamp_min(1e-15))
    return factor * vee


def state_diff(later: TorchNav, e_pos, e_vel, e_dcm) -> torch.Tensor:
    """9-state difference 'later - earlier'."""
    return torch.cat((later.pos.reshape(3) - e_pos.reshape(3), later.vel.reshape(3) - e_vel.reshape(3),
                      so3_log(later.dcm.reshape(3, 3) @ e_dcm.reshape(3, 3).T) / ATT_SIGN))


def mechanize(nav: TorchNav, gyro, accel, dt) -> TorchNav:
    # ECEF coordinates (~6.4e6 m) have a 0.5 m ulp in float32; the INS must stay float64.
    if nav.pos.dtype != torch.float64 or nav.vel.dtype != torch.float64 or nav.dcm.dtype != torch.float64:
        raise TypeError(f'mechanize requires float64 navigation state; received pos={nav.pos.dtype}, '
                        f'vel={nav.vel.dtype}, dcm={nav.dcm.dtype}')
    dt = float(dt)
    C0 = nav.dcm
    f = accel.reshape(3) - nav.ba
    C1 = so3_exp(C0.new_tensor(-OMEGA_IE * dt)) @ C0 @ so3_exp((gyro.reshape(3) - nav.bg) * dt)
    omega = C0.new_tensor(OMEGA_IE)
    r = nav.pos.reshape(3)
    x, y, z = r.unbind()
    radius = torch.linalg.vector_norm(r).clamp_min(1.0)
    z2_r2 = z * z / (radius * radius)
    j2 = 1.5 * J2 * (WGS84_A / radius) ** 2
    xy, zf = 1.0 - j2 * (5.0 * z2_r2 - 1.0), 1.0 - j2 * (5.0 * z2_r2 - 3.0)
    gravity = -GM / radius ** 3 * torch.stack((x * xy, y * xy, z * zf))
    acc = (0.5 * (C0 + C1) @ f + gravity - torch.linalg.cross(omega, torch.linalg.cross(omega, r))
           - 2.0 * torch.linalg.cross(omega, nav.vel))
    vel = nav.vel + acc * dt
    return TorchNav(nav.pos + 0.5 * (nav.vel + vel) * dt, vel, C1, nav.ba, nav.bg)


# --- Network -----------------------------------------------------------------
class CLAOutput(NamedTuple):
    kalman_gain: torch.Tensor                 # [B, 9, nmax], zero beyond N_k
    attention: torch.Tensor
    recurrent_state: tuple | None = None      # LSTM (h, c) carried to the next epoch
    cov_scale: torch.Tensor | None = None     # 'covariance' mode: [B, 10] = (s_1..s_9, lambda)


class MaskedCLA(nn.Module):
    """Masked CNN -> LSTM -> attention -> FC Kalman gain (Yan Fig. 8, Eqs. (21)-(29)).

    Measurements are packed as a compact prefix (no parameter depends on the
    satellite identity).  The LSTM runs along the feature index t of one epoch;
    its final state initializes the next epoch (Eq. (24)).
    """

    def __init__(self, nmax: int, dropout: float = cfg.RECURRENT_DROPOUT,
                 forget_bias: float = cfg.LSTM_FORGET_BIAS, gain_scale_mode: str = 'empirical') -> None:
        super().__init__()
        self.nmax = int(nmax)
        if self.nmax <= 0:
            raise ValueError('nmax must be positive')
        if gain_scale_mode not in GAIN_SCALE_MODES:
            raise ValueError(f'gain_scale_mode must be one of {GAIN_SCALE_MODES}; received {gain_scale_mode!r}')
        # 'empirical': K = diag(g) K_n;  'none': K = K_n (g must stay 1);
        # 'analytic' : K = diag(g) K_n diag(1/sigma_j), g = d (steady-state prior std).
        self.gain_scale_mode = str(gain_scale_mode)
        self.dropout = float(dropout)
        self.forget_bias = float(forget_bias)
        # Creation order fixes the RNG stream and the checkpoint keys; keep it.
        self.conv = nn.Module()
        self.conv.weight = nn.Parameter(torch.empty(CONV_CH, 1, KERNEL))
        self.conv.bias = nn.Parameter(torch.zeros(CONV_CH))
        nn.init.kaiming_uniform_(self.conv.weight, a=5 ** 0.5)
        self.conv.register_buffer('_mask_kernel', torch.ones(1, 1, KERNEL), persistent=False)

        self.lstm = nn.Module()
        self.lstm.lstm = nn.LSTM(input_size=CONV_CH, hidden_size=HIDDEN, num_layers=LAYERS,
                                 dropout=self.dropout, batch_first=True)
        for name, p in self.lstm.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(p)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(p)
            elif 'bias' in name:
                nn.init.zeros_(p)
        # Forget-gate bias (PyTorch gate order i, f, g, o): set on bias_ih only so the
        # effective bias equals forget_bias.  Deterministic fill: RNG stream unchanged.
        with torch.no_grad():
            for layer in range(LAYERS):
                getattr(self.lstm.lstm, f'bias_ih_l{layer}')[HIDDEN:2 * HIDDEN].fill_(self.forget_bias)

        self.attention = nn.Module()
        self.attention.proj = nn.Linear(HIDDEN, HIDDEN, bias=True)
        self.attention.v = nn.Linear(HIDDEN, 1, bias=False)
        nn.init.xavier_uniform_(self.attention.proj.weight)
        nn.init.zeros_(self.attention.proj.bias)
        nn.init.xavier_uniform_(self.attention.v.weight)

        # Near-zero gain init keeps a gradient path into upstream blocks.
        self.gain_head = nn.Linear(HIDDEN, gain_head_out(self.gain_scale_mode, self.nmax))
        nn.init.xavier_uniform_(self.gain_head.weight, gain=1e-2)
        nn.init.zeros_(self.gain_head.bias)

        # Fixed (non-trainable) physical row scale g (meaning set by gain_scale_mode).
        # Set once from a classical EKF on the training split; saved in the checkpoint.
        self.register_buffer('gain_row_scale', torch.ones(N_STATE), persistent=True)
        # Fixed (non-trainable) input scale of the 24 fixed features: RMS of each
        # 3-component group over the training split (classical EKF).  Saved in the checkpoint.
        self.register_buffer('fixed_scale', torch.ones(FIXED_DIM), persistent=True)

    @torch.no_grad()
    def set_gain_row_scale(self, g) -> None:
        g = torch.as_tensor(g, dtype=self.gain_row_scale.dtype, device=self.gain_row_scale.device).reshape(-1)
        if g.numel() != N_STATE:
            raise ValueError(f'gain_row_scale must have {N_STATE} entries; received {g.numel()}')
        if not bool(torch.all(torch.isfinite(g))) or not bool(torch.all(g > 0)):
            raise ValueError('gain_row_scale must be finite and strictly positive')
        if self.gain_scale_mode in ('none', 'covariance') and not bool(torch.all(g == 1)):
            raise ValueError(f"gain_scale_mode={self.gain_scale_mode!r} requires gain_row_scale = 1")
        self.gain_row_scale.copy_(g)

    @torch.no_grad()
    def set_fixed_scale(self, s) -> None:
        s = torch.as_tensor(s, dtype=self.fixed_scale.dtype, device=self.fixed_scale.device).reshape(-1)
        if s.numel() != FIXED_DIM:
            raise ValueError(f'fixed_scale must have {FIXED_DIM} entries; received {s.numel()}')
        if not bool(torch.all(torch.isfinite(s))) or not bool(torch.all(s > 0)):
            raise ValueError('fixed_scale must be finite and strictly positive')
        self.fixed_scale.copy_(s)

    def forward(self, fixed, obs, mask, channel_mask, recurrent_state=None, column_sigma=None,
                prior_cov=None, H=None, R=None) -> CLAOutput:
        """fixed [B,24], obs [B,nmax,2], mask [B,nmax] (compact prefix), channel_mask [B,nmax,2].

        column_sigma [B,nmax]: Eq. (3) std [m] of each measurement row (padding: any
        positive value).  Required for gain_scale_mode='analytic', ignored otherwise.
        prior_cov [B,9,9], H [B,nmax,9] (zero padding rows), R [B,nmax,nmax] (identity
        padding block): required for gain_scale_mode='covariance', ignored otherwise.
        """
        B = fixed.shape[0]
        if obs.shape[1] != self.nmax:
            raise ValueError(f'observations have {obs.shape[1]} rows; model nmax={self.nmax}')
        valid = mask.bool()
        prefix = torch.arange(self.nmax, device=mask.device).unsqueeze(0) < valid.sum(dim=1).unsqueeze(1)
        if not torch.equal(prefix, valid):
            raise ValueError('measurement mask must be a compact prefix (Eq. (21))')
        channel = channel_mask.bool()
        if not torch.equal(channel[:, :, 1], valid):
            raise ValueError('innovation channel_mask must match mask for Eq. (16)')
        obs = obs * channel.to(dtype=obs.dtype)
        # Input normalization of the fixed features (train/validation/test all pass here).
        fixed = fixed / self.fixed_scale.to(dtype=fixed.dtype).view(1, FIXED_DIM)
        fixed_valid = torch.ones((B, FIXED_DIM), dtype=torch.bool, device=fixed.device)

        # Eq. (15)-(17): X_k = [fixed, (dy_1, dyt_1), ..., (dy_Nk, dyt_Nk), 0...].
        x = torch.cat((fixed, obs.reshape(B, 2 * self.nmax)), dim=1)
        # M_{k,t}: valid prefix of length 24 + 2 N_k, minus missing lagged residuals (holes).
        feat_mask = torch.cat((fixed_valid, channel.reshape(B, 2 * self.nmax)), dim=1)

        # Masked CNN + max-pool, Eqs. (22)-(23): [B,D] -> [B,D,24].
        m = feat_mask.to(dtype=x.dtype).unsqueeze(1)
        conv = F.conv1d(x.unsqueeze(1) * m, self.conv.weight, bias=None, stride=1, padding=1)
        count = F.conv1d(m, self.conv._mask_kernel.to(dtype=x.dtype), stride=1, padding=1)
        maps = F.relu(conv / count.clamp_min(MASK_EPS)
                      + self.conv.bias.view(1, -1, 1) * (count > 0).to(dtype=x.dtype)) * m
        pooled = F.max_pool1d(torch.where(m.bool(), maps, torch.full_like(maps, torch.finfo(maps.dtype).min)),
                              kernel_size=POOL, stride=1, padding=POOL // 2)
        feats = torch.where(m.bool(), pooled, torch.zeros_like(pooled)).transpose(1, 2)

        # Masked LSTM along t, Eqs. (24)-(25).  Holding (h, c) at every t with M_{k,t} = 0 equals
        # skipping that step, so the valid steps are gathered into a compact prefix (stable
        # order), run as one packed sequence, and scattered back to their positions.  This
        # also holds the state at holes inside the prefix, not only at the zero-padding.
        shape = (LAYERS, B, HIDDEN)
        if recurrent_state is None:
            h0, c0 = feats.new_zeros(shape), feats.new_zeros(shape)
        else:
            h0, c0 = recurrent_state
            if tuple(h0.shape) != shape or tuple(c0.shape) != shape:
                raise ValueError(f'recurrent_state must have shape {shape}; '
                                 f'received {tuple(h0.shape)}, {tuple(c0.shape)}')
        D = feats.shape[1]
        order = torch.argsort((~feat_mask).to(dtype=torch.int8), dim=1, stable=True)   # valid t first
        compact = feats.gather(1, order.unsqueeze(-1).expand(-1, -1, CONV_CH))
        packed = nn.utils.rnn.pack_padded_sequence(compact, feat_mask.sum(dim=1).to(device='cpu', dtype=torch.int64),
                                                   batch_first=True, enforce_sorted=False)
        out, (h, c) = self.lstm.lstm(packed, (h0.contiguous(), c0.contiguous()))
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=D)
        lstm_out = out.new_zeros((B, D, HIDDEN)).scatter(1, order.unsqueeze(-1).expand(-1, -1, HIDDEN), out)

        # Masked attention, Eqs. (26)-(28).
        att_valid = feat_mask.bool()
        score = self.attention.v(torch.tanh(self.attention.proj(lstm_out))).squeeze(-1)
        score = score.masked_fill(~att_valid, -torch.inf)
        none_valid = ~att_valid.any(dim=1)
        att = torch.softmax(score.masked_fill(none_valid.unsqueeze(1), 0.0), dim=1)
        att = att * att_valid.to(dtype=att.dtype)
        att = torch.where(none_valid.unsqueeze(1), torch.zeros_like(att),
                          att / att.sum(dim=1, keepdim=True).clamp_min(1e-12))
        context = torch.sum(att.unsqueeze(-1) * lstm_out, dim=1)

        # Masked FC gain (Sec. II-B text after Eq. (29); no equation in the paper):
        # K = diag(g) K_n [diag(1/sigma) for 'analytic'], columns j >= N_k zeroed.
        if self.gain_scale_mode == 'covariance':
            if prior_cov is None or H is None or R is None:
                raise ValueError("gain_scale_mode='covariance' requires prior_cov, H and R")
            raw = self.gain_head(context).to(dtype=torch.float64)                  # [B, 10]
            s = 10.0 ** (COV_LOG10_RANGE_P * torch.tanh(raw[:, :N_STATE]))         # [B, 9]
            lam = 10.0 ** (COV_LOG10_RANGE_R * torch.tanh(raw[:, N_STATE]))        # [B]
            P = prior_cov.to(dtype=torch.float64).reshape(B, N_STATE, N_STATE)
            Hm = H.to(dtype=torch.float64).reshape(B, self.nmax, N_STATE)
            Rm = R.to(dtype=torch.float64).reshape(B, self.nmax, self.nmax)
            Ps = s.unsqueeze(2) * P * s.unsqueeze(1)
            HP = Hm @ Ps                                                         # [B, nmax, 9]
            S = HP @ Hm.transpose(1, 2) + lam.view(B, 1, 1) * Rm
            S = 0.5 * (S + S.transpose(1, 2))
            # Clock-projected R is rank-deficient (one mode per clock group), as in
            # kf_update: use the same pseudo-inverse.
            K = HP.transpose(1, 2) @ torch.linalg.pinv(S, rtol=1e-12, hermitian=True)   # P~ H^T S^+
            K = K * valid.to(dtype=K.dtype).unsqueeze(1)
            return CLAOutput(kalman_gain=K, attention=att, recurrent_state=(h, c),
                             cov_scale=torch.cat((s, lam.unsqueeze(1)), dim=1))

        K_n = self.gain_head(context).view(B, N_STATE, self.nmax)
        K = K_n * self.gain_row_scale.to(dtype=K_n.dtype).view(1, N_STATE, 1)
        if self.gain_scale_mode == 'analytic':
            if column_sigma is None:
                raise ValueError("gain_scale_mode='analytic' requires column_sigma")
            sig = column_sigma.to(dtype=K.dtype).reshape(B, self.nmax)
            if not bool(torch.all(torch.isfinite(sig[valid]))) or not bool(torch.all(sig[valid] > 0)):
                raise ValueError('column_sigma must be finite and strictly positive on valid rows')
            sig = torch.where(valid, sig, torch.ones_like(sig))
            K = K / sig.unsqueeze(1)
        K = K * valid.to(dtype=K.dtype).unsqueeze(1)
        return CLAOutput(kalman_gain=K, attention=att, recurrent_state=(h, c))


def network_input(prev: dict, sat_ids, innovation, accel, gyro, *, nmax: int, sigma=None):
    """Compact zero-padded Eq. (15)-(17) input of epoch k.

    prev: context of epoch k-1 ('sat_ids', 'residual', 'sigma', 'dx_innov',
    'dx_resid', 'accel', 'gyro').  Row j = measurement j of sat_ids; the satellite
    identity is used only to find its lagged residual (Eq. (11)).
    sigma: Eq. (3) standard deviation of each current measurement [m].  The network
    sees innovation / sigma and the lagged residual / its own sigma of epoch k-1;
    innovation_padded stays in metres (it multiplies the Kalman gain).
    Returns (fixed, obs, mask, channel_mask, innovation_padded); fixed is raw and is
    normalized inside MaskedCLA.forward.
    """
    sat_ids = tuple(str(s) for s in sat_ids)
    n, nmax = len(sat_ids), int(nmax)
    if len(set(sat_ids)) != n:
        raise ValueError('current_sat_ids must contain unique identities')
    if n > nmax:
        raise ValueError(f'{n} measurements exceed network capacity nmax={nmax}; apply select_measurements first')
    innovation = innovation.reshape(n)
    dev = innovation.device
    sigma = innovation.new_ones(n) if sigma is None else sigma.reshape(n).to(dtype=innovation.dtype)
    prev_sigma = prev.get('sigma')
    if prev_sigma is None:
        prev_sigma = innovation.new_ones(len(prev['sat_ids']))
    if bool(torch.any(sigma <= 0)) or bool(torch.any(prev_sigma <= 0)):
        raise ValueError('measurement sigma must be strictly positive')
    fixed = torch.cat((accel.reshape(3) - prev['accel'], gyro.reshape(3) - prev['gyro'],
                       prev['dx_resid'], prev['dx_innov']))
    prev_row = {str(s): i for i, s in enumerate(prev['sat_ids'])}
    obs = innovation.new_zeros((nmax, OBS_DIM))
    mask = torch.zeros(nmax, dtype=torch.bool, device=dev)
    channel = torch.zeros((nmax, OBS_DIM), dtype=torch.bool, device=dev)
    padded = innovation.new_zeros(nmax)
    for row, s in enumerate(sat_ids):
        if s in prev_row:
            i = prev_row[s]
            obs[row, 0] = prev['residual'][i] / prev_sigma[i]
            channel[row, 0] = True
    if n:
        obs[:n, 1] = innovation / sigma
        mask[:n] = True
        channel[:n, 1] = True
        padded[:n] = innovation
    return fixed, obs, mask, channel, padded


# --- Checkpoint --------------------------------------------------------------
def save_checkpoint(path, model: MaskedCLA, metadata: dict | None = None) -> None:
    meta = {k: v for k, v in (metadata or {}).items()
            if k in ('epoch', 'validation_state_loss', 'validation_position_rmse_m')}
    torch.save({
        'model_state_dict': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        'state_dim': N_STATE, 'state_order': STATE_ORDER, 'nmax': int(model.nmax),
        'gain_shape': list(model.gain_head.weight.shape), 'feature_semantics_version': FEATURE_VERSION,
        'architecture': ARCHITECTURE, 'gain_scale_mode': model.gain_scale_mode, **meta,
    }, Path(path))


def load_checkpoint(path, device, *, expected_nmax=None) -> tuple[MaskedCLA, dict]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'checkpoint is missing: {path}')
    try:
        payload = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location=device)
    if not isinstance(payload, dict):
        raise TypeError('checkpoint must contain a dictionary payload')
    missing = sorted({'model_state_dict', 'state_dim', 'state_order', 'nmax', 'gain_shape',
                      'feature_semantics_version', 'architecture'}.difference(payload))
    if missing:
        raise KeyError(f'checkpoint is missing fields: {missing}')
    state = payload['model_state_dict']
    if not isinstance(state, dict) or any(k not in state for k in ('gain_head.weight', 'gain_row_scale', 'fixed_scale')):
        raise KeyError('checkpoint model_state_dict is missing gain_head.weight, gain_row_scale or fixed_scale')
    nmax = int(payload['nmax'])
    gain_shape = tuple(state['gain_head.weight'].shape)
    if int(payload['state_dim']) != N_STATE or payload['state_order'] != STATE_ORDER:
        raise ValueError(f"expected 9-state checkpoint {STATE_ORDER}; received state_dim={payload['state_dim']}, "
                         f"order={payload['state_order']!r}, gain shape={gain_shape}")
    if nmax <= 0 or (expected_nmax is not None and nmax != int(expected_nmax)):
        raise ValueError(f'checkpoint nmax={nmax} is invalid or does not match expected nmax={expected_nmax}')
    mode = str(payload.get('gain_scale_mode', 'empirical'))     # older checkpoints: empirical
    if mode not in GAIN_SCALE_MODES:
        raise ValueError(f'unsupported checkpoint gain_scale_mode: {mode!r}')
    want = (gain_head_out(mode, nmax), HIDDEN)
    if tuple(int(v) for v in payload['gain_shape']) != gain_shape or gain_shape != want:
        raise ValueError(f"expected gain shape {want}; declared={payload['gain_shape']}, actual={gain_shape}")
    if int(payload['feature_semantics_version']) != FEATURE_VERSION:
        raise ValueError('checkpoint feature semantics are incompatible')
    if payload['architecture'] != ARCHITECTURE:
        raise ValueError(f"unsupported checkpoint architecture: {payload['architecture']!r}")
    model = MaskedCLA(nmax=nmax, gain_scale_mode=mode).to(device=device, dtype=DTYPE)
    model.load_state_dict(state, strict=True)
    model.set_gain_row_scale(model.gain_row_scale.clone())        # validates g
    model.set_fixed_scale(model.fixed_scale.clone())              # validates the input scale
    model.eval()
    return model, payload


@torch.no_grad()
def clip_grad_norm_float64_(parameters, max_norm) -> torch.Tensor:
    """Clip by the float64 total gradient norm; return the pre-clip norm."""
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return torch.zeros((), dtype=torch.float64)
    norms = []
    for g in grads:
        if not bool(torch.all(torch.isfinite(g.detach())).cpu()):
            raise RuntimeError('gradient contains non-finite values before clipping')
        norms.append(torch.linalg.vector_norm(g.detach().to(dtype=torch.float64)))
    total = torch.linalg.vector_norm(torch.stack(norms))
    if not bool(torch.isfinite(total).cpu()):
        raise RuntimeError('float64 total gradient norm is non-finite')
    coef = torch.clamp(total.new_tensor(float(max_norm)) / (total + 1e-6), max=1.0)
    for g in grads:
        g.mul_(coef.to(device=g.device, dtype=g.dtype))
    return total
