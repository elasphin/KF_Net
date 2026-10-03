"""Masked CLA KalmanNet (paper Sec. II-B, Fig. 8, Table III) and its input vector."""
import torch
import torch.nn.functional as F
from torch import nn

import settings as cfg
from navigation.ins_filter import STATE_SIZE

FIXED_FEATURE_SIZE = 6 + 2 * STATE_SIZE      # [d_alpha(3), d_w(3), dx_hat(15), dx_tilde(15)] = 36


def network_input_values(previous, sat_ids, innovation, accel, gyro):
    """Paper Eq. (15): X_k = [d_alpha, d_w, dx_hat, dx_tilde, dy, dy_tilde], and which dy exist.

    previous: quantities of epoch k-1 ('accel', 'gyro', 'state_residual' Eq. (13),
    'state_innovation' Eq. (12), 'residuals' Eq. (11) as {sat_id: value}).
    A satellite that was not used at k-1 has lagged residual 0 (docs/ASSUMPTIONS.md A12).
    The state and measurement quantities are float64 tensors; in training they carry the gradient of the
    earlier corrections (navigation_filter.run_filter, A15).
    Returns X_k (tensor, length 36 + 2 N_k) and the boolean mask of the lagged residuals that exist.
    """
    zero = torch.zeros((), dtype=torch.float64)
    lagged_residual = [previous['residuals'].get(s, zero) for s in sat_ids]
    values = torch.cat([torch.from_numpy(accel - previous['accel']), torch.from_numpy(gyro - previous['gyro']),  # Eq. (14)
                        previous['state_residual'], previous['state_innovation'],                            # Eq. (13), (12)
                        torch.stack(lagged_residual), innovation])                                           # Eq. (11), (10)
    return values, torch.tensor([s in previous['residuals'] for s in sat_ids])


def build_network_input(previous, sat_ids, innovation, accel, gyro, max_measurements, normalization=None):
    """Paper Eq. (15)-(16): X_k, normalized (exp/input-norm), followed by zero padding.

    Returns the padded vector (tensor, length 36 + 2 N_max) and the number of valid entries (mask, Eq. (17)).
    """
    values, present = network_input_values(previous, sat_ids, innovation, accel, gyro)
    if normalization is not None:
        values = normalization(values, sat_ids, present)
    return F.pad(values, (0, FIXED_FEATURE_SIZE + 2 * max_measurements - len(values))), len(values)


SYSTEMS = ('G', 'C', 'L')
FEATURE_GROUPS = ((0, 3), (3, 6), (6, 6 + STATE_SIZE), (6 + STATE_SIZE, FIXED_FEATURE_SIZE))  # d_alpha, d_w, dx_hat, dx_tilde


class InputNormalization(nn.Module):
    """Normalization of the network input X_k (branch exp/input-norm, settings.INPUT_NORMALIZATION).

    'l2': each of the six groups d_alpha, d_w, dx_hat, dx_tilde, dy, dy_tilde divided by its L2 norm
          (as the input features of KalmanNet [14] and KalmanNet4SensorFusion, F.normalize, eps 1e-12).
    'zscore': (x - mean) / std with fixed statistics of the training dataset (set_statistics: features of a
          traditional EKF run): one mean and std per element of the 36 fixed features, and per system (G, C, L)
          for the residuals dy and the innovations dy_tilde, whose positions are not tied to a satellite. A
          missing lagged residual (0, A12) stays 0, i.e. the mean. A std of 0 is replaced by 1.
    The statistics are buffers, so they are saved with the model.
    """

    def __init__(self):
        super().__init__()
        if cfg.INPUT_NORMALIZATION not in ('l2', 'zscore'):
            raise ValueError(f"INPUT_NORMALIZATION must be 'l2' or 'zscore', not {cfg.INPUT_NORMALIZATION!r}")
        self.mode = cfg.INPUT_NORMALIZATION
        for name, size in (('fixed', FIXED_FEATURE_SIZE), ('residual', len(SYSTEMS)), ('innovation', len(SYSTEMS))):
            self.register_buffer(f'{name}_mean', torch.zeros(size, dtype=torch.float64))
            self.register_buffer(f'{name}_std', torch.ones(size, dtype=torch.float64))

    def forward(self, values, sat_ids, present):
        n = len(sat_ids)
        fixed, residual, innovation = values[:FIXED_FEATURE_SIZE], values[FIXED_FEATURE_SIZE:-n], values[-n:]
        if self.mode == 'l2':
            groups = [fixed[a:b] for a, b in FEATURE_GROUPS] + [residual, innovation]
            return torch.cat([F.normalize(g, dim=0, eps=1e-12) for g in groups])
        system = torch.tensor([SYSTEMS.index(s[0]) for s in sat_ids])
        return torch.cat([(fixed - self.fixed_mean) / self.fixed_std,
                          torch.where(present, (residual - self.residual_mean[system]) / self.residual_std[system],
                                      torch.zeros_like(residual)),
                          (innovation - self.innovation_mean[system]) / self.innovation_std[system]])

    @torch.no_grad()
    def set_statistics(self, samples):
        """samples: [(X_k, present, sat_ids)] of the training dataset (navigation_filter.run_filter input_samples)."""
        fixed = torch.stack([values[:FIXED_FEATURE_SIZE] for values, _, _ in samples])
        self.fixed_mean.copy_(fixed.mean(dim=0))
        self.fixed_std.copy_(_nonzero(fixed.std(dim=0)))
        for name in ('residual', 'innovation'):
            for i, system in enumerate(SYSTEMS):
                group = []
                for values, present, sat_ids in samples:
                    n = len(sat_ids)
                    in_system = torch.tensor([s[0] == system for s in sat_ids])
                    if name == 'residual':                                     # only the lagged residuals that exist
                        group.append(values[FIXED_FEATURE_SIZE:-n][in_system & present])
                    else:
                        group.append(values[-n:][in_system])
                group = torch.cat(group)
                if len(group) > 1:
                    getattr(self, f'{name}_mean')[i] = group.mean()
                    getattr(self, f'{name}_std')[i] = _nonzero(group.std())


def _nonzero(std):
    return torch.where(std > 0, std, torch.ones_like(std))


class MaskedCLANetwork(nn.Module):
    """Masked CNN -> masked LSTM -> masked attention -> masked FC -> Kalman gain (15 x N_max)."""

    def __init__(self, max_measurements):
        super().__init__()
        self.max_measurements = max_measurements
        self.conv = nn.Conv1d(1, cfg.CONV_FILTERS, cfg.CONV_KERNEL_SIZE, stride=1, bias=False)
        self.conv_bias = nn.Parameter(torch.zeros(cfg.CONV_FILTERS))                    # b of Eq. (22)
        self.lstm = nn.LSTM(cfg.CONV_FILTERS, cfg.LSTM_UNITS, num_layers=cfg.LSTM_LAYERS,
                            dropout=cfg.LSTM_DROPOUT, batch_first=True)
        self.attention_hidden = nn.Linear(cfg.LSTM_UNITS, cfg.LSTM_UNITS)                # W_h, b_h of Eq. (26)
        self.attention_vector = nn.Linear(cfg.LSTM_UNITS, 1, bias=False)                 # v of Eq. (26)
        self.fc_hidden = nn.Linear(cfg.LSTM_UNITS, cfg.FC_HIDDEN_UNITS)
        self.fc_output = nn.Linear(cfg.FC_HIDDEN_UNITS, STATE_SIZE * max_measurements)
        nn.init.zeros_(self.fc_output.weight)            # K = 0 before training (A12)
        nn.init.zeros_(self.fc_output.bias)
        self.input_normalization = InputNormalization()

    def forward(self, features, valid_length, measurement_count, hidden=None):
        """features: padded X_k [D]; returns the Kalman gain [15, N_max] and the LSTM state (h, c)."""
        D = features.shape[0]
        mask = (torch.arange(D) < valid_length).to(features.dtype).view(1, 1, D)       # M_k, Eq. (17), (21)
        pad = cfg.CONV_KERNEL_SIZE // 2

        # Masked CNN, Eq. (22): Z / max(N_t, eps) + 1[N_t > 0] b, N_t = valid samples in the window.
        z = self.conv(F.pad(features.view(1, 1, D) * mask, (pad, pad)))
        valid_count = F.conv1d(F.pad(mask, (pad, pad)), torch.ones(1, 1, cfg.CONV_KERNEL_SIZE))
        s = F.relu(z / valid_count.clamp_min(cfg.MASK_EPSILON)
                   + (valid_count > 0).to(z.dtype) * self.conv_bias.view(1, -1, 1)) * mask
        # Pooling (Fig. 8) with stride 1 over valid samples, so that M_out = M_in (Eq. (23)).
        pooled = F.max_pool1d(s.masked_fill(mask == 0, float('-inf')), cfg.POOL_KERNEL_SIZE, stride=1,
                              padding=cfg.POOL_KERNEL_SIZE // 2)
        s = torch.where(mask > 0, pooled, torch.zeros_like(pooled))

        # Masked LSTM, Eq. (24)-(25): steps with M = 0 keep the previous (h, c). They are all at the end
        # of X_k, so running the LSTM on the valid steps only is exactly Eq. (25).
        valid_steps = s[:, :, :valid_length].transpose(1, 2)                            # [1, L, 24]
        outputs, hidden = self.lstm(valid_steps, hidden)

        # Masked attention, Eq. (26)-(29): steps with M = 0 have score -inf, i.e. are left out of the softmax.
        scores = self.attention_vector(torch.tanh(self.attention_hidden(outputs[0]))).squeeze(1)
        weights = torch.softmax(scores, dim=0)
        context = weights @ outputs[0]

        # Masked FC: Kalman gain, columns j >= N_k are zero.
        gain = self.fc_output(F.relu(self.fc_hidden(context))).view(STATE_SIZE, self.max_measurements)
        column_mask = (torch.arange(self.max_measurements) < measurement_count).to(gain.dtype)
        return gain * column_mask, hidden
