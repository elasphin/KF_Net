"""Masked CLA KalmanNet (paper Sec. II-B, Fig. 8, Table III) and its input vector."""
import torch
import torch.nn.functional as F
from torch import nn

import settings as cfg
from navigation.ins_filter import STATE_SIZE

FIXED_FEATURE_SIZE = 6 + 2 * STATE_SIZE      # [d_alpha(3), d_w(3), dx_hat(15), dx_tilde(15)] = 36


def build_network_input(previous, sat_ids, innovation, accel, gyro, max_measurements):
    """Paper Eq. (15)-(16): X_k = [d_alpha, d_w, dx_hat, dx_tilde, dy, dy_tilde] followed by zero padding.

    previous: quantities of epoch k-1 ('accel', 'gyro', 'state_residual' Eq. (13),
    'state_innovation' Eq. (12), 'residuals' Eq. (11) as {sat_id: value}).
    A satellite that was not used at k-1 has lagged residual 0 (docs/ASSUMPTIONS.md A12).
    The state and measurement quantities are float64 tensors; in training they carry the gradient of the
    earlier corrections (navigation_filter.run_filter, A15).
    Returns the padded vector (tensor, length 36 + 2 N_max) and the number of valid entries (mask, Eq. (17)).
    """
    zero = torch.zeros((), dtype=torch.float64)
    lagged_residual = [previous['residuals'].get(s, zero) for s in sat_ids]
    values = torch.cat([torch.from_numpy(accel - previous['accel']), torch.from_numpy(gyro - previous['gyro']),  # Eq. (14)
                        previous['state_residual'], previous['state_innovation'],                            # Eq. (13), (12)
                        torch.stack(lagged_residual), innovation])                                           # Eq. (11), (10)
    return F.pad(values, (0, FIXED_FEATURE_SIZE + 2 * max_measurements - len(values))), len(values)


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
        self.register_buffer('gain_row_scale', torch.ones(STATE_SIZE))                  # A11

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
        return gain * column_mask * self.gain_row_scale.view(STATE_SIZE, 1), hidden
