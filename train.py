"""Offline training of the Masked CLA KalmanNet on the training dataset (paper Sec. II-C, Fig. 2).

    python train.py      ->  outputs/masked_cla_network.pt, outputs/training_history.json, outputs/training_loss.png

Loss: paper Eq. (30) ||x_k - x_hat_k||^2 on position, velocity and attitude (the
post-processed truth has no IMU biases), averaged over the T training epochs
plus gamma ||Theta||^2 (Eq. (32)); Adam with learning rate 0.01 (Table III).
"""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

import settings as cfg
from ins_filter import (STATE_SIZE, apply_correction, error_matrix, initial_state, measurement_model, propagate_ins,
                        state_difference, transition_matrix, truth_state)
from masked_cla_network import MaskedCLANetwork, build_network_input
from navigation_filter import first_epoch_features, next_epoch_features, prepare_measurements, run_filter
from read_dataset import load_navigation_data

CHECKPOINT_FILE = cfg.OUTPUT_FOLDER / 'masked_cla_network.pt'


def training_pass(network, data, measurements):
    """One pass over the training trajectory with the current network; accumulates the Eq. (32) gradient.

    The gradient of a correction dx_k also reaches later epochs through the linear
    error propagation Phi (a correction at k shifts the prior at k+1 by Phi dx_k),
    truncated every BACKPROP_WINDOW epochs (ASSUMPTIONS.md A15).
    """
    network.train()
    state = initial_state(data)
    previous = first_epoch_features(data, state)
    hidden, link, window_loss = None, torch.zeros(STATE_SIZE, dtype=torch.float64), 0.0
    times = data.fusion_times
    epoch_count = len(times) - 1
    loss_sum, position_square_sum = 0.0, 0.0
    for k in range(1, len(times)):
        state, mean_force, accel, gyro = propagate_ins(state, data, times[k - 1], times[k])
        Phi, _ = transition_matrix(error_matrix(state, mean_force), times[k] - times[k - 1])
        link = torch.from_numpy(Phi) @ link
        model = measurement_model(state, measurements[k], data.lever_arm, times[k], data.klobuchar_alpha,
                                  data.klobuchar_beta, network.max_measurements)
        if model is not None:
            count = len(model.innovation)
            features, length = build_network_input(previous, model.measurements.sat_ids, model.innovation, accel,
                                                   gyro, network.max_measurements)
            gain, hidden = network(torch.tensor(features, dtype=torch.float32), length, count, hidden)
            dx = gain[:, :count].double() @ torch.from_numpy(model.innovation)
            prior_error = torch.from_numpy(state_difference(truth_state(data, k), state)[:9])
            error = prior_error - (link - link.detach())[:9] - dx[:9]          # x_k - (x_k,k-1 + K dy_k)
            loss = error @ error / epoch_count                                  # Eq. (30), mean of Eq. (32)
            window_loss = window_loss + loss
            loss_sum += loss.item()
            position_square_sum += float(error[:3].detach() @ error[:3].detach())
            new_state = apply_correction(state, dx.detach().numpy())
            link = link + dx
            previous = next_epoch_features(data, k, previous, new_state, model, dx.detach().numpy(), accel, gyro)
            state = new_state
        if k % cfg.BACKPROP_WINDOW == 0 or k == len(times) - 1:
            if torch.is_tensor(window_loss):
                window_loss.backward()
            window_loss, link = 0.0, torch.zeros(STATE_SIZE, dtype=torch.float64)
            hidden = None if hidden is None else tuple(h.detach() for h in hidden)
    return loss_sum, np.sqrt(position_square_sum / epoch_count)


def save_loss_plot(history):
    epochs = [h['epoch'] for h in history]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, [h['position_rmse_m'] for h in history], color='#2a78d6', linewidth=2)
    ax.set(xlabel='Training epoch', ylabel='Position RMSE [m]', title='Training position RMSE (cf. paper Fig. 15)')
    ax.grid(color='#e5e5e3', linewidth=0.8)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(cfg.OUTPUT_FOLDER / 'training_loss.png', dpi=150)
    plt.close(fig)


def main():
    torch.manual_seed(cfg.RANDOM_SEED)
    cfg.OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    data = load_navigation_data('train')
    measurements = prepare_measurements(data, 'train')
    max_measurements = max(len(m) for m in measurements)                       # N_max of Eq. (16)
    print(f'{data.name}: {len(data.fusion_times)} fusion epochs, N_max = {max_measurements}')

    # Output scale of the gain rows from a traditional EKF pass (ASSUMPTIONS.md A11).
    classical = run_filter(data, measurements, network=None, use_fault_detection=False)
    network = MaskedCLANetwork(max_measurements)
    network.gain_row_scale.copy_(torch.tensor(np.maximum(classical['gain_row_rms'], 1e-12)))

    optimizer = torch.optim.Adam(network.parameters(), lr=cfg.LEARNING_RATE)
    history = []
    for epoch in range(1, cfg.TRAINING_EPOCHS + 1):
        optimizer.zero_grad()
        data_loss, position_rmse = training_pass(network, data, measurements)
        regularization = cfg.L2_WEIGHT * sum(torch.sum(p ** 2) for p in network.parameters())   # Eq. (32)
        regularization.backward()
        optimizer.step()
        history.append({'epoch': epoch, 'loss': data_loss + regularization.item(), 'position_rmse_m': position_rmse})
        print(f'epoch {epoch:4d} | loss {history[-1]["loss"]:.4f} | position RMSE {position_rmse:.3f} m')
        torch.save({'state_dict': network.state_dict(), 'max_measurements': max_measurements}, CHECKPOINT_FILE)
        (cfg.OUTPUT_FOLDER / 'training_history.json').write_text(json.dumps(history, indent=1))
    save_loss_plot(history)


if __name__ == '__main__':
    main()
