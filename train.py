"""Offline training of the Masked CLA KalmanNet on the training dataset (paper Sec. II-C, Fig. 2).

    python train.py  ->  outputs/masked_cla_network.pt (best validation model),
                         outputs/training_history.json, outputs/training_info.json

Loss: paper Eq. (30) ||x_k - x_hat_k||^2 on position, velocity and attitude (the
post-processed truth has no IMU biases), averaged over the epochs plus
gamma ||Theta||^2 (Eq. (32)); Adam with learning rate 0.01 (Table III).
The first 80 % of the training dataset trains the network, the last 20 %
validates it (ASSUMPTIONS.md A21).
"""
import json
import time

import numpy as np
import torch

import settings as cfg
from ins_filter import (STATE_SIZE, apply_correction, error_matrix, initial_state, measurement_model, propagate_ins,
                        state_difference, transition_matrix, truth_state)
from masked_cla_network import FIXED_FEATURE_SIZE, MaskedCLANetwork, build_network_input
from navigation_filter import first_epoch_features, next_epoch_features, prepare_measurements, run_filter
from read_dataset import load_navigation_data

CHECKPOINT_FILE = cfg.OUTPUT_FOLDER / 'masked_cla_network.pt'


def run_network(network, data, measurements, first, last, training):
    """Filter with the network gain over fusion epochs first..last (start from the truth at 'first').

    training=True accumulates the Eq. (32) gradient. The gradient of a correction
    dx_k also reaches later epochs through the linear error propagation Phi (a
    correction at k shifts the prior at k+1 by Phi dx_k), truncated every
    BACKPROP_WINDOW epochs (ASSUMPTIONS.md A15).
    Returns the mean Eq. (30) loss and the position RMSE [m].
    """
    network.train(training)
    state = initial_state(data, first)
    previous = first_epoch_features(data, state, first)
    hidden, link, window_loss = None, torch.zeros(STATE_SIZE, dtype=torch.float64), 0.0
    times = data.fusion_times
    epoch_count = last - first
    loss_sum, position_square_sum = 0.0, 0.0
    with torch.set_grad_enabled(training):
        for k in range(first + 1, last + 1):
            state, mean_force, accel, gyro = propagate_ins(state, data, times[k - 1], times[k])
            Phi, _ = transition_matrix(error_matrix(state, mean_force), times[k] - times[k - 1])
            link = torch.from_numpy(Phi) @ link
            model = measurement_model(state, measurements[k], data.lever_arm, times[k], data.klobuchar_alpha,
                                      data.klobuchar_beta, network.max_measurements)
            if model is not None:
                count = len(model.innovation)
                features, length = build_network_input(previous, model.measurements.sat_ids, model.innovation,
                                                       accel, gyro, network.max_measurements)
                gain, hidden = network(torch.tensor(features, dtype=torch.float32), length, count, hidden)
                dx = gain[:, :count].double() @ torch.from_numpy(model.innovation)
                prior_error = torch.from_numpy(state_difference(truth_state(data, k), state)[:9])
                error = prior_error - (link - link.detach())[:9] - dx[:9]      # x_k - (x_k,k-1 + K dy_k)
                loss = error @ error / epoch_count                              # Eq. (30), mean of Eq. (32)
                window_loss = window_loss + loss
                loss_sum += loss.item()
                position_square_sum += float(error[:3].detach() @ error[:3].detach())
                new_state = apply_correction(state, dx.detach().numpy())
                link = link + dx
                previous = next_epoch_features(data, k, previous, new_state, model, dx.detach().numpy(), accel, gyro)
                state = new_state
            if k % cfg.BACKPROP_WINDOW == 0 or k == last:
                if training and torch.is_tensor(window_loss):
                    window_loss.backward()
                window_loss, link = 0.0, torch.zeros(STATE_SIZE, dtype=torch.float64)
                hidden = None if hidden is None else tuple(h.detach() for h in hidden)
    return loss_sum, np.sqrt(position_square_sum / epoch_count)


def main():
    torch.manual_seed(cfg.RANDOM_SEED)
    cfg.OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    start_time = time.time()
    data = load_navigation_data('train')
    measurements = prepare_measurements(data, 'train')
    last_epoch = len(data.fusion_times) - 1
    split = int(round(last_epoch * (1.0 - cfg.VALIDATION_FRACTION)))           # train: 0..split, validation: split..end
    max_measurements = max(len(m) for m in measurements[:split + 1])          # N_max of Eq. (16)

    # Output scale of the gain rows from a traditional EKF on the training part (ASSUMPTIONS.md A11).
    classical = run_filter(data, measurements, network=None, use_fault_detection=False, last=split)
    network = MaskedCLANetwork(max_measurements)
    network.gain_row_scale.copy_(torch.tensor(np.maximum(classical['gain_row_rms'], 1e-12)))
    optimizer = torch.optim.Adam(network.parameters(), lr=cfg.LEARNING_RATE)

    info = {
        'dataset': data.name,
        'training_samples': split, 'validation_samples': last_epoch - split,
        'learning_rate': cfg.LEARNING_RATE, 'max_epochs': cfg.TRAINING_EPOCHS,
        'early_stopping_patience': cfg.EARLY_STOPPING_PATIENCE, 'l2_weight': cfg.L2_WEIGHT,
        'backprop_window': cfg.BACKPROP_WINDOW,
        'max_measurements': max_measurements, 'input_size': FIXED_FEATURE_SIZE + 2 * max_measurements,
        'network': f'Conv1D {cfg.CONV_FILTERS}x{cfg.CONV_KERNEL_SIZE} -> max-pool {cfg.POOL_KERNEL_SIZE} -> '
                   f'LSTM {cfg.LSTM_LAYERS}x{cfg.LSTM_UNITS} (dropout {cfg.LSTM_DROPOUT}) -> attention -> '
                   f'FC {cfg.FC_HIDDEN_UNITS} -> gain {STATE_SIZE}x{max_measurements}',
        'trainable_parameters': sum(p.numel() for p in network.parameters()),
        'classical_ekf_training_position_rmse_m': float(np.sqrt(np.mean(np.sum(
            (classical['estimate'] - classical['truth']) ** 2, axis=1)))),
    }
    print(json.dumps(info, indent=1))

    history, best_loss, epochs_without_improvement = [], np.inf, 0
    for epoch in range(1, cfg.TRAINING_EPOCHS + 1):
        optimizer.zero_grad()
        train_loss, train_rmse = run_network(network, data, measurements, 0, split, training=True)
        regularization = cfg.L2_WEIGHT * sum(torch.sum(p ** 2) for p in network.parameters())   # Eq. (32)
        regularization.backward()
        optimizer.step()
        validation_loss, validation_rmse = run_network(network, data, measurements, split, last_epoch, training=False)

        history.append({'epoch': epoch, 'train_loss': train_loss, 'validation_loss': validation_loss,
                        'train_position_rmse_m': train_rmse, 'validation_position_rmse_m': validation_rmse})
        print(f'epoch {epoch:4d} | train loss {train_loss:.4g} | validation loss {validation_loss:.4g} | '
              f'train RMSE {train_rmse:.3f} m | validation RMSE {validation_rmse:.3f} m')
        if validation_loss < best_loss:                                       # keep the best validation model
            best_loss, epochs_without_improvement = validation_loss, 0
            info.update(best_epoch=epoch, best_validation_loss=validation_loss,
                        best_validation_position_rmse_m=validation_rmse)
            torch.save({'state_dict': network.state_dict(), 'max_measurements': max_measurements}, CHECKPOINT_FILE)
        else:
            epochs_without_improvement += 1
        info.update(epochs_run=epoch, training_time_s=time.time() - start_time)
        (cfg.OUTPUT_FOLDER / 'training_history.json').write_text(json.dumps(history, indent=1))
        (cfg.OUTPUT_FOLDER / 'training_info.json').write_text(json.dumps(info, indent=1))
        if epochs_without_improvement >= cfg.EARLY_STOPPING_PATIENCE:
            print(f'early stopping: no better validation loss for {cfg.EARLY_STOPPING_PATIENCE} epochs')
            break


if __name__ == '__main__':
    main()
