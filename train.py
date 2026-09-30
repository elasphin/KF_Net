"""Offline training of the Masked CLA KalmanNet on the training dataset (paper Sec. II-C, Fig. 2).

    python train.py  ->  outputs/masked_cla_network.pt (best validation model),
                         outputs/training_history.json, outputs/training_info.json

Loss: paper Eq. (30) ||x_k - x_hat_k||^2 on position, velocity and attitude (the
post-processed truth has no IMU biases), averaged over the epochs plus
gamma ||Theta||^2 (Eq. (32)); Adam with learning rate 0.01 (Table III); one
Adam step per epoch (A15). The first 80 % of the training dataset trains the
network, the last 20 % validates it (A21).
"""
import json
import time

import numpy as np
import torch

import settings as cfg
from ins_filter import STATE_SIZE
from masked_cla_network import FIXED_FEATURE_SIZE, MaskedCLANetwork
from navigation_filter import prepare_measurements, run_filter
from read_dataset import load_navigation_data

CHECKPOINT_FILE = cfg.OUTPUT_FOLDER / 'masked_cla_network.pt'


def main():
    torch.manual_seed(cfg.RANDOM_SEED)
    cfg.OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    start_time = time.time()
    data = load_navigation_data('train')
    measurements = prepare_measurements(data, 'train')
    last = len(data.fusion_times) - 1
    split = int(round(last * (1.0 - cfg.VALIDATION_FRACTION)))       # train: 0..split, validation: split..last
    max_measurements = max(len(m) for m in measurements[:split + 1])  # N_max of Eq. (16)

    # Output scale of the gain rows from a traditional EKF on the training part (A11).
    classical = run_filter(data, measurements, last=split, fault_detection=False)
    network = MaskedCLANetwork(max_measurements)
    network.gain_row_scale.copy_(torch.tensor(np.maximum(classical['gain_row_rms'], 1e-12)))
    optimizer = torch.optim.Adam(network.parameters(), lr=cfg.LEARNING_RATE)

    info = {
        'dataset': data.name, 'training_samples': split, 'validation_samples': last - split,
        'learning_rate': cfg.LEARNING_RATE, 'max_epochs': cfg.TRAINING_EPOCHS,
        'early_stopping_patience': cfg.EARLY_STOPPING_PATIENCE, 'l2_weight': cfg.L2_WEIGHT,
        'backprop_window': cfg.BACKPROP_WINDOW,
        'max_measurements': max_measurements, 'input_size': FIXED_FEATURE_SIZE + 2 * max_measurements,
        'network': f'Conv1D {cfg.CONV_FILTERS}x{cfg.CONV_KERNEL_SIZE} -> max-pool {cfg.POOL_KERNEL_SIZE} -> '
                   f'LSTM {cfg.LSTM_LAYERS}x{cfg.LSTM_UNITS} (dropout {cfg.LSTM_DROPOUT}) -> attention -> '
                   f'FC {cfg.FC_HIDDEN_UNITS} -> gain {STATE_SIZE}x{max_measurements}',
        'trainable_parameters': sum(p.numel() for p in network.parameters()),
        'classical_ekf_training_position_rmse_m': classical['position_rmse_m'],
    }
    print(json.dumps(info, indent=1))

    history, best_loss, epochs_without_improvement = [], np.inf, 0
    for epoch in range(1, cfg.TRAINING_EPOCHS + 1):
        optimizer.zero_grad()
        train = run_filter(data, measurements, network, 0, split, fault_detection=False, training=True)
        regularization = cfg.L2_WEIGHT * sum(torch.sum(p ** 2) for p in network.parameters())   # Eq. (32)
        regularization.backward()
        optimizer.step()
        validation = run_filter(data, measurements, network, split, last, fault_detection=False)

        history.append({'epoch': epoch, 'train_loss': train['loss'], 'validation_loss': validation['loss'],
                        'train_position_rmse_m': train['position_rmse_m'],
                        'validation_position_rmse_m': validation['position_rmse_m']})
        print(f"epoch {epoch:4d} | train loss {train['loss']:.4g} | validation loss {validation['loss']:.4g} | "
              f"train RMSE {train['position_rmse_m']:.3f} m | validation RMSE {validation['position_rmse_m']:.3f} m")
        if validation['loss'] < best_loss:                                # keep the best validation model
            best_loss, epochs_without_improvement = validation['loss'], 0
            info.update(best_epoch=epoch, best_validation_loss=validation['loss'],
                        best_validation_position_rmse_m=validation['position_rmse_m'])
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
