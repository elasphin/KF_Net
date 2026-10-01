"""Offline training of the Masked CLA KalmanNet on the training dataset (paper Sec. II-C, Fig. 2).

    python train.py  ->  outputs/masked_cla_network.pt (best validation model),
                         outputs/training_history.json, outputs/training_info.json

Loss: paper Eq. (30) ||x_k - x_hat_k||^2 on the position only (paper Sec. II-B:
"postprocessing position results as training labels", Fig. 8: truth trajectory),
averaged over the epochs and the three components (MSE, Table III) plus
gamma ||Theta||^2 (Eq. (32)); Adam with learning rate 0.01 (Table III).
Alternating optimization (paper Sec. II-B, Ref. [15] Algorithm 2): in every
epoch the filter part theta (LSTM, attention, FC) is updated with the encoder
psi (masked CNN) frozen, then psi is updated with theta frozen; one Adam step
each (A15), gradient norm clipped to 1 (A22). The first 80 % of the training
dataset trains the network, the last 20 % validates it (A21). The filter uses
the LEO orbit settings.LEO_TRAIN_ORBIT (A26; the true orbit by default).
"""
import json
import time

import numpy as np
import torch

import settings as cfg
from data_cache import load_dataset
from ins_filter import STATE_SIZE
from masked_cla_network import FIXED_FEATURE_SIZE, MaskedCLANetwork
from navigation_filter import run_filter

CHECKPOINT_FILE = cfg.OUTPUT_FOLDER / 'masked_cla_network.pt'


def training_step(network, optimizer, parameters, data, measurements, last):
    """One pass over the training part that updates only 'parameters' (the others are frozen)."""
    for p in network.parameters():
        p.requires_grad_(any(p is q for q in parameters))
    optimizer.zero_grad()
    result = run_filter(data, measurements, network, 0, last, fault_detection=False, training=True)
    (cfg.L2_WEIGHT * sum(torch.sum(p ** 2) for p in parameters)).backward()          # gamma ||Theta||^2, Eq. (32)
    torch.nn.utils.clip_grad_norm_(parameters, cfg.GRADIENT_CLIP_NORM)                # A22
    optimizer.step()
    return result


def main():
    torch.manual_seed(cfg.RANDOM_SEED)
    cfg.OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    start_time = time.time()
    data, orbits = load_dataset('train')
    measurements = orbits[cfg.LEO_TRAIN_ORBIT]
    last = len(data.fusion_times) - 1
    split = int(round(last * (1.0 - cfg.VALIDATION_FRACTION)))       # train: 0..split, validation: split..last
    max_measurements = max(len(m) for m in measurements[:split + 1])  # N_max of Eq. (16)

    # Output scale of the gain rows from a traditional EKF on the training part (A11).
    classical = run_filter(data, measurements, last=split, fault_detection=False)
    network = MaskedCLANetwork(max_measurements)
    network.gain_row_scale.copy_(torch.tensor(np.maximum(classical['gain_row_rms'], 1e-12)))
    encoder = [network.conv.weight, network.conv_bias]                  # psi of Ref. [15]: masked CNN
    filter_part = [p for p in network.parameters() if not any(p is q for q in encoder)]   # theta: LSTM, attention, FC
    encoder_optimizer = torch.optim.Adam(encoder, lr=cfg.LEARNING_RATE)
    filter_optimizer = torch.optim.Adam(filter_part, lr=cfg.LEARNING_RATE)

    info = {
        'dataset': data.name, 'leo_train_orbit': cfg.LEO_TRAIN_ORBIT,
        'training_samples': split, 'validation_samples': last - split,
        'learning_rate': cfg.LEARNING_RATE, 'max_epochs': cfg.TRAINING_EPOCHS,
        'early_stopping_patience': cfg.EARLY_STOPPING_PATIENCE, 'l2_weight': cfg.L2_WEIGHT,
        'backprop_window': cfg.BACKPROP_WINDOW, 'gradient_clip_norm': cfg.GRADIENT_CLIP_NORM, 'optimization': 'alternating: LSTM-attention-FC, then CNN [15]',
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
        training_step(network, filter_optimizer, filter_part, data, measurements, split)   # theta, psi frozen
        train = training_step(network, encoder_optimizer, encoder, data, measurements, split)  # psi, theta frozen
        validation = run_filter(data, measurements, network, split, last, fault_detection=False)

        history.append({'epoch': epoch, 'train_loss': train['loss'], 'validation_loss': validation['loss'],
                        'train_position_rmse_m': train['position_rmse_m'],
                        'validation_position_rmse_m': validation['position_rmse_m']})
        print(f"epoch {epoch:4d} | train loss {train['loss']:.4g} | validation loss {validation['loss']:.4g} | "
              f"train RMSE {train['position_rmse_m']:.3f} m | validation RMSE {validation['position_rmse_m']:.3f} m | "
              f"{time.time() - start_time:.0f} s")
        if validation['loss'] < best_loss:                                # keep the best validation model
            best_loss, epochs_without_improvement = validation['loss'], 0
            info.update(best_epoch=epoch, best_validation_loss=validation['loss'],
                        best_validation_position_rmse_m=validation['position_rmse_m'])
            torch.save({'state_dict': network.state_dict(), 'max_measurements': max_measurements,
                        'leo_train_orbit': cfg.LEO_TRAIN_ORBIT}, CHECKPOINT_FILE)
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
