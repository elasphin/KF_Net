"""Offline training of the Masked CLA KalmanNet on the training dataset (paper Sec. II-C, Fig. 2).

    python train.py  ->  outputs/masked_cla_network.pt (model after the last epoch),
                         outputs/training_history.json, outputs/training_info.json
    python train.py --restart   (a new training even if a saved one exists)

An interrupted training continues after its last epoch when train.py runs again with the same
settings, code and data (training_state.py, outputs/training_state.pt).

Training of exp/paper with one change (branch exp/gain-scale): the network gain is scaled per row,
K = diag(g) K_net, g_i = RMS of row i of the traditional EKF gain on the training dataset (A11).
Loss: paper Eq. (30) ||x_k - x_hat_k||^2 on the position only (paper Sec. II-B:
"postprocessing position results as training labels", Fig. 8: truth trajectory),
averaged over the epochs and the three components (MSE, Table III) plus
gamma ||Theta||^2 (Eq. (32)); single-step gradient of Eq. (31): the filter state and the
LSTM state are detached at every fusion epoch (settings.BACKPROP_WINDOW = 1).
Adam with learning rate 0.01 (Table III) for 480 epochs (Fig. 15).
Alternating optimization (paper Sec. II-B, Ref. [15] Algorithm 2): in every
epoch the filter part theta (LSTM, attention, FC) is updated with the encoder
psi (masked CNN) frozen, then psi is updated with theta frozen; one Adam step
each (A15). The whole training dataset trains the network (paper Sec. III); there
is no validation, early stopping or gradient clipping, and the model
after the last epoch is tested. The filter uses the LEO orbit settings.LEO_TRAIN_ORBIT
(A26; the true orbit by default).
"""
import json
import time

import numpy as np
import torch

import settings as cfg
from data_io.data_cache import load_dataset
from navigation.ins_filter import STATE_SIZE
from navigation.masked_cla_network import FIXED_FEATURE_SIZE, MaskedCLANetwork
from navigation.navigation_filter import run_filter
import training_state

CHECKPOINT_FILE = cfg.OUTPUT_FOLDER / 'masked_cla_network.pt'


def training_step(network, optimizer, parameters, data, measurements):
    """One pass over the training dataset that updates only 'parameters' (the others are frozen)."""
    for p in network.parameters():
        p.requires_grad_(any(p is q for q in parameters))
    optimizer.zero_grad()
    result = run_filter(data, measurements, network, fault_detection=False, training=True)
    (cfg.L2_WEIGHT * sum(torch.sum(p ** 2) for p in parameters)).backward()          # gamma ||Theta||^2, Eq. (32)
    optimizer.step()
    return result


def main():
    torch.manual_seed(cfg.RANDOM_SEED)
    cfg.OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    if training_state.finished():
        return
    start_time = time.time()
    data, orbits = load_dataset('train')
    measurements = orbits[cfg.LEO_TRAIN_ORBIT]
    max_measurements = max(len(m) for m in measurements)             # N_max of Eq. (16)

    # Output scale of the gain rows from a traditional EKF on the training dataset (A11).
    classical = run_filter(data, measurements, fault_detection=False)
    network = MaskedCLANetwork(max_measurements)
    network.gain_row_scale.copy_(torch.tensor(np.maximum(classical['gain_row_rms'], 1e-12)))
    encoder = [network.conv.weight, network.conv_bias]                  # psi of Ref. [15]: masked CNN
    filter_part = [p for p in network.parameters() if not any(p is q for q in encoder)]   # theta: LSTM, attention, FC
    encoder_optimizer = torch.optim.Adam(encoder, lr=cfg.LEARNING_RATE)
    filter_optimizer = torch.optim.Adam(filter_part, lr=cfg.LEARNING_RATE)

    info = {
        'experiment': cfg.EXPERIMENT, 'dataset': data.name, 'leo_train_orbit': cfg.LEO_TRAIN_ORBIT,
        'training_samples': len(data.fusion_times) - 1,
        'learning_rate': cfg.LEARNING_RATE, 'max_epochs': cfg.TRAINING_EPOCHS, 'l2_weight': cfg.L2_WEIGHT,
        'backprop_window': cfg.BACKPROP_WINDOW, 'optimization': 'alternating: LSTM-attention-FC, then CNN [15]',
        'max_measurements': max_measurements, 'input_size': FIXED_FEATURE_SIZE + 2 * max_measurements,
        'network': f'Conv1D {cfg.CONV_FILTERS}x{cfg.CONV_KERNEL_SIZE} -> max-pool {cfg.POOL_KERNEL_SIZE} -> '
                   f'LSTM {cfg.LSTM_LAYERS}x{cfg.LSTM_UNITS} (dropout {cfg.LSTM_DROPOUT}) -> attention -> '
                   f'FC {cfg.FC_HIDDEN_UNITS} -> gain {STATE_SIZE}x{max_measurements}',
        'trainable_parameters': sum(p.numel() for p in network.parameters()),
        'classical_ekf_training_position_rmse_m': classical['position_rmse_m'],
        'gain_row_scale': network.gain_row_scale.tolist(),
    }
    print(json.dumps(info, indent=1))

    optimizers = (filter_optimizer, encoder_optimizer)
    first_epoch, history, info, time_before = training_state.resume(network, optimizers, info)
    for epoch in range(first_epoch, cfg.TRAINING_EPOCHS + 1):
        training_step(network, filter_optimizer, filter_part, data, measurements)   # theta, psi frozen
        train = training_step(network, encoder_optimizer, encoder, data, measurements)  # psi, theta frozen

        history.append({'epoch': epoch, 'train_loss': train['loss'], 'train_position_rmse_m': train['position_rmse_m']})
        print(f"epoch {epoch:4d} | train loss {train['loss']:.4g} | train RMSE {train['position_rmse_m']:.3f} m | "
              f"{time.time() - start_time:.0f} s")
        torch.save({'state_dict': network.state_dict(), 'max_measurements': max_measurements,     # model after the
                    'leo_train_orbit': cfg.LEO_TRAIN_ORBIT}, CHECKPOINT_FILE)                      # last epoch
        info.update(epochs_run=epoch, final_train_loss=train['loss'],
                    final_train_position_rmse_m=train['position_rmse_m'],
                    training_time_s=time_before + time.time() - start_time)
        (cfg.OUTPUT_FOLDER / 'training_history.json').write_text(json.dumps(history, indent=1))
        (cfg.OUTPUT_FOLDER / 'training_info.json').write_text(json.dumps(info, indent=1))
        training_state.save(epoch, network, optimizers, history, info)


if __name__ == '__main__':
    main()
