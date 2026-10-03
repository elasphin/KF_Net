"""Offline training of the Masked CLA KalmanNet on the training dataset (paper Sec. II-C, Fig. 2).

    python train.py  ->  outputs/masked_cla_network.pt (model after the last epoch),
                         outputs/training_history.json, outputs/training_info.json
    python train.py --restart   (a new training even if a saved one exists)

An interrupted training continues after its last epoch when train.py runs again with the same
settings, code and data (training_state.py, outputs/training_state.pt).

Training of exp/paper with one change (branch exp/alternating): joint instead of alternating optimization.
Loss: paper Eq. (30) ||x_k - x_hat_k||^2 on the position only (paper Sec. II-B:
"postprocessing position results as training labels", Fig. 8: truth trajectory),
averaged over the epochs and the three components (MSE, Table III) plus
gamma ||Theta||^2 (Eq. (32)); single-step gradient of Eq. (31): the filter state and the
LSTM state are detached at every fusion epoch (settings.BACKPROP_WINDOW = 1).
Adam with learning rate 0.01 (Table III) for 480 epochs (Fig. 15).
Joint optimization: in every epoch two passes over the training dataset, each followed
by one Adam step on all parameters (the same passes and steps as the alternating
optimization of exp/paper, where the first step updates only the LSTM, attention and
FC and the second only the masked CNN). The whole training dataset trains the network (paper Sec. III); there
is no validation, early stopping, gradient clipping or gain scale, and the model
after the last epoch is tested. The filter uses the LEO orbit settings.LEO_TRAIN_ORBIT
(A26; the true orbit by default).
"""
import json
import time

import torch

import settings as cfg
from data_io.data_cache import load_dataset
from navigation.ins_filter import STATE_SIZE
from navigation.masked_cla_network import FIXED_FEATURE_SIZE, MaskedCLANetwork
from navigation.navigation_filter import run_filter
import training_state

CHECKPOINT_FILE = cfg.OUTPUT_FOLDER / 'masked_cla_network.pt'


def training_step(network, optimizer, data, measurements):
    """One pass over the training dataset that updates all parameters."""
    optimizer.zero_grad()
    result = run_filter(data, measurements, network, fault_detection=False, training=True)
    (cfg.L2_WEIGHT * sum(torch.sum(p ** 2) for p in network.parameters())).backward()  # gamma ||Theta||^2, Eq. (32)
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

    classical = run_filter(data, measurements, fault_detection=False)  # traditional EKF, for comparison only
    network = MaskedCLANetwork(max_measurements)
    optimizer = torch.optim.Adam(network.parameters(), lr=cfg.LEARNING_RATE)

    info = {
        'experiment': cfg.EXPERIMENT, 'dataset': data.name, 'leo_train_orbit': cfg.LEO_TRAIN_ORBIT,
        'training_samples': len(data.fusion_times) - 1,
        'learning_rate': cfg.LEARNING_RATE, 'max_epochs': cfg.TRAINING_EPOCHS, 'l2_weight': cfg.L2_WEIGHT,
        'backprop_window': cfg.BACKPROP_WINDOW, 'optimization': 'joint: all parameters, 2 Adam steps per epoch',
        'max_measurements': max_measurements, 'input_size': FIXED_FEATURE_SIZE + 2 * max_measurements,
        'network': f'Conv1D {cfg.CONV_FILTERS}x{cfg.CONV_KERNEL_SIZE} -> max-pool {cfg.POOL_KERNEL_SIZE} -> '
                   f'LSTM {cfg.LSTM_LAYERS}x{cfg.LSTM_UNITS} (dropout {cfg.LSTM_DROPOUT}) -> attention -> '
                   f'FC {cfg.FC_HIDDEN_UNITS} -> gain {STATE_SIZE}x{max_measurements}',
        'trainable_parameters': sum(p.numel() for p in network.parameters()),
        'classical_ekf_training_position_rmse_m': classical['position_rmse_m'],
    }
    print(json.dumps(info, indent=1))

    optimizers = (optimizer,)
    first_epoch, history, info, time_before = training_state.resume(network, optimizers, info)
    for epoch in range(first_epoch, cfg.TRAINING_EPOCHS + 1):
        training_step(network, optimizer, data, measurements)
        train = training_step(network, optimizer, data, measurements)

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
