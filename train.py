"""Offline training of the Masked CLA KalmanNet on the training dataset (paper Sec. II-C, Fig. 2).

    python train.py  ->  outputs/masked_cla_network.pt (best validation model),
                         outputs/training_history.json, outputs/training_info.json
    python train.py --restart   (a new training even if a saved one exists)

An interrupted training continues after its last epoch when train.py runs again with the same
settings, code and data (train.py, outputs/training_state.pt); one ended by early
stopping is done.

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

Resume an interrupted training (train.py), e.g. after the end of a Colab or Kaggle session.

After every epoch train.py keeps the whole training state in OUTPUT_FOLDER/training_state.pt: network,
optimizers, random generators, history, info and a fingerprint of everything the training depends on
(settings, training code, data cache key). Running train.py again continues after the last saved epoch
if the fingerprint is the same, so the result equals that of an uninterrupted run; otherwise (settings,
code or data changed) a new training starts. TRAINING_EPOCHS is not in the fingerprint: raising it
continues a training that ran all its epochs (not one ended by early stopping).
python train.py --restart always starts a new training.
"""
import functools
import hashlib
import json
import sys
import time

import numpy as np
import torch

import settings as cfg
from dataset import cache_key, load_dataset, training_split
from navigation import FIXED_FEATURE_SIZE, STATE_SIZE, MaskedCLANetwork, run_filter


# ===== Resume of an interrupted training ==============================================================================
# Not in the fingerprint: the number of epochs, the run-time settings (same results) and the folders
# (names ending in _FOLDER, OUTPUT_ROOT: the state is in the output folder itself).
NOT_TRAINING_SETTINGS = {'TRAINING_EPOCHS', 'INS_MECHANIZATION', 'LEO_FORCE_MODEL', 'DATA_CACHE', 'OUTPUT_ROOT'}
TRAINING_CODE = ('train.py', 'navigation.py')


def training_state_file():
    return cfg.OUTPUT_FOLDER / 'training_state.pt'


def training_data_key():
    return cache_key('train')


@functools.cache
def training_fingerprint():
    """Hash of the settings, the training code and the training data (dataset key)."""
    digest = hashlib.sha256(training_data_key().encode())
    for name, value in sorted(vars(cfg).items()):
        if name.isupper() and name not in NOT_TRAINING_SETTINGS and not name.endswith('_FOLDER'):
            digest.update(f'{name}={value!r}\n'.encode())
    for name in TRAINING_CODE:
        digest.update((cfg.PROJECT_FOLDER / name).read_bytes())
    return digest.hexdigest()[:16]


@functools.cache
def saved_training_state():
    """The saved state of the same settings, code and data, or None (none, other fingerprint, --restart);
    read once, at the start of train.py."""
    path = training_state_file()
    if '--restart' in sys.argv[1:] or not path.exists():
        return None
    state = torch.load(path)
    if state['fingerprint'] != training_fingerprint():
        print(f'{path} is from other settings, code or data: new training')
        return None
    return state


def training_finished():
    """True (with a message) if the saved training already has TRAINING_EPOCHS epochs or stopped early."""
    state = saved_training_state()
    if state is None or state['epoch'] < cfg.TRAINING_EPOCHS and not state.get('stopped', False):
        return False
    print(f"training already done ({state['epoch']} epochs{', early stopping' if state.get('stopped') else ''}, "
          f"{training_state_file()}); python train.py --restart trains again")
    return True


def resume_training(network, optimizers, info, rng=None):
    """Restore the saved state into network, optimizers and random generators (torch and the NumPy rng).

    info: the info of this run (settings); the saved one adds what the training wrote (epochs_run, ...).
    Returns (first epoch to run, history, info, training time already spent [s]); for a new training
    (1, [], info, 0.0).
    """
    state = saved_training_state()
    if state is None:
        return 1, [], info, 0.0
    network.load_state_dict(state['network'])
    for optimizer, optimizer_state in zip(optimizers, state['optimizers']):
        optimizer.load_state_dict(optimizer_state)
    torch.set_rng_state(state['torch_rng'])
    if rng is not None:
        rng.bit_generator.state = state['numpy_rng']
    saved = state['info']
    info = {**saved, **info, 'resumed_after_epochs': saved.get('resumed_after_epochs', []) + [state['epoch']]}
    print(f"resumed after epoch {state['epoch']} ({training_state_file()})")
    return state['epoch'] + 1, state['history'], info, saved['training_time_s']


def save_training_state(epoch, network, optimizers, history, info, rng=None, stopped=False):
    """Keep the state after 'epoch' (written to a temporary file first, so an interruption cannot spoil it);
    stopped: the training ended by early stopping."""
    path = training_state_file()
    temporary = path.with_suffix('.tmp')
    torch.save({'fingerprint': training_fingerprint(), 'epoch': epoch, 'network': network.state_dict(),
                'optimizers': [optimizer.state_dict() for optimizer in optimizers],
                'torch_rng': torch.get_rng_state(), 'numpy_rng': None if rng is None else rng.bit_generator.state,
                'history': history, 'info': info, 'stopped': stopped}, temporary)
    temporary.replace(path)


# ===== Training =======================================================================================================
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
    if training_finished():
        return
    start_time = time.time()
    data, orbits = load_dataset('train')
    measurements = orbits[cfg.LEO_TRAIN_ORBIT]
    last = len(data.fusion_times) - 1
    split = training_split(len(data.fusion_times))                  # train: 0..split, validation: split..last
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
        'experiment': cfg.EXPERIMENT, 'dataset': data.name, 'leo_train_orbit': cfg.LEO_TRAIN_ORBIT,
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

    optimizers = (filter_optimizer, encoder_optimizer)
    first_epoch, history, info, time_before = resume_training(network, optimizers, info)
    losses = [h['validation_loss'] for h in history]                 # early stopping state of a resumed training
    best_loss = min(losses, default=np.inf)
    epochs_without_improvement = len(losses) - 1 - losses.index(best_loss) if losses else 0
    for epoch in range(first_epoch, cfg.TRAINING_EPOCHS + 1):
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
        stopped = epochs_without_improvement >= cfg.EARLY_STOPPING_PATIENCE
        info.update(epochs_run=epoch, final_train_loss=train['loss'],
                    final_train_position_rmse_m=train['position_rmse_m'],
                    training_time_s=time_before + time.time() - start_time)
        (cfg.OUTPUT_FOLDER / 'training_history.json').write_text(json.dumps(history, indent=1))
        (cfg.OUTPUT_FOLDER / 'training_info.json').write_text(json.dumps(info, indent=1))
        save_training_state(epoch, network, optimizers, history, info, stopped=stopped)
        if stopped:
            print(f'early stopping: no better validation loss for {cfg.EARLY_STOPPING_PATIENCE} epochs')
            break


if __name__ == '__main__':
    main()
