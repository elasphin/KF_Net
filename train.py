"""Offline training of the Masked CLA KalmanNet on the training dataset (paper Sec. II-C, Fig. 2).

    python train.py  ->  outputs/masked_cla_network.pt (model of the best validation loss, the one tested),
                         outputs/masked_cla_network_last.pt (model after the last epoch),
                         outputs/training_history.json, outputs/training_info.json
    python train.py --restart   (a new training even if a saved one exists)

An interrupted training continues after its last epoch when train.py runs again with the same
settings, code and data (train.py, outputs/training_state.pt).

Training of exp/paper with one change (branch exp/alternating): joint instead of alternating optimization.
As on every branch, a validation dataset is added (A21).
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
is no early stopping, gradient clipping or gain scale. Validation (A21, not in the paper):
after every epoch the network runs (no dropout, no gradient, no fault detection) on a third
SmartPNT-POS dataset (settings.VALIDATION_FOLDER_NAME), from the truth at its start, and the
model with the lowest validation loss (Eq. (30), as the training loss) is the one tested; all
TRAINING_EPOCHS epochs run and the model after the last epoch is kept too. The validation
run uses no random numbers, so the training itself is the same as without it. The filter
uses the LEO orbit settings.LEO_TRAIN_ORBIT (A26; the true orbit by default), also on the
validation dataset.

Resume an interrupted training (train.py), e.g. after the end of a Colab or Kaggle session.

After every epoch train.py keeps the whole training state in OUTPUT_FOLDER/training_state.pt: network,
optimizers, random generators, history, info and a fingerprint of everything the training depends on
(settings, training code, data cache keys of the training and validation datasets). Running train.py
again continues after the last saved epoch if the fingerprint is the same, so the result equals that of
an uninterrupted run; otherwise (settings, code or data changed) a new training starts. TRAINING_EPOCHS is
not in the fingerprint: raising it continues a finished training. python train.py --restart always starts a
new training.
"""
import functools
import hashlib
import json
import sys
import time

import torch

import settings as cfg
from dataset import cache_key, load_dataset
from navigation import FIXED_FEATURE_SIZE, STATE_SIZE, MaskedCLANetwork, run_filter


# ===== Resume of an interrupted training ==============================================================================
# Not in the fingerprint: the number of epochs, the run-time settings (same results) and the folders
# (names ending in _FOLDER, OUTPUT_ROOT: the state is in the output folder itself).
NOT_TRAINING_SETTINGS = {'TRAINING_EPOCHS', 'INS_MECHANIZATION', 'LEO_FORCE_MODEL', 'DATA_CACHE', 'OUTPUT_ROOT'}
TRAINING_CODE = ('train.py', 'navigation.py')


def training_state_file():
    return cfg.OUTPUT_FOLDER / 'training_state.pt'


def training_data_key():
    return cache_key('train') + cache_key('validation')


@functools.cache
def training_fingerprint():
    """Hash of the settings, the training code and the training and validation data (dataset keys)."""
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
    """True (with a message) if the saved training already has TRAINING_EPOCHS epochs."""
    state = saved_training_state()
    if state is None or state['epoch'] < cfg.TRAINING_EPOCHS:
        return False
    print(f"training already done ({state['epoch']} epochs, {training_state_file()}); python train.py --restart trains again")
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


def save_training_state(epoch, network, optimizers, history, info, rng=None):
    """Keep the state after 'epoch' (written to a temporary file first, so an interruption cannot spoil it)."""
    path = training_state_file()
    temporary = path.with_suffix('.tmp')
    torch.save({'fingerprint': training_fingerprint(), 'epoch': epoch, 'network': network.state_dict(),
                'optimizers': [optimizer.state_dict() for optimizer in optimizers],
                'torch_rng': torch.get_rng_state(), 'numpy_rng': None if rng is None else rng.bit_generator.state,
                'history': history, 'info': info}, temporary)
    temporary.replace(path)


# ===== Training =======================================================================================================
CHECKPOINT_FILE = cfg.OUTPUT_FOLDER / 'masked_cla_network.pt'              # best validation loss: tested
LAST_CHECKPOINT_FILE = cfg.OUTPUT_FOLDER / 'masked_cla_network_last.pt'    # after the last epoch


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
    if training_finished():
        return
    start_time = time.time()
    data, orbits = load_dataset('train')
    measurements = orbits[cfg.LEO_TRAIN_ORBIT]
    max_measurements = max(len(m) for m in measurements)             # N_max of Eq. (16)
    validation_data, validation_orbits = load_dataset('validation')  # A21; N_max of the training dataset (A16)
    validation_measurements = validation_orbits[cfg.LEO_TRAIN_ORBIT]

    classical = run_filter(data, measurements, fault_detection=False)  # traditional EKF, for comparison only
    network = MaskedCLANetwork(max_measurements)
    optimizer = torch.optim.Adam(network.parameters(), lr=cfg.LEARNING_RATE)

    info = {
        'experiment': cfg.EXPERIMENT, 'dataset': data.name, 'leo_train_orbit': cfg.LEO_TRAIN_ORBIT,
        'training_samples': len(data.fusion_times) - 1,
        'validation_dataset': validation_data.name, 'validation_samples': len(validation_data.fusion_times) - 1,
        'model_selection': 'lowest validation loss (A21)',
        'learning_rate': cfg.LEARNING_RATE, 'max_epochs': cfg.TRAINING_EPOCHS, 'l2_weight': cfg.L2_WEIGHT,
        'backprop_window': cfg.BACKPROP_WINDOW, 'optimization': 'joint: all parameters, 2 Adam steps per epoch',
        'max_measurements': max_measurements, 'input_size': FIXED_FEATURE_SIZE + 2 * max_measurements,
        'network': f'Conv1D {cfg.CONV_FILTERS}x{cfg.CONV_KERNEL_SIZE} -> max-pool {cfg.POOL_KERNEL_SIZE} -> '
                   f'LSTM {cfg.LSTM_LAYERS}x{cfg.LSTM_UNITS} (dropout {cfg.LSTM_DROPOUT}) -> attention -> '
                   f'FC {cfg.FC_HIDDEN_UNITS} -> gain {STATE_SIZE}x{max_measurements}',
        'trainable_parameters': sum(p.numel() for p in network.parameters()),
        'classical_ekf_training_position_rmse_m': classical['position_rmse_m'],
        'classical_ekf_validation_position_rmse_m': run_filter(
            validation_data, validation_measurements, fault_detection=False,
            max_measurements=max_measurements)['position_rmse_m'],
    }
    print(json.dumps(info, indent=1))

    optimizers = (optimizer,)
    first_epoch, history, info, time_before = resume_training(network, optimizers, info)
    best_loss = min((h['validation_loss'] for h in history), default=float('inf'))
    for epoch in range(first_epoch, cfg.TRAINING_EPOCHS + 1):
        training_step(network, optimizer, data, measurements)
        train = training_step(network, optimizer, data, measurements)
        validation = run_filter(validation_data, validation_measurements, network, fault_detection=False)   # A21

        history.append({'epoch': epoch, 'train_loss': train['loss'], 'train_position_rmse_m': train['position_rmse_m'],
                        'validation_loss': validation['loss'],
                        'validation_position_rmse_m': validation['position_rmse_m']})
        print(f"epoch {epoch:4d} | train loss {train['loss']:.4g} | train RMSE {train['position_rmse_m']:.3f} m | "
              f"validation loss {validation['loss']:.4g} | validation RMSE {validation['position_rmse_m']:.3f} m | "
              f"{time.time() - start_time:.0f} s")
        checkpoint = {'state_dict': network.state_dict(), 'max_measurements': max_measurements,
                      'leo_train_orbit': cfg.LEO_TRAIN_ORBIT, 'epoch': epoch}
        torch.save(checkpoint, LAST_CHECKPOINT_FILE)
        if validation['loss'] < best_loss or 'best_epoch' not in info:   # the model that is tested (A21)
            best_loss = validation['loss']
            torch.save(checkpoint, CHECKPOINT_FILE)
            info.update(best_epoch=epoch, best_validation_loss=validation['loss'],
                        best_validation_position_rmse_m=validation['position_rmse_m'],
                        best_epoch_train_position_rmse_m=train['position_rmse_m'])
        info.update(epochs_run=epoch, final_train_loss=train['loss'],
                    final_train_position_rmse_m=train['position_rmse_m'],
                    final_validation_loss=validation['loss'],
                    final_validation_position_rmse_m=validation['position_rmse_m'],
                    training_time_s=time_before + time.time() - start_time)
        (cfg.OUTPUT_FOLDER / 'training_history.json').write_text(json.dumps(history, indent=1))
        (cfg.OUTPUT_FOLDER / 'training_info.json').write_text(json.dumps(info, indent=1))
        save_training_state(epoch, network, optimizers, history, info)


if __name__ == '__main__':
    main()
