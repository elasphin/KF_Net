"""Offline training of the Masked CLA KalmanNet on the training dataset (paper Sec. II-C, Fig. 2).

    python train.py  ->  outputs/masked_cla_network.pt (model after the last epoch),
                         outputs/training_history.json, outputs/training_info.json
    python train.py --restart   (a new training even if a saved one exists)

An interrupted training continues after its last epoch when train.py runs again with the same
settings, code and data (train.py, outputs/training_state.pt).

Training of exp/paper with one change (branch exp/input-norm): the network input X_k is normalized,
settings.INPUT_NORMALIZATION = 'l2' (each feature group to unit norm, as KalmanNet) or 'zscore' (mean and
std of the features of a traditional EKF run on the training dataset; navigation.InputNormalization).
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
is no validation, early stopping, gradient clipping or gain scale, and the model
after the last epoch is tested. The filter uses the LEO orbit settings.LEO_TRAIN_ORBIT
(A26; the true orbit by default).

Resume an interrupted training (train.py), e.g. after the end of a Colab or Kaggle session.

After every epoch train.py keeps the whole training state in OUTPUT_FOLDER/training_state.pt: network,
optimizers, random generators, history, info and a fingerprint of everything the training depends on
(settings, training code, data cache key). Running train.py again continues after the last saved epoch
if the fingerprint is the same, so the result equals that of an uninterrupted run; otherwise (settings,
code or data changed) a new training starts. TRAINING_EPOCHS is not in the fingerprint: raising it
continues a finished training. python train.py --restart always starts a new training.
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
    if training_finished():
        return
    start_time = time.time()
    data, orbits = load_dataset('train')
    measurements = orbits[cfg.LEO_TRAIN_ORBIT]
    max_measurements = max(len(m) for m in measurements)             # N_max of Eq. (16)

    # Traditional EKF, for comparison and for the statistics of the z-score input normalization.
    input_samples = []
    classical = run_filter(data, measurements, fault_detection=False, input_samples=input_samples)
    network = MaskedCLANetwork(max_measurements)
    if cfg.INPUT_NORMALIZATION == 'zscore':
        network.input_normalization.set_statistics(input_samples)
    del input_samples
    encoder = [network.conv.weight, network.conv_bias]                  # psi of Ref. [15]: masked CNN
    filter_part = [p for p in network.parameters() if not any(p is q for q in encoder)]   # theta: LSTM, attention, FC
    encoder_optimizer = torch.optim.Adam(encoder, lr=cfg.LEARNING_RATE)
    filter_optimizer = torch.optim.Adam(filter_part, lr=cfg.LEARNING_RATE)

    info = {
        'experiment': cfg.EXPERIMENT, 'dataset': data.name, 'leo_train_orbit': cfg.LEO_TRAIN_ORBIT,
        'training_samples': len(data.fusion_times) - 1,
        'learning_rate': cfg.LEARNING_RATE, 'max_epochs': cfg.TRAINING_EPOCHS, 'l2_weight': cfg.L2_WEIGHT,
        'backprop_window': cfg.BACKPROP_WINDOW, 'input_normalization': cfg.INPUT_NORMALIZATION, 'optimization': 'alternating: LSTM-attention-FC, then CNN [15]',
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
        save_training_state(epoch, network, optimizers, history, info)


if __name__ == '__main__':
    main()
