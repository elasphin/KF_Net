"""Offline training of the Masked CLA KalmanNet on the training dataset (paper Sec. II-C, Fig. 2).

    python train.py  ->  outputs/masked_cla_network.pt (model of the best validation loss, the one tested),
                         outputs/masked_cla_network_last.pt (model after the last epoch),
                         outputs/training_history.json, outputs/training_info.json
    python train.py --restart   (a new training even if a saved one exists)

An interrupted training continues after its last epoch when train.py runs again with the same
settings, code and data (train.py, outputs/training_state.pt).

Training of main with one change (branch exp/bptt-kalmannet): back-propagation through time
as in KalmanNet [14] Sec. III-D instead of the single-step gradient of Eq. (31).
V2 (truncated BPTT): the training part is divided into consecutive sub-trajectories of T fusion
epochs, each filtered from the truth with a new LSTM state (A7); they are shuffled every epoch and
grouped into mini-batches of M; the gradient goes through the whole sub-trajectory (filter
linearization of navigation.run_filter, LSTM state and network inputs; BACKPROP_WINDOW = None)
and one Adam step follows each mini-batch, with the loss averaged over its sub-trajectories (Ref. [14]
Eq. (14)). V1 (the whole trajectory at once) does not fit in memory, so the warm-up with T = 100 is
followed by a fine-tuning with T = 1000 (settings.SUBTRAJECTORY_LENGTHS, WARMUP_EPOCHS).
As on every branch, a validation part is added (A21).
Loss: paper Eq. (30) ||x_k - x_hat_k||^2 on the position, velocity and attitude of the post-processed truth
(A28; the paper uses the position only), each error divided by its scale settings.LOSS_SCALES,
averaged over the epochs and the nine components (MSE, Table III) plus
gamma ||Theta||^2 (Eq. (32)). Adam with learning rate 0.01 (Table III) for 480 epochs (Fig. 15).
Alternating optimization (paper Sec. II-B, Ref. [15] Algorithm 2): in every
epoch the filter part theta (LSTM, attention, FC) is updated over all mini-batches with the encoder
psi (masked CNN) frozen, then psi over the same mini-batches with theta frozen (A15). The training
dataset trains the network (paper Sec. III) except its last settings.VALIDATION_FRACTION of fusion
epochs; there is no early stopping, gradient clipping or gain scale (the network gives K directly, as
in main). The training loss and RMSE are those of the sub-trajectories (each starts from the truth).
Validation (A21, not in the paper): after every epoch the network runs (no dropout, no gradient, no
fault detection) on that last part of the training dataset, whole (not in sub-trajectories, as the
test), from the truth at its first epoch, and the model with the lowest validation loss (Eq. (30), as
the training loss) is the one tested; all TRAINING_EPOCHS epochs run and the model after the last
epoch is kept too. The validation run uses no random numbers, so the training itself is the same as
without it. The filter uses the LEO orbit settings.LEO_TRAIN_ORBIT (A26; the true orbit by default),
also on the validation part.

Resume an interrupted training (train.py), e.g. after the end of a Colab or Kaggle session.

After every epoch train.py keeps the whole training state in OUTPUT_FOLDER/training_state.pt: network,
optimizers, random generators, history, info and a fingerprint of everything the training depends on
(settings, training code, data cache key of the training dataset). Running train.py
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

import numpy as np
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


@functools.cache
def training_fingerprint():
    """Hash of the settings, the training code and the training data (dataset key; the validation part is in it)."""
    digest = hashlib.sha256(cache_key('train').encode())
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


def validation_start(data):
    """First fusion epoch of the validation part (A21): the last VALIDATION_FRACTION of the training dataset.
    The training runs over fusion epochs 0..start, the validation over start..last, from the truth at start."""
    last = len(data.fusion_times) - 1
    start = last - round(cfg.VALIDATION_FRACTION * last)
    if not 0 < start < last:
        raise ValueError(f'VALIDATION_FRACTION = {cfg.VALIDATION_FRACTION} leaves no training or no validation '
                         f'epochs of the {last} fusion epochs of {data.name}')
    return start


def subtrajectories(last, length):
    """(first, last) of the consecutive sub-trajectories of 'length' fusion epochs over 0..last (KalmanNet V2)."""
    return [(first, min(first + length, last)) for first in range(0, last, length)]


def training_pass(network, optimizer, parameters, data, measurements, batches):
    """One pass over the mini-batches of sub-trajectories that updates only 'parameters' (the others are frozen):
    one Adam step per mini-batch, its loss averaged over the sub-trajectories (Ref. [14] Eq. (14)).
    Returns the loss and the position RMSE of the pass."""
    for p in network.parameters():
        p.requires_grad_(any(p is q for q in parameters))
    loss_sum, square_error_sum, epochs = 0.0, 0.0, 0
    for batch in batches:
        optimizer.zero_grad()
        for first, last in batch:
            result = run_filter(data, measurements, network, first, last, fault_detection=False, training=True)
            loss_sum += result['loss'] * (last - first)
            square_error_sum += result['position_rmse_m'] ** 2 * (last - first)
            epochs += last - first
        for p in parameters:                                                    # mean over the mini-batch
            if p.grad is not None:
                p.grad /= len(batch)
        (cfg.L2_WEIGHT * sum(torch.sum(p ** 2) for p in parameters)).backward()      # gamma ||Theta||^2, Eq. (32)
        optimizer.step()
    return {'loss': loss_sum / epochs, 'position_rmse_m': float(np.sqrt(square_error_sum / epochs))}


def main():
    torch.manual_seed(cfg.RANDOM_SEED)
    rng = np.random.default_rng(cfg.RANDOM_SEED)
    cfg.OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    if training_finished():
        return
    start_time = time.time()
    data, orbits = load_dataset('train')
    measurements = orbits[cfg.LEO_TRAIN_ORBIT]
    split, last = validation_start(data), len(data.fusion_times) - 1  # training 0..split, validation split..last (A21)
    max_measurements = max(len(m) for m in measurements[:split + 1])  # N_max of Eq. (16), training part only

    classical = run_filter(data, measurements, last=split, fault_detection=False)  # traditional EKF, comparison only
    network = MaskedCLANetwork(max_measurements)
    encoder = [network.conv.weight, network.conv_bias]                  # psi of Ref. [15]: masked CNN
    filter_part = [p for p in network.parameters() if not any(p is q for q in encoder)]   # theta: LSTM, attention, FC
    encoder_optimizer = torch.optim.Adam(encoder, lr=cfg.LEARNING_RATE)
    filter_optimizer = torch.optim.Adam(filter_part, lr=cfg.LEARNING_RATE)

    info = {
        'experiment': cfg.EXPERIMENT, 'dataset': data.name, 'leo_train_orbit': cfg.LEO_TRAIN_ORBIT,
        'training_samples': split, 'training_fusion_epochs': f'0..{split}',
        'validation_dataset': data.name, 'validation_fraction': cfg.VALIDATION_FRACTION,
        'validation_samples': last - split, 'validation_fusion_epochs': f'{split}..{last}',
        'model_selection': 'lowest validation loss (A21)',
        'learning_rate': cfg.LEARNING_RATE, 'max_epochs': cfg.TRAINING_EPOCHS, 'l2_weight': cfg.L2_WEIGHT,
        'backprop_window': cfg.BACKPROP_WINDOW, 'optimization': 'alternating: LSTM-attention-FC, then CNN [15]',
        'bptt': f'KalmanNet V2: sub-trajectories of {cfg.SUBTRAJECTORY_LENGTHS[0]} epochs for {cfg.WARMUP_EPOCHS} '
                f'epochs, then of {cfg.SUBTRAJECTORY_LENGTHS[1]}; {cfg.BPTT_BATCH_SIZE} per Adam step, shuffled',
        'max_measurements': max_measurements, 'input_size': FIXED_FEATURE_SIZE + 2 * max_measurements,
        'network': f'Conv1D {cfg.CONV_FILTERS}x{cfg.CONV_KERNEL_SIZE} -> max-pool {cfg.POOL_KERNEL_SIZE} -> '
                   f'LSTM {cfg.LSTM_LAYERS}x{cfg.LSTM_UNITS} (dropout {cfg.LSTM_DROPOUT}) -> attention -> '
                   f'FC {cfg.FC_HIDDEN_UNITS} -> gain {STATE_SIZE}x{max_measurements}',
        'trainable_parameters': sum(p.numel() for p in network.parameters()),
        'classical_ekf_training_position_rmse_m': classical['position_rmse_m'],
        'classical_ekf_validation_position_rmse_m': run_filter(
            data, measurements, first=split, fault_detection=False,
            max_measurements=max_measurements)['position_rmse_m'],
    }
    print(json.dumps(info, indent=1))

    optimizers = (filter_optimizer, encoder_optimizer)
    first_epoch, history, info, time_before = resume_training(network, optimizers, info, rng)
    best_loss = min((h['validation_loss'] for h in history), default=float('inf'))
    for epoch in range(first_epoch, cfg.TRAINING_EPOCHS + 1):
        length = cfg.SUBTRAJECTORY_LENGTHS[0 if epoch <= cfg.WARMUP_EPOCHS else 1]
        pieces = subtrajectories(split, length)                          # training part only (A21)
        order = rng.permutation(len(pieces))                             # shuffled every epoch (V2)
        batches = [[pieces[i] for i in order[j:j + cfg.BPTT_BATCH_SIZE]]
                   for j in range(0, len(order), cfg.BPTT_BATCH_SIZE)]
        training_pass(network, filter_optimizer, filter_part, data, measurements, batches)   # theta, psi frozen
        train = training_pass(network, encoder_optimizer, encoder, data, measurements, batches)  # psi, theta frozen
        validation = run_filter(data, measurements, network, first=split, fault_detection=False)   # A21

        history.append({'epoch': epoch, 'subtrajectory_length': length, 'train_loss': train['loss'],
                        'train_position_rmse_m': train['position_rmse_m'], 'validation_loss': validation['loss'],
                        'validation_position_rmse_m': validation['position_rmse_m']})
        print(f"epoch {epoch:4d} | T {length:4d} | train loss {train['loss']:.4g} | "
              f"train RMSE {train['position_rmse_m']:.3f} m | validation loss {validation['loss']:.4g} | "
              f"validation RMSE {validation['position_rmse_m']:.3f} m | {time.time() - start_time:.0f} s")
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
        save_training_state(epoch, network, optimizers, history, info, rng)


if __name__ == '__main__':
    main()
