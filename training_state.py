"""Resume an interrupted training (train.py), e.g. after the end of a Colab or Kaggle session.

After every epoch train.py keeps the whole training state in OUTPUT_FOLDER/training_state.pt: network,
optimizers, random generators, history, info and a fingerprint of everything the training depends on
(settings, training code, data cache key). Running train.py again continues after the last saved epoch
if the fingerprint is the same, so the result equals that of an uninterrupted run; otherwise (settings,
code or data changed) a new training starts. TRAINING_EPOCHS is not in the fingerprint: raising it
continues a finished training. python train.py --restart always starts a new training.
"""
import functools
import hashlib
import sys

import torch

import settings as cfg
from data_io import data_cache

# Not in the fingerprint: the number of epochs, the run-time settings (same results) and the folders
# (names ending in _FOLDER, OUTPUT_ROOT: the state is in the output folder itself).
NOT_TRAINING_SETTINGS = {'TRAINING_EPOCHS', 'INS_MECHANIZATION', 'LEO_FORCE_MODEL', 'DATA_CACHE', 'OUTPUT_ROOT'}
TRAINING_CODE = ('train.py', 'navigation/masked_cla_network.py', 'navigation/navigation_filter.py',
                 'navigation/ins_filter.py')


def state_file():
    return cfg.OUTPUT_FOLDER / 'training_state.pt'


def data_key():
    return data_cache.cache_key('train')


@functools.cache
def fingerprint():
    """Hash of the settings, the training code and the training data (data_cache key)."""
    digest = hashlib.sha256(data_key().encode())
    for name, value in sorted(vars(cfg).items()):
        if name.isupper() and name not in NOT_TRAINING_SETTINGS and not name.endswith('_FOLDER'):
            digest.update(f'{name}={value!r}\n'.encode())
    for name in TRAINING_CODE:
        digest.update((cfg.PROJECT_FOLDER / name).read_bytes())
    return digest.hexdigest()[:16]


@functools.cache
def saved_state():
    """The saved state of the same settings, code and data, or None (none, other fingerprint, --restart);
    read once, at the start of train.py."""
    path = state_file()
    if '--restart' in sys.argv[1:] or not path.exists():
        return None
    state = torch.load(path)
    if state['fingerprint'] != fingerprint():
        print(f'{path} is from other settings, code or data: new training')
        return None
    return state


def finished():
    """True (with a message) if the saved training already has TRAINING_EPOCHS epochs."""
    state = saved_state()
    if state is None or state['epoch'] < cfg.TRAINING_EPOCHS:
        return False
    print(f"training already done ({state['epoch']} epochs, {state_file()}); python train.py --restart trains again")
    return True


def resume(network, optimizers, info, rng=None):
    """Restore the saved state into network, optimizers and random generators (torch and the NumPy rng).

    info: the info of this run (settings); the saved one adds what the training wrote (epochs_run, ...).
    Returns (first epoch to run, history, info, training time already spent [s]); for a new training
    (1, [], info, 0.0).
    """
    state = saved_state()
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
    print(f"resumed after epoch {state['epoch']} ({state_file()})")
    return state['epoch'] + 1, state['history'], info, saved['training_time_s']


def save(epoch, network, optimizers, history, info, rng=None):
    """Keep the state after 'epoch' (written to a temporary file first, so an interruption cannot spoil it)."""
    path = state_file()
    temporary = path.with_suffix('.tmp')
    torch.save({'fingerprint': fingerprint(), 'epoch': epoch, 'network': network.state_dict(),
                'optimizers': [optimizer.state_dict() for optimizer in optimizers],
                'torch_rng': torch.get_rng_state(), 'numpy_rng': None if rng is None else rng.bit_generator.state,
                'history': history, 'info': info}, temporary)
    temporary.replace(path)
