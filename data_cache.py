"""Read dataset and simulated measurements of 'train' or 'test', kept on disk between runs (settings.DATA_CACHE).

    from data_cache import load_dataset
    data, measurements = load_dataset('train')

Reading the RINEX/IMU/truth files, the GNSS satellite orbits and the LEO orbit integration and
simulation give the same result in every run (fixed seeds), so they are done once and kept in
OUTPUT_FOLDER/cache/<split>_<key>.pkl. The key is a hash of everything they depend on:
  - all settings except the network, training, integrity and run-time ones (NOT_DATA_SETTINGS),
  - the code that makes them (DATA_CODE files),
  - name, size and modification time of the input files (dataset folder, products, TLE files).
A change in any of these makes a new cache file (the old one of that split is removed). The LEO
orbit error variance of the filter R (A25) is not kept: it is set from leo_orbit_error_train.json
in every run, as before.
"""
import hashlib
import pickle
from pathlib import Path

import settings as cfg
from gnss_measurements import merge_measurements, prepare_gnss_measurements
from kaggle_download import find_dataset_folder
from leo_orbits import find_tle_folder
from leo_simulation import orbit_error_variance, real_error_bins, simulate_leo_measurements
from read_dataset import load_navigation_data

CODE_FOLDER = Path(__file__).resolve().parent
DATA_CODE = ('read_dataset.py', 'kaggle_download.py', 'earth_models.py', 'gnss_measurements.py', 'leo_orbits.py',
             'leo_simulation.py', 'numba_kernels.py', 'egm96_degree20.txt', 'data_cache.py')
NOT_DATA_SETTINGS = {
    'CONV_FILTERS', 'CONV_KERNEL_SIZE', 'POOL_KERNEL_SIZE', 'LSTM_UNITS', 'LSTM_LAYERS', 'LSTM_DROPOUT',
    'FC_HIDDEN_UNITS', 'MASK_EPSILON', 'RANDOM_SEED', 'LEARNING_RATE', 'TRAINING_EPOCHS', 'L2_WEIGHT',
    'BACKPROP_WINDOW', 'VALIDATION_FRACTION', 'EARLY_STOPPING_PATIENCE', 'GRADIENT_CLIP_NORM',
    'FALSE_ALARM_PROBABILITY', 'HORIZONTAL_PL_FACTOR', 'VERTICAL_PL_FACTOR', 'ALERT_LIMIT',
    'INS_MECHANIZATION', 'DATA_CACHE',
    'TRAIN_FOLDER_NAME', 'TEST_FOLDER_NAME',      # the folder of the split is in the key itself
}


def simulate_measurements(data, split):
    """GPS + BDS-3 (real) and LEO (simulated) measurements of every fusion epoch, LEO orbit variance not yet set.

    Returns (GNSS epochs, LEO epochs, range errors of the predicted LEO orbit).
    """
    gnss = prepare_gnss_measurements(data)
    leo, range_errors = simulate_leo_measurements(data, real_error_bins(data, gnss), cfg.LEO_NOISE_SEED[split])
    return gnss, leo, range_errors


def prepare_measurements(split, gnss, leo, range_errors):
    """Merged measurements of every fusion epoch with the LEO orbit error variance of the filter R."""
    variance = orbit_error_variance(range_errors, split)            # from the training dataset (A25)
    for meas in leo:
        meas.orbit_variance[:] = variance
    return [merge_measurements(g, l) for g, l in zip(gnss, leo)]


def input_files(folder):
    """Files read for one dataset folder: the folder itself, the products folder and the LEO TLE files."""
    files = [p for p in folder.iterdir() if p.is_file()]
    if cfg.PRODUCTS_FOLDER.is_dir():
        files += [p for p in cfg.PRODUCTS_FOLDER.iterdir() if p.is_file()]
    return sorted(files) + sorted(find_tle_folder().rglob('*.txt'))


def cache_key(split):
    folder_name = cfg.TRAIN_FOLDER_NAME if split == 'train' else cfg.TEST_FOLDER_NAME
    folder = find_dataset_folder(folder_name)
    digest = hashlib.sha256(f'{split} {folder_name}'.encode())
    for name, value in sorted(vars(cfg).items()):
        if name.isupper() and name not in NOT_DATA_SETTINGS:
            digest.update(f'{name}={value!r}\n'.encode())
    for name in DATA_CODE:
        digest.update((CODE_FOLDER / name).read_bytes())
    for path in input_files(folder):
        stat = path.stat()
        digest.update(f'{path} {stat.st_size} {stat.st_mtime_ns}\n'.encode())
    return digest.hexdigest()[:16]


def read_and_simulate(split):
    """(data, GNSS epochs, LEO epochs, LEO range errors): from the cache, or computed and then kept."""
    if not cfg.DATA_CACHE:
        data = load_navigation_data(split)
        return (data, *simulate_measurements(data, split))
    folder = cfg.OUTPUT_FOLDER / 'cache'
    path = folder / f'{split}_{cache_key(split)}.pkl'
    if path.exists():
        print(f'dataset and measurements read from {path}')
        with path.open('rb') as f:
            return pickle.load(f)
    data = load_navigation_data(split)
    result = (data, *simulate_measurements(data, split))
    folder.mkdir(parents=True, exist_ok=True)
    for old in folder.glob(f'{split}_*.pkl'):
        old.unlink()
    temporary = path.with_suffix('.tmp')
    with temporary.open('wb') as f:
        pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)
    print(f'dataset and measurements kept in {path}')
    return result


def load_dataset(split):
    """Dataset and merged GNSS + LEO measurements of every fusion epoch of 'train' or 'test'."""
    data, gnss, leo, range_errors = read_and_simulate(split)
    return data, prepare_measurements(split, gnss, leo, range_errors)
