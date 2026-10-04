"""All settings of the Masked-KalmanNet GNSS/LEO/INS simulation.

Paper: J. Yan et al., "A Robust Position Approach Based on Masked KalmanNet for
GNSS/LEO/INS Integrated Navigation System", IEEE IoT Journal, vol. 13, no. 11, 2026.

Every value carries a source tag:
    [paper ...]   value given in the paper (equation, table, figure or text)
    [ref N]       value given in reference [N] of the paper
    [choice AX]   not given by the paper; see item AX in docs/ASSUMPTIONS.md

python settings.py shows the dataset and output folders of the current environment.
"""
import os
from pathlib import Path

import numpy as np

# --- Dataset and output folders in Colab, on Kaggle or on my computer --------
PROJECT_FOLDER = Path(__file__).resolve().parent

# Colab: Google Drive folder "My Drive/Dataset" (or "My Drive/dataset": Drive names are case-sensitive)
# https://drive.google.com/drive/folders/1npnGKO7qwgKPvfoKTclzeA59wfpm860g
# Outputs also go to Drive, so they are kept after the Colab runtime ends.
COLAB_FOLDER = Path('/content/drive/MyDrive/Dataset')
COLAB_OUTPUT_FOLDER = Path('/content/drive/MyDrive/KF_Net_outputs')

# Kaggle: dataset elasphin/dataset (title "Dataset") attached to the notebook ("Add Input"); all of /kaggle/input
# is searched if this folder is missing. Outputs go to /kaggle/working, the only writable folder that is kept
# with the notebook version.
KAGGLE_FOLDER = Path('/kaggle/input/datasets/elasphin/dataset')
KAGGLE_OUTPUT_FOLDER = Path('/kaggle/working/outputs')

# My computer: the Dataset folder next to this file, else the SmartPNT-POS Dataset folder of my computer;
# the outputs folder next to this file
LOCAL_FOLDER = PROJECT_FOLDER / 'Dataset'
MY_COMPUTER_FOLDER = Path(r'C:\Users\elasp\Desktop\MSc\MSc Project\SmartPNT-Pos\Dataset')
LOCAL_OUTPUT_FOLDER = PROJECT_FOLDER / 'outputs'


def first_folder(*folders: Path) -> Path:
    """The first of the folders that exists (the first one if none does, for the error messages)."""
    return next((folder for folder in folders if folder.is_dir()), folders[0])


def get_folders() -> tuple[Path, Path]:
    """(dataset folder, output folder) of the current environment.

    The dataset folder holds the data folders (Data01_..., Data02_...), LEO_TLE, IMUErrorModel.txt and
    the products, at any depth (they are searched in the whole folder); the same structure in Colab, on Kaggle and
    on my computer.
    """
    # By the environment variables, not the folders: Kaggle also has google.colab installed and Colab can have a
    # /kaggle folder.
    if 'KAGGLE_KERNEL_RUN_TYPE' in os.environ:    # Kaggle
        return first_folder(KAGGLE_FOLDER, Path('/kaggle/input')), KAGGLE_OUTPUT_FOLDER
    if 'COLAB_RELEASE_TAG' in os.environ:          # Colab
        if not COLAB_FOLDER.parent.is_dir():       # Drive not mounted yet
            from google.colab import drive
            drive.mount('/content/drive')
        return first_folder(COLAB_FOLDER, COLAB_FOLDER.with_name('dataset')), COLAB_OUTPUT_FOLDER
    return first_folder(LOCAL_FOLDER, MY_COMPUTER_FOLDER), LOCAL_OUTPUT_FOLDER


DATASET_FOLDER, OUTPUT_ROOT = get_folders()
# Training setup of this branch (exp/<EXPERIMENT>): its outputs and its data cache go to OUTPUT_ROOT/EXPERIMENT,
# so the runs of the experiment branches do not overwrite each other.
EXPERIMENT = 'alternating'          # exp/paper with joint instead of alternating optimization
OUTPUT_FOLDER = OUTPUT_ROOT / EXPERIMENT

# --- Dataset (paper Sec. III: SmartPNT-POS, Kaggle) --------------------------
KAGGLE_DATASET = 'elasphin/dataset'     # my copy of SmartPNT-POS (+ Data11, LEO_TLE), for kagglehub [paper Sec. III]
PRODUCTS_FOLDER = DATASET_FOLDER / 'products'   # SP3, CLK, brdm if not in the data folder; then DATASET_FOLDER itself
TRAIN_FOLDER_NAME = 'Data01_20230102_ISA-100C_Vehicle_Complex'  # training set (~150 min, Table I)
TEST_FOLDER_NAME = 'Data02_20220309_ISA-100C_Vehicle_Complex'   # testing set (~110 min, Table I)
# Validation set: the last VALIDATION_FRACTION of the fusion epochs of the training dataset (of its first
# MAX_FUSION_EPOCHS['train']); the network trains on the epochs before it. The paper has no validation. It only
# selects the model that is tested (best validation loss); it never trains it.  [choice A21]
VALIDATION_FRACTION = 0.2
# First fusion epochs of each dataset; None = all (paper)
MAX_FUSION_EPOCHS = {'train': None, 'test': None}
RANDOM_SEED = 0

# --- IMU error model (paper Table II) ----------------------------------------
ACCEL_BIAS_INSTABILITY = 0.01e-3 * 9.80665               # 0.01 mg -> m/s^2      [paper Table II]
ACCEL_WHITE_NOISE = 0.03 / 60.0                          # 0.03 m/s/sqrt(h)       [paper Table II]
GYRO_BIAS_INSTABILITY = np.deg2rad(0.01) / 3600.0        # 0.01 deg/h -> rad/s    [paper Table II]
GYRO_WHITE_NOISE = np.deg2rad(0.03) / 60.0               # 0.03 deg/sqrt(h)       [paper Table II]
BIAS_CORRELATION_TIME = 3600.0     # s, bias random-walk density 2*BS^2/tau       [choice A6]
INITIAL_POSITION_STD = 0.1         # m, accuracy of the post-processed truth      [paper Sec. III-A]
INITIAL_VELOCITY_STD = 0.01        # m/s                                          [choice A7]
INITIAL_ATTITUDE_STD = np.deg2rad(0.01)  # rad                                    [choice A7]
# Initial bias standard deviations (turn-on bias, not the in-run instability BS of Table II): ISDV_AccelBias and
# ISDV_GyrosBias of the dataset's IMU error model file for the IMU of README.xml (ISA-100C: 0.02 m/s^2 ~ 2 mg,
# 9e-5 deg/s = 0.32 deg/h). The static start of Data01 shows ~3.2 mg on one accelerometer (check_dataset.py).
IMU_ERROR_MODEL_FILE_NAME = 'IMUErrorModel.txt'  # in the data folder, a folder above it or DATASET_FOLDER [choice A7]

# --- GNSS (paper Table II, Eq. (3)-(4), Ref. [35], [37]) ---------------------
BDS3_FIRST_PRN = 19                # BDS-3 satellites are C19 and above           [paper Table II]
GNSS_ELEVATION_MASK_DEG = 10.0     # deg, GPS/BDS-3 rows used (GINav [44] default) [choice A10]
URA_STD = 1.5                      # m, [ref 35]: "commonly 1-2 m"                [choice A9]
IONO_VERTICAL_STD = ((20.0, 9.0), (55.0, 4.5), (90.0, 6.0))  # (|lat| limit deg, m) [ref 35 Eq. 16]
IONO_MEAN_HEIGHT = 350e3           # m                                            [ref 35]
EARTH_MEAN_RADIUS = 6378.14e3      # m                                            [ref 35]
TROPO_ZENITH_STD = 0.12            # m                                            [ref 35 Eq. 17]
GOGPS_A, GOGPS_a, GOGPS_S0, GOGPS_S1 = 30.0, 20.0, 10.0, 50.0  # Eq. (4)          [ref 37]
DLL_BANDWIDTH = 2.0                # Hz                                           [ref 35]
DLL_CORRELATOR_SPACING = 0.1       # chip                                         [ref 35]
DLL_INTEGRATION_TIME = 0.02        # s                                            [ref 35]

# --- LEO signals of opportunity (paper Sec. II-A; docs/ASSUMPTIONS.md A1-A4, A24-A26) ---
LEO_TLE_FOLDER_NAME = 'LEO_TLE'    # TLE files (*.txt) of the LEO satellites, inside the dataset folder [choice A1]
LEO_TLE_MIN_AGE = 0.0              # h, predictor uses the latest TLE published this long before the start [choice A1]
LEO_MAX_TLE_MISMATCH = 10e3        # m, predictor vs reference TLE at the reference epoch; larger = manoeuvre,
                                   # satellite left out                                                [choice A1]
LEO_GRAVITY_DEGREE = 20            # EGM96 degree and order of the reference orbit                    [choice A1]
LEO_MIN_PERIGEE_ALTITUDE = 700e3   # m, lower satellites (e.g. still raising their orbit) are left out:
                                   # drag is not in the reference orbit                                 [choice A1]
LIGHT_TIME_THRESHOLD = 1e-3        # m, epsilon_T of Eq. (5) step 3               [choice A1]
LEO_ELEVATION_MASK_DEG = 10.0      # deg                                          [choice A2]
LEO_ELEVATION_BIN_DEG = 10.0       # deg, bins for the mean C/N0                   [choice A3]
LEO_NOISE_DOF = 5.0                # Student-t degrees of freedom of the LEO MP/NLOS noise (> 2; large = Gaussian) [choice A3]
LEO_CODE_CHIP_RATE = 1.023e6       # chip/s, used by the receiver-noise term      [choice A3]
LEO_NOISE_SEED = {'train': 1, 'test': 2}
# LEO orbit used by the filter (A26): 'reference' = true orbit (the one the measurements are made with),
# 'tle' = SGP4 of the predictor TLE, 'network' = leo_pseudorange.network_orbit (orbit predicted by a neural
# network, to be written). The network is trained once with LEO_TRAIN_ORBIT and tested with every orbit of
# LEO_TEST_ORBITS on the same measurements; R is the same in all of them (orbit variance of LEO_TRAIN_ORBIT, A25).
LEO_TRAIN_ORBIT = 'reference'                       # [choice A26]
LEO_TEST_ORBITS = ('reference', 'tle')              # add 'network' when network_orbit exists    [choice A26]

# --- Masked CLA network (paper Table III, Fig. 8) ----------------------------
CONV_FILTERS = 24                  # [paper Table III]
CONV_KERNEL_SIZE = 3               # [paper Table III]
POOL_KERNEL_SIZE = 3               # pooling of Fig. 8, stride 1 so Eq. (23) holds [choice A12]
LSTM_UNITS = 64                    # [paper Table III]
LSTM_LAYERS = 5                    # [paper Table III]
LSTM_DROPOUT = 0.2                 # [paper Table III]
FC_HIDDEN_UNITS = 64               # hidden layer of the masked FC in Fig. 8      [choice A12]
MASK_EPSILON = 1e-6                # epsilon of Eq. (22)                          [paper Eq. (22)]

# --- Training (paper Sec. II-C, Table III, Fig. 15) --------------------------
# As in the paper: the training dataset trains the network for TRAINING_EPOCHS epochs, no early stopping,
# gradient clipping or gain scale. Added (A21): its last VALIDATION_FRACTION is kept out of the training; after
# every epoch the network runs on it and the model with the lowest validation loss is the one tested (the model
# after the last epoch is kept as well).
LEARNING_RATE = 0.01               # Adam                                         [paper Table III, Sec. III-B]
TRAINING_EPOCHS = 480              # learning curves of Fig. 15 run to ~480 epochs [paper Fig. 15]
L2_WEIGHT = 1e-5                   # gamma of Eq. (32)                            [choice A14]
BACKPROP_WINDOW = 1                # single-step gradient of Eq. (31): state and LSTM state detached every epoch [paper Eq. (31)]

# --- Fault detection and integrity (paper Sec. II-D, Fig. 20) ----------------
FALSE_ALARM_PROBABILITY = 1e-3     # alpha of the chi-square test                 [ref 33]
HORIZONTAL_PL_FACTOR = 6.0         # K_H                                          [choice A17]
VERTICAL_PL_FACTOR = 5.33          # K_V                                          [choice A17]
ALERT_LIMIT = 30.0                 # m, Stanford diagrams                         [paper Fig. 20]

# --- Run time only (the results do not change; README.md "Run time") --------
INS_MECHANIZATION = 'numba'        # 'python' (navigation.mechanize loop) or 'numba' (same arithmetic, compiled)
LEO_FORCE_MODEL = 'numba'          # 'python' (leo_pseudorange.equations_of_motion) or 'numba' (same, compiled)
DATA_CACHE = True                  # keep the read dataset and the simulated measurements in OUTPUT_FOLDER/cache

if __name__ == '__main__':
    print('Dataset folder:', DATASET_FOLDER, '' if DATASET_FOLDER.is_dir() else '(missing)')
    for path in sorted(DATASET_FOLDER.iterdir()) if DATASET_FOLDER.is_dir() else ():
        print('  ', path.name)
    print('Output folder: ', OUTPUT_FOLDER)
