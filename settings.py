"""All settings of the Masked-KalmanNet GNSS/LEO/INS simulation.

Paper: J. Yan et al., "A Robust Position Approach Based on Masked KalmanNet for
GNSS/LEO/INS Integrated Navigation System", IEEE IoT Journal, vol. 13, no. 11, 2026.

Every value carries a source tag:
    [paper ...]   value given in the paper (equation, table, figure or text)
    [ref N]       value given in reference [N] of the paper
    [choice AX]   not given by the paper; see item AX in ASSUMPTIONS.md
"""
from pathlib import Path

import numpy as np

from dataset_path import DATASET_FOLDER

PROJECT_FOLDER = Path(__file__).resolve().parent

# --- Dataset (paper Sec. III: SmartPNT-POS, Kaggle) --------------------------
KAGGLE_DATASET = 'fengzhusgg/smartpnt-pos'                      # [paper Sec. III]
# DATASET_FOLDER (dataset_path.py): Google Drive in Colab, /kaggle/input on Kaggle, ./Dataset on my computer
PRODUCTS_FOLDER = DATASET_FOLDER / 'products'                   # SP3, CLK and brdm files if not in the data folder
TRAIN_FOLDER_NAME = 'Data01_20230102_ISA-100C_Vehicle_Complex'  # training set (~150 min, Table I)
TEST_FOLDER_NAME = 'Data02_20220309_ISA-100C_Vehicle_Complex'   # testing set (~110 min, Table I)
MAX_FUSION_EPOCHS = None           # None = whole dataset; e.g. 300 for a quick run
OUTPUT_FOLDER = PROJECT_FOLDER / 'outputs'
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

# --- GNSS (paper Table II, Eq. (3)-(4), Ref. [35], [37]) ---------------------
BDS3_FIRST_PRN = 19                # BDS-3 satellites are C19 and above           [paper Table II]
URA_STD = 1.5                      # m, [ref 35]: "commonly 1-2 m"                [choice A9]
IONO_VERTICAL_STD = ((20.0, 9.0), (55.0, 4.5), (90.0, 6.0))  # (|lat| limit deg, m) [ref 35 Eq. 16]
IONO_MEAN_HEIGHT = 350e3           # m                                            [ref 35]
EARTH_MEAN_RADIUS = 6378.14e3      # m                                            [ref 35]
TROPO_ZENITH_STD = 0.12            # m                                            [ref 35 Eq. 17]
GOGPS_A, GOGPS_a, GOGPS_S0, GOGPS_S1 = 30.0, 20.0, 10.0, 50.0  # Eq. (4)          [ref 37]
DLL_BANDWIDTH = 2.0                # Hz                                           [ref 35]
DLL_CORRELATOR_SPACING = 0.1       # chip                                         [ref 35]
DLL_INTEGRATION_TIME = 0.02        # s                                            [ref 35]

# --- LEO simulation (paper Sec. II-A, Table II) ------------------------------
WALKER_SATELLITES, WALKER_PLANES, WALKER_PHASING = 144, 12, 1   # count [paper Table II]; planes/phasing [choice A1]
WALKER_INCLINATION_DEG, WALKER_ALTITUDE = 55.0, 1000e3                                             # [choice A1]
POLAR_SATELLITES, POLAR_PLANES, POLAR_PHASING = 81, 9, 1        # count [paper Table II]; planes/phasing [choice A1]
POLAR_INCLINATION_DEG, POLAR_ALTITUDE = 90.0, 1000e3                                               # [choice A1]
IONO_LOWER_HEIGHT = 100e3          # m, h_L of Eq. (2)                            [paper Sec. II-A]
IONO_UPPER_HEIGHT = 1000e3         # m, h_H of Eq. (2)                            [paper Sec. II-A]
LIGHT_TIME_THRESHOLD = 1e-3        # m, epsilon_T of Eq. (5) step 3               [choice A1]
LEO_ELEVATION_MASK_DEG = 10.0      # deg                                          [choice A2]
LEO_ELEVATION_BIN_DEG = 10.0       # deg, bins of the real-data error model       [choice A3]
LEO_CODE_CHIP_RATE = 1.023e6       # chip/s, used by the receiver-noise term      [choice A3]
LEO_NOISE_SEED = {'train': 1, 'test': 2}

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
LEARNING_RATE = 0.01               # Adam                                         [paper Table III, Sec. III-B]
TRAINING_EPOCHS = 500              # learning curves of Fig. 15 run to ~480 epochs [paper Fig. 15]
L2_WEIGHT = 1e-5                   # gamma of Eq. (32)                            [choice A14]
BACKPROP_WINDOW = 10               # fusion epochs per truncated back-propagation [choice A15]
VALIDATION_FRACTION = 0.2          # last 20 % of the training dataset            [choice A21]
EARLY_STOPPING_PATIENCE = 50       # epochs without a better validation loss      [choice A21]
GRADIENT_CLIP_NORM = 1.0           # max gradient norm before each Adam step      [choice A22]

# --- Fault detection and integrity (paper Sec. II-D, Fig. 20) ----------------
FALSE_ALARM_PROBABILITY = 1e-3     # alpha of the chi-square test                 [ref 33]
HORIZONTAL_PL_FACTOR = 6.0         # K_H                                          [choice A17]
VERTICAL_PL_FACTOR = 5.33          # K_V                                          [choice A17]
ALERT_LIMIT = 30.0                 # m, Stanford diagrams                         [paper Fig. 20]
