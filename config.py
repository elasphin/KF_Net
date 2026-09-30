"""All run settings of the Yan et al. (2026) Masked-KalmanNet simulation.

Edit values here only.  Physical constants and fixed model constants stay at
the top of the module that uses them.
"""
from pathlib import Path

# --- Paths -------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATASET_ROOT = ROOT / 'Dataset'
TRAIN_DIR = DATASET_ROOT / 'Data01_20230102_ISA-100C_Vehicle_Complex'
TEST_DIR = DATASET_ROOT / 'Data02_20220309_ISA-100C_Vehicle_Complex'
IMU_MODEL_FILE = DATASET_ROOT / 'IMUErrorModel.txt'
TLE_DIR = DATASET_ROOT / 'LEO_TLE'

LEO_CACHE_DIR = ROOT / 'leo_cache'
TRAIN_LEO_CACHE = LEO_CACHE_DIR / 'data01_leo_measurements.npz'
TEST_LEO_CACHE = LEO_CACHE_DIR / 'data02_leo_measurements.npz'

# One output folder per gain-scale mode (see GAIN_SCALE_MODE below), so the three
# variants never overwrite each other.  Defined after GAIN_SCALE_MODE.

# --- Data --------------------------------------------------------------------
TRAIN_FUSION_EPOCHS = 501         # GNSS epochs read from Data01
TEST_FUSION_EPOCHS = 50           # GNSS epochs read from Data02 (re-run leo.py after changing)
MAX_TRUTH_GAP_S = 2.0              # max gap for truth interpolation
GNSS_MIN_ELEV_DEG = 5.0

# --- LEO simulation (Yan Sec. II-A) -----------------------------------------
LEO_MIN_ELEV_DEG = 10.0
LEO_PREFILTER_GUARD_DEG = 0.5      # coarse pre-check margin before light-time
LEO_SEED = {'train': 0, 'test': 1}
LEO_TX_EPS_M = 1e-3                # light-time iteration tolerance
LEO_TX_MAX_ITER = 20
# Science missions without a ranging downlink (Swarm, Sentinel-1A/3A/3B).
LEO_EXCLUDED_SAT_IDS = (
    'NORAD-39451', 'NORAD-39452', 'NORAD-39453',
    'NORAD-39634', 'NORAD-41335', 'NORAD-43437',
)

# --- TLE truth/estimator pairs (consecutive-TLE differencing) ---------------
# Truth: TLE nearest the window centre.  Estimator: newest TLE published
# before the window, at least TLE_MIN_PAIR_SEP_S older than the truth TLE.
TLE_MAX_AGE_DAYS = 1.0
TLE_ALLOW_DEGRADED_EOP = False
TLE_ALLOW_NON_TLE_FILES = True
TLE_PUBLICATION_LATENCY_S = 0.0
TLE_TRUTH_MAX_OFFSET_DAYS = 1.0
TLE_MIN_PAIR_SEP_S = 3600.0
# Maneuver screening: ballistic pairs show |da| <= ~25 m, discrepancy <= ~8 km.
TLE_MANEUVER_SMA_JUMP_M = 100.0
TLE_MANEUVER_CONSISTENCY_M = 20000.0

# --- Network and training ----------------------------------------------------
RECURRENT_DROPOUT = 0.0           # Yan Table III (applied between the 5 LSTM layers)
LSTM_FORGET_BIAS = 1.0            # forget-gate bias init (not in Yan; 0.0 = previous behaviour)
TRAINING_EPOCHS = 40
TRAIN_EVAL_EVERY = 1               # report-only full-train rollout every k epochs (and epoch 1)
SEQUENCE_LENGTH = 40               # training subsequence length
SEQUENCE_STRIDE = 40               # start offset between (overlapping) subsequences
BATCH_SIZE = 4                   # subsequences per mini-batch (requested)
LEARNING_RATE = 1e-4               # Adam
LR_PLATEAU_FACTOR = 0.5            # ReduceLROnPlateau on validation loss
LR_PLATEAU_PATIENCE = 4
LR_MIN = 1e-5
WINDOW_SIZE = 4                    # fusion steps per optimizer step
TBPTT_DETACH_STEP = 2              # cut autograd history every k steps (= WINDOW_SIZE)
GRAD_CLIP_NORM = 1.0
EARLY_STOP_PATIENCE = 8
VALIDATION_FRACTION = 0.20
GAMMA_L2 = 1e-5                    # Yan Eq. (32) L2 weight
SEED = 0

# --- Kalman-gain output scale (not in Yan; documented deviation) ------------
# The network head gives a dimensionless K_n [B, 9, nmax]; the filter uses
#   'empirical': K = diag(g) K_n,  g = RMS of the classical EKF gain rows
#                (training split, all epochs; the previous behaviour)
#   'none'     : K = K_n            (g = 1, closest to Yan / KalmanNet)
#   'analytic' : K = diag(d) K_n diag(1 / sigma_j)
#                d = steady-state prior std of each error state (classical EKF,
#                per-block adaptive burn-in, median), sigma_j = Eq. (3) std of
#                measurement j (the same sigma that normalizes the input).
#                State scaling (D'Souza/Zanetti/Woffinden 2019) + measurement whitening.
#   'covariance': K = P~ H^T (H P~ H^T + lambda R)^-1, P~ = diag(s) P^- diag(s).
#                The network outputs only s (9) and lambda (1); H and P^- come from
#                the filter, so the gain follows satellite geometry and does not
#                depend on slot order (FDE exclusions, other datasets).  Zero network
#                output = classical EKF.  Not in Yan; documented deviation.
GAIN_SCALE_MODE = 'covariance'
# 'analytic' only:
SCALE_SETTLE_FACTOR = 2.0          # a block is settled once its prior std <= factor x steady median
SCALE_MIN_BURN_IN_S = 30.0         # burn-in is never shorter than this [s]
SCALE_MIN_EPOCHS = 50              # minimum steady-state epochs per block
SCALE_ROUND_POW10 = True           # round d to the nearest power of 10 (reproducible, robust)
SCALE_ROW_BALANCE = False          # also multiply row i by median RMS of K_tilde_EKF row i

# --- FDE (Yan Sec. II-D, Ref. [33]) ------------------------------------------
# Yan gives no false-alarm probability; Ref. [33] uses alpha = 1e-3.
FDE_ALPHA = 1e-3

# --- Output paths (depend on GAIN_SCALE_MODE) --------------------------------
if GAIN_SCALE_MODE not in ('empirical', 'none', 'analytic', 'covariance'):
    raise ValueError(f"GAIN_SCALE_MODE must be 'empirical', 'none', 'analytic' or 'covariance'; "
                     f"received {GAIN_SCALE_MODE!r}")
OUTPUT_DIR = ROOT / f'outputs_minimal_9state_norm_g-{GAIN_SCALE_MODE}'
CHECKPOINT_PATH = OUTPUT_DIR / 'best_model.pt'
TEST_OUTPUT_DIR = OUTPUT_DIR / 'test'
