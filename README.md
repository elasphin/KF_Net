# Masked KalmanNet for GNSS/LEO/INS tightly coupled navigation

Simulation of J. Yan et al., *"A Robust Position Approach Based on Masked KalmanNet for GNSS/LEO/INS
Integrated Navigation System"*, IEEE Internet of Things Journal, vol. 13, no. 11, 2026
(DOI 10.1109/JIOT.2026.3673906).

Real GPS + BDS-3 pseudoranges and IMU data from the SmartPNT-POS dataset, plus simulated pseudoranges of real
LEO satellites used as signals of opportunity (Iridium, Orbcomm, Globalstar, OneWeb from TLE files: measured
with a numerical reference orbit, predicted by the filter with SGP4), are fused by a 15-state tightly coupled filter whose Kalman gain comes from a masked CNN–LSTM–attention
network. Fault detection and DIA adaptation run on the INS-predicted innovation.

What follows the paper exactly, what was removed, and every assumption the paper leaves open (with the
alternatives to choose from) are listed in [docs/ASSUMPTIONS.md](docs/ASSUMPTIONS.md). Each value in `settings.py` is
tagged `[paper]`, `[ref N]` or `[choice AX]`.

## Training experiments (branches `exp/...`)

Each branch is one training setup (`settings.EXPERIMENT`); its outputs go to their own folder (see Run).
`main` trains exactly as the paper says and nothing more (the setup of `exp/paper`, merged into `main`): the whole
Data01 trains the network (no validation, no early stopping), single-step gradient of Eq. (31) (filter and LSTM state
detached every epoch), alternating optimization [15], Adam with learning rate 0.01 (Table III) for 480 epochs
(Fig. 15), no gradient clipping and no gain scale; the model after the last epoch is tested. Every `exp/...` branch is
this training with one change: `exp/alternating` (joint instead of alternating optimization), `exp/grad-clip`,
`exp/gain-scale`, `exp/lr-sweep` (the learning rates of Fig. 15), `exp/input-norm` (L2 or z-score),
`exp/bptt-kalmannet` (BPTT of KalmanNet [14]) and `exp/tbptt-sensorfusion` (truncated BPTT of
KalmanNet4SensorFusion). Each branch describes its change in its README and docs/ASSUMPTIONS.md.

This branch, `exp/lr-sweep`: one learning rate per run, set in `settings.LEARNING_RATE` (0.01 of Table III; Fig. 15
compares 0.001, 0.003, 0.005, 0.01, 0.02), or with `LEARNING_RATE` in the notebook. Each rate writes to its own folder
`lr-sweep/lr_<rate>` (the data cache in `lr-sweep/cache` is shared); `python show_results.py compare` compares the
rates run so far with the other experiments.

## Running the experiments (Colab / Kaggle)

`run_experiments.ipynb` (the same on every branch) runs `main` or one `exp/...` branch: in Colab, File → Open notebook →
GitHub → `elasphin/KF_Net`, any branch → `run_experiments.ipynb` (on Kaggle: upload it, attach the dataset,
Internet on). Choose `BRANCH` and `MODE` in its first cell and run all cells: it clones the branch, sets the data and
epochs of the mode (`quick`: 1500 / 600 fusion epochs and 3 epochs, to check that a branch runs; `screening`:
3000 / 1000 and 100 epochs, the same for all branches, to compare them; `full`: all data and 480 epochs, as the
paper), runs `train.py`, `test.py` and `show_results.py` and compares every
experiment of that mode. The outputs of a mode are in `KF_Net_outputs/<mode>/<experiment>`.

- **Resume** (`train.py`): after every epoch `train.py` keeps the whole training state in
  `training_state.pt`. If a session ends, running it again (all cells of the notebook) continues after the last
  saved epoch, with the same result as an uninterrupted run. A change of settings, training code or data starts a
  new training; raising `TRAINING_EPOCHS` continues a finished one; `python train.py --restart` always starts anew.
- **Comparison** (`show_results.py compare`): reads every experiment folder of `OUTPUT_ROOT` and writes
  `comparison.txt` (training samples and epochs, with a warning if they differ; final training RMSE; test 3-D RMSE
  of the network and the EKF and the improvement, per LEO orbit) and `comparison.png` (test 3-D RMSE of each
  experiment, one panel per LEO orbit, EKF as reference). The experiments must have run with the same mode.

## Files

| File | Content |
|---|---|
| `settings.py` | All settings with their source, and the dataset and output folders in Colab (Google Drive), on Kaggle or on my computer |
| `dataset.py` | Finds the dataset folders under that path (or downloads the needed files) and reads RINEX, IMU (.imr), truth, SP3, CLK, broadcast header; reads the dataset, prepares the GNSS and LEO measurements and keeps them on disk between runs |
| `measurements.py` | Constants, frames, gravity, Klobuchar, Saastamoinen, variance of Eq. (3)-(4); satellite positions/clocks and the pseudorange model of Eq. (1) |
| `leo_pseudorange.py` | LEO orbits from TLE files (numerical reference orbit with EGM96 20x20, Sun, Moon; filter orbit: true, SGP4 of a TLE or neural network), the LEO pseudorange simulation, Eq. (1), (5) (Student-t MP/NLOS noise with the variance of Eq. (4), A3), and the orbit error variance of the filter |
| `egm96_degree20.txt` | EGM96 gravity coefficients up to degree 20 (for `leo_pseudorange.py`) |
| `navigation.py` | 15-state INS error model (Eq. (6)-(9)), measurement model, Kalman update; network input Eq. (10)-(17) and the masked CLA network Eq. (21)-(29); the filter of Fig. 2 (network or traditional EKF gain), shared by training and test, with fault detection Eq. (33), identification, DIA Eq. (34) (repeated after each identified fault, A27) and protection levels |
| `train.py` | Offline training on the whole training dataset, Eq. (30)-(32); resume of an interrupted training: the training state after every epoch |
| `run_experiments.ipynb` | Colab / Kaggle notebook that runs one experiment branch in one mode and compares |
| `test.py` | Online test: RMSE (Table IV) and Stanford percentages (Fig. 20) |
| `check_dataset.py` | Checks of the real dataset reading against the truth (IMR header, truth columns, lever arm, 1-s INS with IMU time offsets, free INS, pseudorange residuals) and the free-INS / EKF GNSS / EKF GNSS+LEO / network baselines on the whole dataset |
| `show_results.py` | Figures and table as in the paper: training loss and RMSE per epoch, trajectory and north/east/down errors (Fig. 18), error CDFs (Fig. 19), Stanford diagram per method (Fig. 20), Table IV, data sizes, learning rate and network size; `python show_results.py compare`: table and figure comparing the test results of every experiment branch run |

## Data

The paper uses the SmartPNT-POS dataset: <https://www.kaggle.com/datasets/fengzhusgg/smartpnt-pos>
(training: `Data01_20230102_ISA-100C_Vehicle_Complex`, testing: `Data02_20220309_ISA-100C_Vehicle_Complex`;
change the names in `settings.py` if the folders are named differently on Kaggle).

The data folder is set at the top of `settings.py` (`python settings.py` shows it):

1. **Colab**: Google Drive is mounted and the data is read from `/content/drive/MyDrive/Dataset`
   (the Drive folder <https://drive.google.com/drive/folders/1npnGKO7qwgKPvfoKTclzeA59wfpm860g>). If a script
   run with `!python` cannot mount Drive, run `from google.colab import drive; drive.mount('/content/drive')`
   in a notebook cell first.
2. **Kaggle notebook** with the dataset attached: read directly from `/kaggle/input/datasets/elasphin/mknet-project`
   (all of `/kaggle/input` is searched if that folder is missing).
3. **My computer**: anywhere under `Dataset/` next to the code (any depth), for example:

   ```
   Dataset/
   ├── Data01_20230102_ISA-100C_Vehicle_Complex/
   ├── Data02_20220309_ISA-100C_Vehicle_Complex/
   ├── LEO_TLE/           # TLE files (*.txt) of the LEO satellites around both days
   └── products/          # *.sp3, *.clk, brdm* of both days, if not inside the data folders
   ```

If the folders are not found there, only the needed files are downloaded with `kagglehub`. Put your Kaggle
API token in `~/.kaggle/kaggle.json` (or set `KAGGLE_USERNAME` and `KAGGLE_KEY`).

The LEO satellites are read from TLE files (`*.txt`, e.g. from Space-Track) in a folder `LEO_TLE` anywhere in the
dataset folder. They must cover the dataset days and at least a day around them (an older TLE for the
prediction and a newer one for the reference orbit); the public Kaggle dataset does not contain them.

Precise orbit/clock products and the broadcast navigation file of the observation day (`*.sp3`, `*.clk`,
`brdm*`) are read from the data folder, then from `products/` inside the dataset path, then from the dataset
folder itself (on Kaggle they are at its root); an SP3 or CLK file that does not cover the dataset stops the run. If the Kaggle folders do
not contain them, download the MGEX products of that day (e.g. from the IGS/BKG or CDDIS archives) into that
`products/` folder.

The initial standard deviations of the IMU biases are read from the dataset's `IMUErrorModel.txt` (block of the IMU
type in `README.xml`, docs/ASSUMPTIONS.md A7), in the data folder, a folder above it or anywhere in the dataset path.

## Run

In Colab, first get the code: `!git clone https://github.com/elasphin/KF_Net.git` and `%cd KF_Net`.

```bash
pip install -r requirements.txt
python train.py         # outputs/masked_cla_network.pt (model after the last epoch), training_history.json, training_info.json,
                        #         leo_orbit_error_train.json (orbit error of the training LEO orbit, used in R)
python test.py          # outputs/test_summary.json, test_epochs.csv, leo_orbit_error_test.json (one entry per LEO orbit)
python check_dataset.py # optional: dataset reading checks and baselines (python check_dataset.py test for Data02)
python show_results.py  # outputs/results_training.png, results_orbits.png, results_table.txt and, per LEO orbit,
                        #         results_errors_<orbit>.png, results_cdf_<orbit>.png, results_stanford_<orbit>.png
```

LEO orbit of the filter (docs/ASSUMPTIONS.md A26): the network is trained once with `LEO_TRAIN_ORBIT` (`'reference'`, the
true orbit) and `test.py` runs it and the traditional EKF with every orbit of `LEO_TEST_ORBITS` on the same
measurements and the same R: `'reference'` (upper bound), `'tle'` (SGP4 of the TLE available before the dataset) and,
once `leo_pseudorange.network_orbit` is written, `'network'` (neural-network orbit prediction).

The `outputs/` files are written to `OUTPUT_FOLDER` of `settings.py`, the folder `EXPERIMENT` (e.g. `main`) inside
`My Drive/KF_Net_outputs` in Colab (kept after the runtime ends), `/kaggle/working/outputs` on Kaggle, `outputs/` next
to the code on my computer.

By default both datasets are used whole (`MAX_FUSION_EPOCHS = {'train': None, 'test': None}`, as in the paper). Note
that both start with a static period (Data01: about 13 min, Data02: about 2 min before the vehicle moves).
For a quick run set `MAX_FUSION_EPOCHS` (one limit per dataset, e.g. `{'train': 1500, 'test': 600}`) and
`TRAINING_EPOCHS` in `settings.py`. Everything is then done on the first `MAX_FUSION_EPOCHS[split]` fusion epochs
of each dataset only: the RINEX observations and the two truth files are read only
over them, the SP3 and CLK products over them +- 3 h (`dataset.PRODUCT_MARGIN`, enough for the 10-point SP3
interpolation, so the GNSS satellite positions and clocks are the same as with the whole files), the LEO orbits are
made at these epochs, and the LEO C/N0 per elevation (`mean_cn0_bins`) and the LEO orbit error of R come from them. The LEO reference orbit is still integrated from the epoch of its
reference TLE (docs/ASSUMPTIONS.md A1), so the time from that epoch to the fusion epochs is not shortened. The time of
each stage (reading, GNSS, LEO) and of the training is printed.

## Run time

Three settings at the end of `settings.py` only change the run time; the results stay the same (checked bit
for bit against the Python versions):

| Setting | Values | What it does |
|---|---|---|
| `INS_MECHANIZATION` | `'numba'` (default) or `'python'` | INS mechanization of every IMU sample (`navigation.propagate_ins`): the Python loop of `mechanize`, or the same operations compiled with Numba (`navigation.mechanize_samples_numba`), ~16x faster |
| `LEO_FORCE_MODEL` | `'numba'` (default) or `'python'` | Equations of motion of the LEO reference orbit (EGM96 20x20, Sun, Moon) integrated by DOP853: `leo_pseudorange.equations_of_motion` or its compiled copy `leo_pseudorange.equations_of_motion_numba`, ~70x faster |
| `DATA_CACHE` | `True` (default) or `False` | Keeps the read dataset and the simulated measurements in `OUTPUT_FOLDER/cache` (one file per split); later runs read that file. A new file is made when a setting that changes the data, the code that makes it or an input file (dataset, products, TLE) changes; network, training and integrity settings do not count. Delete the folder to force a new one. |

The Numba versions compile on the first run (a few seconds) and keep the compiled code in `__pycache__`.
To compare the two `LEO_FORCE_MODEL` versions, note that with `DATA_CACHE = True` the orbits are computed only
when the cache is made (changing `LEO_FORCE_MODEL` makes a new cache file).

In training (network gain, no fault detection) the filter covariance P is not needed, so it is
not propagated there; those runs return NaN protection levels. The traditional EKF and `test.py` compute P as
before.

## References

PDFs in `docs/papers/`.

- Main paper: J. Yan et al., "A Robust Position Approach Based on Masked KalmanNet for GNSS/LEO/INS Integrated
  Navigation System," IEEE Internet Things J., vol. 13, no. 11, 2026.
- [14] G. Revach et al., "KalmanNet: Neural network aided Kalman filtering for partially known dynamics," IEEE TSP, 2022.
- [15] I. Buchnik et al., "Latent-KalmanNet: Learned Kalman filtering for tracking from high-dimensional signals," IEEE TSP, 2023.
- [33] S. Ciuban, P. J. G. Teunissen, C. C. J. M. Tiberius, "Dependence between parameter estimation and statistical
  hypothesis testing," IEEE T-ITS, 2025 (fault detection and DIA, Eq. (33)-(34)).
- [35] N. S. Zewge, H. Bang, "Fast multi-constellation GNSS satellite selection," IEEE TVT, 2025 (error model, Eq. (3)).
- [38] H. Zhao, Z. Yang, "A novel fault detection and exclusion method for applying low-cost INS/GNSS integrated
  navigation system in urban environments," IEEE T-ITS, 2025 (system matrix F).
