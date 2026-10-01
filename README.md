# Masked KalmanNet for GNSS/LEO/INS tightly coupled navigation

Simulation of J. Yan et al., *"A Robust Position Approach Based on Masked KalmanNet for GNSS/LEO/INS
Integrated Navigation System"*, IEEE Internet of Things Journal, vol. 13, no. 11, 2026
(DOI 10.1109/JIOT.2026.3673906).

Real GPS + BDS-3 pseudoranges and IMU data from the SmartPNT-POS dataset, plus simulated pseudoranges of real
LEO satellites used as signals of opportunity (Iridium, Orbcomm, Globalstar, OneWeb from TLE files: measured
with a numerical reference orbit, predicted by the filter with SGP4), are fused by a 15-state tightly coupled filter whose Kalman gain comes from a masked CNN–LSTM–attention
network. Fault detection and DIA adaptation run on the INS-predicted innovation.

What follows the paper exactly, what was removed, and every assumption the paper leaves open (with the
alternatives to choose from) are listed in [ASSUMPTIONS.md](ASSUMPTIONS.md). Each value in `settings.py` is
tagged `[paper]`, `[ref N]` or `[choice AX]`.

## Files

| File | Content |
|---|---|
| `settings.py` | All settings with their source, and the dataset and output folders in Colab (Google Drive), on Kaggle or on my computer |
| `read_dataset.py` | Finds the dataset folders under that path (or downloads the needed files) and reads RINEX, IMU (.imr), truth, SP3, CLK, broadcast header |
| `earth_models.py` | Constants, frames, gravity, Klobuchar, Saastamoinen, variance of Eq. (3)-(4) |
| `gnss_measurements.py` | Satellite positions/clocks and the pseudorange model of Eq. (1) |
| `leo_pseudorange.py` | LEO orbits from TLE files (numerical reference orbit with EGM96 20x20, Sun, Moon; filter orbit: true, SGP4 of a TLE or neural network), the LEO pseudorange simulation, Eq. (1), (5), and the orbit error variance of the filter |
| `egm96_degree20.txt` | EGM96 gravity coefficients up to degree 20 (for `leo_pseudorange.py`) |
| `data_cache.py` | Reads the dataset, prepares the GNSS and LEO measurements and keeps them on disk between runs |
| `ins_filter.py` | 15-state INS error model (Eq. (6)-(9)), measurement model, Kalman update |
| `masked_cla_network.py` | Network input Eq. (10)-(17) and the masked CLA network Eq. (21)-(29) |
| `navigation_filter.py` | The filter of Fig. 2 (network or traditional EKF gain), shared by training, validation and test, with fault detection Eq. (33), identification, DIA Eq. (34) (repeated after each identified fault, A27) and protection levels |
| `train.py` | Offline training with validation, Eq. (30)-(32) |
| `test.py` | Online test: RMSE (Table IV) and Stanford percentages (Fig. 20) |
| `check_dataset.py` | Checks of the real dataset reading against the truth (IMR header, truth columns, lever arm, 1-s INS with IMU time offsets, free INS, pseudorange residuals) and the free-INS / EKF GNSS / EKF GNSS+LEO / network baselines on the training and validation parts |
| `show_results.py` | Figures and table as in the paper: train/validation loss and RMSE per epoch, trajectory and north/east/down errors (Fig. 18), error CDFs (Fig. 19), Stanford diagram per method (Fig. 20), Table IV, data sizes, learning rate and network size |

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
3. **My computer**: anywhere under `Dataset/`.

If the folders are not found there, only the needed files are downloaded with `kagglehub`. Put your Kaggle
API token in `~/.kaggle/kaggle.json` (or set `KAGGLE_USERNAME` and `KAGGLE_KEY`).

The LEO satellites are read from TLE files (`*.txt`, e.g. from Space-Track) in a folder `LEO_TLE` anywhere in the
dataset folder. They must cover the dataset days and at least a day around them (an older TLE for the
prediction and a newer one for the reference orbit); the public Kaggle dataset does not contain them.

Precise orbit/clock products and the broadcast navigation file of the observation day (`*.sp3`, `*.clk`,
`brdm*`) are read from the data folder or from `products/` inside the dataset path. If the Kaggle folders do
not contain them, download the MGEX products of that day (e.g. from the IGS/BKG or CDDIS archives) into that
`products/` folder.

## Run

In Colab, first get the code: `!git clone https://github.com/elasphin/KF_Net.git` and `%cd KF_Net`.

```bash
pip install -r requirements.txt
python train.py         # outputs/masked_cla_network.pt (best validation model), training_history.json, training_info.json,
                        #         leo_orbit_error_train.json (orbit error of the training LEO orbit, used in R)
python test.py          # outputs/test_summary.json, test_epochs.csv, leo_orbit_error_test.json (one entry per LEO orbit)
python check_dataset.py # optional: dataset reading checks and baselines (python check_dataset.py test for Data02)
python show_results.py  # outputs/results_training.png, results_orbits.png, results_table.txt and, per LEO orbit,
                        #         results_errors_<orbit>.png, results_cdf_<orbit>.png, results_stanford_<orbit>.png
```

LEO orbit of the filter (ASSUMPTIONS.md A26): the network is trained once with `LEO_TRAIN_ORBIT` (`'reference'`, the
true orbit) and `test.py` runs it and the traditional EKF with every orbit of `LEO_TEST_ORBITS` on the same
measurements and the same R: `'reference'` (upper bound), `'tle'` (SGP4 of the TLE available before the dataset) and,
once `leo_pseudorange.network_orbit` is written, `'network'` (neural-network orbit prediction).

The `outputs/` files are written to `OUTPUT_FOLDER` of `settings.py`: `My Drive/KF_Net_outputs` in Colab
(kept after the runtime ends), `/kaggle/working/outputs` on Kaggle, `outputs/` next to the code on my computer.

For a quick run set `MAX_FUSION_EPOCHS` (one limit per dataset, e.g. `{'train': 300, 'test': 300}`) and
`TRAINING_EPOCHS` in `settings.py`. Everything is then done on the first `MAX_FUSION_EPOCHS[split]` fusion epochs
of each dataset only: the RINEX observations and the two truth files are read only
over them, the SP3 and CLK products over them +- 3 h (`read_dataset.PRODUCT_MARGIN`, enough for the 10-point SP3
interpolation, so the GNSS satellite positions and clocks are the same as with the whole files), the LEO orbits are
made at these epochs, and the LEO error statistics (`real_error_bins`), the LEO orbit error of R and the
training/validation split come from them. The LEO reference orbit is still integrated from the epoch of its
reference TLE (ASSUMPTIONS.md A1), so the time from that epoch to the fusion epochs is not shortened. The time of
each stage (reading, GNSS, LEO) and of the training is printed.

## Run time

Three settings at the end of `settings.py` only change the run time; the results stay the same (checked bit
for bit against the Python versions):

| Setting | Values | What it does |
|---|---|---|
| `INS_MECHANIZATION` | `'numba'` (default) or `'python'` | INS mechanization of every IMU sample (`ins_filter.propagate_ins`): the Python loop of `mechanize`, or the same operations compiled with Numba (`ins_filter.mechanize_samples_numba`), ~16x faster |
| `LEO_FORCE_MODEL` | `'numba'` (default) or `'python'` | Equations of motion of the LEO reference orbit (EGM96 20x20, Sun, Moon) integrated by DOP853: `leo_pseudorange.equations_of_motion` or its compiled copy `leo_pseudorange.equations_of_motion_numba`, ~70x faster |
| `DATA_CACHE` | `True` (default) or `False` | Keeps the read dataset and the simulated measurements in `OUTPUT_FOLDER/cache` (one file per split); later runs read that file. A new file is made when a setting that changes the data, the code that makes it or an input file (dataset, products, TLE) changes; network, training and integrity settings do not count. Delete the folder to force a new one. |

The Numba versions compile on the first run (a few seconds) and keep the compiled code in `__pycache__`.
To compare the two `LEO_FORCE_MODEL` versions, note that with `DATA_CACHE = True` the orbits are computed only
when the cache is made (changing `LEO_FORCE_MODEL` makes a new cache file).

In training and validation (network gain, no fault detection) the filter covariance P is not needed, so it is
not propagated there; those runs return NaN protection levels. The traditional EKF and `test.py` compute P as
before.

## References

PDFs in `Papers/`.

- Main paper: J. Yan et al., "A Robust Position Approach Based on Masked KalmanNet for GNSS/LEO/INS Integrated
  Navigation System," IEEE Internet Things J., vol. 13, no. 11, 2026.
- [14] G. Revach et al., "KalmanNet: Neural network aided Kalman filtering for partially known dynamics," IEEE TSP, 2022.
- [15] I. Buchnik et al., "Latent-KalmanNet: Learned Kalman filtering for tracking from high-dimensional signals," IEEE TSP, 2023.
- [33] S. Ciuban, P. J. G. Teunissen, C. C. J. M. Tiberius, "Dependence between parameter estimation and statistical
  hypothesis testing," IEEE T-ITS, 2025 (fault detection and DIA, Eq. (33)-(34)).
- [35] N. S. Zewge, H. Bang, "Fast multi-constellation GNSS satellite selection," IEEE TVT, 2025 (error model, Eq. (3)).
- [38] H. Zhao, Z. Yang, "A novel fault detection and exclusion method for applying low-cost INS/GNSS integrated
  navigation system in urban environments," IEEE T-ITS, 2025 (system matrix F).
