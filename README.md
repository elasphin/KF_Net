# Masked KalmanNet for GNSS/LEO/INS tightly coupled navigation

Simulation of J. Yan et al., *"A Robust Position Approach Based on Masked KalmanNet for GNSS/LEO/INS
Integrated Navigation System"*, IEEE Internet of Things Journal, vol. 13, no. 11, 2026
(DOI 10.1109/JIOT.2026.3673906).

Real GPS + BDS-3 pseudoranges and IMU data from the SmartPNT-POS dataset, plus simulated LEO pseudoranges,
are fused by a 15-state tightly coupled filter whose Kalman gain comes from a masked CNN–LSTM–attention
network. Fault detection and DIA adaptation run on the INS-predicted innovation.

What follows the paper exactly, what was removed, and every assumption the paper leaves open (with the
alternatives to choose from) are listed in [ASSUMPTIONS.md](ASSUMPTIONS.md). Each value in `settings.py` is
tagged `[paper]`, `[ref N]` or `[choice AX]`.

## Files

| File | Content |
|---|---|
| `settings.py` | All settings with their source |
| `dataset_path.py` | Dataset and output folders in Colab (Google Drive), on Kaggle or on my computer |
| `kaggle_download.py` | Finds the dataset folders under that path, or downloads the needed files |
| `read_dataset.py` | Reads RINEX, IMU (.imr), truth, SP3, CLK, broadcast header |
| `earth_models.py` | Constants, frames, gravity, Klobuchar, Saastamoinen, variance of Eq. (3)-(4) |
| `gnss_measurements.py` | Satellite positions/clocks and the pseudorange model of Eq. (1) |
| `leo_simulation.py` | LEO constellation and pseudorange simulation, Eq. (1), (2), (5) |
| `ins_filter.py` | 15-state INS error model (Eq. (6)-(9)), measurement model, Kalman update |
| `masked_cla_network.py` | Network input Eq. (10)-(17) and the masked CLA network Eq. (21)-(29) |
| `fault_detection.py` | Fault detection Eq. (33), identification, DIA Eq. (34), protection levels |
| `navigation_filter.py` | The filter of Fig. 2 (network or traditional EKF gain), shared by training, validation and test |
| `train.py` | Offline training with validation, Eq. (30)-(32) |
| `test.py` | Online test: RMSE (Table IV) and Stanford percentages (Fig. 20) |
| `show_results.py` | Figures and table as in the paper: train/validation loss and RMSE per epoch, trajectory and north/east/down errors (Fig. 18), error CDFs (Fig. 19), Stanford diagram per method (Fig. 20), Table IV, data sizes, learning rate and network size |

## Data

The paper uses the SmartPNT-POS dataset: <https://www.kaggle.com/datasets/fengzhusgg/smartpnt-pos>
(training: `Data01_20230102_ISA-100C_Vehicle_Complex`, testing: `Data02_20220309_ISA-100C_Vehicle_Complex`;
change the names in `settings.py` if the folders are named differently on Kaggle).

The data folder is set in `dataset_path.py`:

1. **Colab**: Google Drive is mounted and the data is read from `/content/drive/MyDrive/Dataset`
   (the Drive folder <https://drive.google.com/drive/folders/1npnGKO7qwgKPvfoKTclzeA59wfpm860g>). If a script
   run with `!python` cannot mount Drive, run `from google.colab import drive; drive.mount('/content/drive')`
   in a notebook cell first.
2. **Kaggle notebook** with the dataset attached: read directly from `/kaggle/input/datasets/elasphin/mknet-project`
   (all of `/kaggle/input` is searched if that folder is missing).
3. **My computer**: anywhere under `Dataset/`.

If the folders are not found there, only the needed files are downloaded with `kagglehub`. Put your Kaggle
API token in `~/.kaggle/kaggle.json` (or set `KAGGLE_USERNAME` and `KAGGLE_KEY`).

Precise orbit/clock products and the broadcast navigation file of the observation day (`*.sp3`, `*.clk`,
`brdm*`) are read from the data folder or from `products/` inside the dataset path. If the Kaggle folders do
not contain them, download the MGEX products of that day (e.g. from the IGS/BKG or CDDIS archives) into that
`products/` folder.

## Run

In Colab, first get the code: `!git clone https://github.com/elasphin/KF_Net.git` and `%cd KF_Net`.

```bash
pip install -r requirements.txt
python train.py         # outputs/masked_cla_network.pt (best validation model), training_history.json, training_info.json
python test.py          # outputs/test_summary.json, test_epochs.csv
python show_results.py  # outputs/results_training.png, results_errors.png, results_cdf.png, results_stanford.png, results_table.txt
```

The `outputs/` files are written to `OUTPUT_FOLDER` of `dataset_path.py`: `My Drive/KF_Net_outputs` in Colab
(kept after the runtime ends), `/kaggle/working/outputs` on Kaggle, `outputs/` next to the code on my computer.

For a quick run set `MAX_FUSION_EPOCHS` (e.g. 300) and `TRAINING_EPOCHS` in `settings.py`.
