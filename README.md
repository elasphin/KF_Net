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
| `kaggle_download.py` | Finds the dataset on Kaggle / locally, or downloads the needed files |
| `read_dataset.py` | Reads RINEX, IMU (.imr), truth, SP3, CLK, broadcast header |
| `earth_models.py` | Constants, frames, gravity, Klobuchar, Saastamoinen, variance of Eq. (3)-(4) |
| `gnss_measurements.py` | Satellite positions/clocks and the pseudorange model of Eq. (1) |
| `leo_simulation.py` | LEO constellation and pseudorange simulation, Eq. (1), (2), (5) |
| `ins_filter.py` | 15-state INS error model (Eq. (6)-(9)), measurement model, Kalman update |
| `masked_cla_network.py` | Network input Eq. (10)-(17) and the masked CLA network Eq. (21)-(29) |
| `fault_detection.py` | Fault detection Eq. (33), identification, DIA Eq. (34), protection levels |
| `navigation_filter.py` | Online filter of Fig. 2 (network gain or traditional EKF gain) |
| `train.py` | Offline training, Eq. (30)-(32) |
| `test.py` | Online test: RMSE (Table IV) and Stanford diagrams (Fig. 20) |
| `show_results.py` | One figure and table: train/validation loss per epoch, test error CDF, data sizes, learning rate, network size and test metrics |

## Data

The paper uses the SmartPNT-POS dataset: <https://www.kaggle.com/datasets/fengzhusgg/smartpnt-pos>
(training: `Data01_20230102_ISA-100C_Vehicle_Complex`, testing: `Data02_20220309_ISA-100C_Vehicle_Complex`;
change the names in `settings.py` if the folders are named differently on Kaggle).

The data folder is found in this order:

1. **Kaggle notebook** with the dataset attached: read directly from `/kaggle/input/smartpnt-pos`.
2. **Local copy** anywhere under `Dataset/`.
3. **Automatic download** of only the needed files with `kagglehub`. Put your Kaggle API token in
   `~/.kaggle/kaggle.json` (or set `KAGGLE_USERNAME` and `KAGGLE_KEY`).

Precise orbit/clock products and the broadcast navigation file of the observation day (`*.sp3`, `*.clk`,
`brdm*`) are read from the data folder or from `Dataset/products/`. If the Kaggle folders do not contain
them, download the MGEX products of that day (e.g. from the IGS/BKG or CDDIS archives) into `Dataset/products/`.

## Run

```bash
pip install -r requirements.txt
python train.py         # outputs/masked_cla_network.pt (best validation model), training_history.json, training_info.json
python test.py          # outputs/test_summary.json, test_epochs.csv, test_position_error.png, test_stanford.png
python show_results.py  # outputs/results.png and a printed table
```

For a quick run set `MAX_FUSION_EPOCHS` (e.g. 300) and `TRAINING_EPOCHS` in `settings.py`.
