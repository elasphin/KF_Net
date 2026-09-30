"""Online test on the testing dataset (paper Sec. III-C): Masked CLA KalmanNet vs traditional EKF.

    python test.py   ->  outputs/test_summary.json, outputs/test_epochs.csv,
                         outputs/test_position_error.png, outputs/test_stanford.png

Both filters use the same measurements and the same FDE (Eq. (33)-(34)).
"""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

import settings as cfg
from earth_models import ecef_to_llh, ecef_to_ned_matrix
from fault_detection import stanford_percentages
from masked_cla_network import MaskedCLANetwork
from navigation_filter import prepare_measurements, run_filter
from read_dataset import load_navigation_data
from train import CHECKPOINT_FILE

METHOD_COLORS = {'masked_cla_kalmannet': '#2a78d6', 'traditional_ekf': '#eb6834'}


def north_east_down_errors(result):
    return np.array([ecef_to_ned_matrix(*ecef_to_llh(t)[:2]) @ (e - t)
                     for e, t in zip(result['estimate'], result['truth'])])


def summarize(result):
    """Position RMSE as in paper Table IV and Stanford percentages as in Fig. 20."""
    ned = north_east_down_errors(result)
    rmse = np.sqrt(np.mean(ned ** 2, axis=0))
    return {
        'rmse_north_m': rmse[0], 'rmse_east_m': rmse[1], 'rmse_down_m': rmse[2],
        'rmse_3d_m': float(np.sqrt(np.mean(np.sum(ned ** 2, axis=1)))),
        'stanford_horizontal_percent': stanford_percentages(np.hypot(ned[:, 0], ned[:, 1]), result['horizontal_pl']),
        'stanford_vertical_percent': stanford_percentages(np.abs(ned[:, 2]), result['vertical_pl']),
        'epochs_with_fault': int(sum(1 for s in result['faulty_satellite'] if s)),
        'epochs': len(ned),
    }


def save_plots(results):
    fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    for name, result in results.items():
        ned = north_east_down_errors(result)
        seconds = result['time'] - result['time'][0]
        for axis, column, label in zip(axes, ned.T, ('North', 'East', 'Down')):
            axis.plot(seconds, column, color=METHOD_COLORS[name], linewidth=1.2, label=name)
            axis.set_ylabel(f'{label} error [m]')
    for axis in axes:
        axis.grid(color='#e5e5e3', linewidth=0.8)
        axis.spines[['top', 'right']].set_visible(False)
    axes[0].legend(frameon=False)
    axes[-1].set_xlabel('Time [s]')
    fig.suptitle('Position error on the testing dataset (cf. paper Fig. 18)')
    fig.tight_layout()
    fig.savefig(cfg.OUTPUT_FOLDER / 'test_position_error.png', dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for name, result in results.items():
        ned = north_east_down_errors(result)
        for axis, error, pl in ((axes[0], np.hypot(ned[:, 0], ned[:, 1]), result['horizontal_pl']),
                                (axes[1], np.abs(ned[:, 2]), result['vertical_pl'])):
            axis.scatter(error, pl, s=8, color=METHOD_COLORS[name], alpha=0.5, label=name)
    limit = 1.3 * cfg.ALERT_LIMIT
    for axis, title in zip(axes, ('Horizontal', 'Vertical')):
        axis.plot([0, limit], [0, limit], color='#52514e', linewidth=1)
        axis.axhline(cfg.ALERT_LIMIT, color='#52514e', linewidth=1, linestyle='--')
        axis.axvline(cfg.ALERT_LIMIT, color='#52514e', linewidth=1, linestyle='--')
        axis.set(xlim=(0, limit), ylim=(0, limit), xlabel='Position error [m]', ylabel='Protection level [m]',
                 title=f'{title} Stanford diagram (AL = {cfg.ALERT_LIMIT:g} m)')
        axis.spines[['top', 'right']].set_visible(False)
    axes[0].legend(frameon=False, loc='upper left')
    fig.tight_layout()
    fig.savefig(cfg.OUTPUT_FOLDER / 'test_stanford.png', dpi=150)
    plt.close(fig)


def save_epoch_table(results):
    lines = ['method,time_gpst_s,north_error_m,east_error_m,down_error_m,horizontal_pl_m,vertical_pl_m,'
             'measurement_count,faulty_satellite']
    for name, result in results.items():
        ned = north_east_down_errors(result)
        for i, t in enumerate(result['time']):
            lines.append(f'{name},{t:.3f},{ned[i, 0]:.4f},{ned[i, 1]:.4f},{ned[i, 2]:.4f},'
                         f'{result["horizontal_pl"][i]:.4f},{result["vertical_pl"][i]:.4f},'
                         f'{result["measurement_count"][i]},{result["faulty_satellite"][i]}')
    (cfg.OUTPUT_FOLDER / 'test_epochs.csv').write_text('\n'.join(lines) + '\n')


def main():
    cfg.OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(CHECKPOINT_FILE)
    network = MaskedCLANetwork(checkpoint['max_measurements'])
    network.load_state_dict(checkpoint['state_dict'])
    network.eval()

    data = load_navigation_data('test')
    measurements = prepare_measurements(data, 'test')
    results = {'masked_cla_kalmannet': run_filter(data, measurements, network),
               'traditional_ekf': run_filter(data, measurements, network=None)}
    summary = {name: summarize(result) for name, result in results.items()}
    (cfg.OUTPUT_FOLDER / 'test_summary.json').write_text(json.dumps(summary, indent=2, default=float))
    save_epoch_table(results)
    save_plots(results)
    for name, s in summary.items():
        print(f'{name:22s} RMSE N {s["rmse_north_m"]:.2f}  E {s["rmse_east_m"]:.2f}  D {s["rmse_down_m"]:.2f}  '
              f'3D {s["rmse_3d_m"]:.2f} m')


if __name__ == '__main__':
    main()
