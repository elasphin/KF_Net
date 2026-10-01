"""Online test on the testing dataset (paper Sec. III-C): Masked CLA KalmanNet vs traditional EKF.

    python test.py   ->  outputs/test_summary.json, outputs/test_epochs.csv   (plots: show_results.py)

Both filters use the same measurements and the same FDE (Eq. (33)-(34)).
"""
import json

import numpy as np
import torch

import settings as cfg
from earth_models import ecef_to_llh, ecef_to_ned_matrix
from fault_detection import stanford_percentages
from masked_cla_network import MaskedCLANetwork
from navigation_filter import prepare_measurements, run_filter
from read_dataset import load_navigation_data
from train import CHECKPOINT_FILE


def summarize(result):
    """Position RMSE as in paper Table IV and Stanford percentages as in Fig. 20."""
    ned = result['ned_error']
    rmse = np.sqrt(np.mean(ned ** 2, axis=0))
    return {
        'rmse_north_m': rmse[0], 'rmse_east_m': rmse[1], 'rmse_down_m': rmse[2],
        'rmse_3d_m': result['position_rmse_m'],
        'stanford_horizontal_percent': stanford_percentages(np.hypot(ned[:, 0], ned[:, 1]), result['horizontal_pl']),
        'stanford_vertical_percent': stanford_percentages(np.abs(ned[:, 2]), result['vertical_pl']),
        'epochs_with_fault': sum(1 for s in result['faulty_satellite'] if s),
        'epochs': len(ned),
    }


def main():
    cfg.OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(CHECKPOINT_FILE)
    network = MaskedCLANetwork(checkpoint['max_measurements'])
    network.load_state_dict(checkpoint['state_dict'])

    data = load_navigation_data('test')
    measurements = prepare_measurements(data, 'test')
    results = {'masked_cla_kalmannet': run_filter(data, measurements, network),
               'traditional_ekf': run_filter(data, measurements, max_measurements=network.max_measurements)}

    summary = {name: summarize(result) for name, result in results.items()}
    (cfg.OUTPUT_FOLDER / 'test_summary.json').write_text(json.dumps(summary, indent=2, default=float))
    # Truth antenna trajectory in a local north/east frame at the start (paper Fig. 18(a)).
    origin = data.truth_antenna_position[0]
    truth_local = (data.truth_antenna_position[1:] - origin) @ ecef_to_ned_matrix(*ecef_to_llh(origin)[:2]).T
    lines = ['method,time_gpst_s,truth_north_m,truth_east_m,north_error_m,east_error_m,down_error_m,'
             'horizontal_pl_m,vertical_pl_m,measurement_count,faulty_satellite']
    for name, r in results.items():
        for i, t in enumerate(r['time']):
            n, e, d = r['ned_error'][i]
            lines.append(f"{name},{t:.3f},{truth_local[i, 0]:.4f},{truth_local[i, 1]:.4f},{n:.4f},{e:.4f},{d:.4f},"
                         f"{r['horizontal_pl'][i]:.4f},{r['vertical_pl'][i]:.4f},{r['measurement_count'][i]},"
                         f"{r['faulty_satellite'][i]}")
    (cfg.OUTPUT_FOLDER / 'test_epochs.csv').write_text('\n'.join(lines) + '\n')
    for name, s in summary.items():
        print(f"{name:22s} RMSE N {s['rmse_north_m']:.2f}  E {s['rmse_east_m']:.2f}  D {s['rmse_down_m']:.2f}  "
              f"3D {s['rmse_3d_m']:.2f} m")


if __name__ == '__main__':
    main()
