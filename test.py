"""Online test on the testing dataset (paper Sec. III-C): Masked CLA KalmanNet vs traditional EKF.

    python test.py --lr 0.003  ->  outputs/test_summary.json, outputs/test_epochs.csv   (plots: show_results.py)
    (the network trained by train.py --lr 0.003; outputs = OUTPUT_ROOT/lr-sweep/lr_0.003)

The network trained once (LEO_TRAIN_ORBIT) is tested with every LEO filter orbit of
settings.LEO_TEST_ORBITS on the same measurements and the same R (A26): the true orbit
(upper bound), the TLE orbit and, later, the neural-network orbit. Both filters use the
same measurements and the same FDE (Eq. (33)-(34)).
"""
import json

import numpy as np
import torch

import settings as cfg
from dataset import load_dataset
from measurements import ecef_to_llh, ecef_to_ned_matrix
from navigation import MaskedCLANetwork, run_filter, stanford_percentages
from train import CHECKPOINT_FILE
METHODS = ('masked_cla_kalmannet', 'traditional_ekf')


def summarize(result):
    """Position RMSE as in paper Table IV, Stanford percentages as in Fig. 20 and the FDE alarms."""
    ned = result['ned_error']
    rmse = np.sqrt(np.mean(ned ** 2, axis=0))
    return {
        'rmse_north_m': rmse[0], 'rmse_east_m': rmse[1], 'rmse_down_m': rmse[2],
        'rmse_3d_m': result['position_rmse_m'],
        'stanford_horizontal_percent': stanford_percentages(np.hypot(ned[:, 0], ned[:, 1]), result['horizontal_pl']),
        'stanford_vertical_percent': stanford_percentages(np.abs(ned[:, 2]), result['vertical_pl']),
        'epochs_with_fault': sum(1 for s in result['faulty_satellite'] if s),
        'epochs_with_leo_fault': sum(1 for s in result['faulty_satellite'] if 'L' in s),   # LEO ids: L + NORAD number
        'epochs': len(ned),
    }


def main():
    cfg.OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(CHECKPOINT_FILE)
    if checkpoint.get('leo_train_orbit') != cfg.LEO_TRAIN_ORBIT:
        raise ValueError(f"{CHECKPOINT_FILE} was trained with LEO orbit {checkpoint.get('leo_train_orbit')!r}, "
                         f'settings.LEO_TRAIN_ORBIT is {cfg.LEO_TRAIN_ORBIT!r}; run train.py again')
    network = MaskedCLANetwork(checkpoint['max_measurements'])
    network.load_state_dict(checkpoint['state_dict'])

    data, orbits = load_dataset('test')
    range_errors = json.loads((cfg.OUTPUT_FOLDER / 'leo_orbit_error_test.json').read_text())
    # Truth antenna trajectory in a local north/east frame at the start (paper Fig. 18(a)).
    origin = data.truth_antenna_position[0]
    truth_local = (data.truth_antenna_position[1:] - origin) @ ecef_to_ned_matrix(*ecef_to_llh(origin)[:2]).T
    summary = {}
    lines = ['leo_orbit,method,time_gpst_s,truth_north_m,truth_east_m,north_error_m,east_error_m,down_error_m,'
             'horizontal_pl_m,vertical_pl_m,measurement_count,faulty_satellite']
    for orbit, measurements in orbits.items():
        results = {'masked_cla_kalmannet': run_filter(data, measurements, network),
                   'traditional_ekf': run_filter(data, measurements, max_measurements=network.max_measurements)}
        summary[orbit] = {'leo_range_error': range_errors[orbit],
                          **{name: summarize(result) for name, result in results.items()}}
        for name, r in results.items():
            for i, t in enumerate(r['time']):
                n, e, d = r['ned_error'][i]
                lines.append(f"{orbit},{name},{t:.3f},{truth_local[i, 0]:.4f},{truth_local[i, 1]:.4f},{n:.4f},{e:.4f},"
                             f"{d:.4f},{r['horizontal_pl'][i]:.4f},{r['vertical_pl'][i]:.4f},"
                             f"{r['measurement_count'][i]},{r['faulty_satellite'][i]}")
        for name in METHODS:
            s = summary[orbit][name]
            print(f"LEO orbit {orbit:10s} {name:22s} RMSE N {s['rmse_north_m']:.2f}  E {s['rmse_east_m']:.2f}  "
                  f"D {s['rmse_down_m']:.2f}  3D {s['rmse_3d_m']:.2f} m | "
                  f"LEO fault epochs {s['epochs_with_leo_fault']}")
    (cfg.OUTPUT_FOLDER / 'test_summary.json').write_text(json.dumps(summary, indent=2, default=float))
    (cfg.OUTPUT_FOLDER / 'test_epochs.csv').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
