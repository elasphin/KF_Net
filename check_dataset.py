"""Checks that the real dataset is read and synchronized correctly, and baselines for the training run.

    python check_dataset.py          (after train.py: uses the cache and leo_orbit_error_train.json)
    python check_dataset.py test     (the testing dataset; after test.py has written leo_orbit_error_test.json)

Everything is compared with the post-processed truth of the same dataset; the expected values are printed with
each result. Sections:
  1. IMR header (delta or rate samples, GPS or UTC time tags, time tag bias) and the IMU sample interval
  2. first rows of both truth files with their column numbers (the columns read by read_dataset.truth_row)
  3. truth velocity vs the time derivative of the truth position; antenna truth vs IMU truth + C lever arm
  4. INS over each 1 s fusion interval from the truth, for several IMU time offsets (attitude convention,
     mounting, IMU scale and time tags: a wrong one gives a large velocity error or a non-zero best offset)
  5. free INS from the truth over 10 / 30 / 60 / 100 s
  6. GPS / BDS-3 / LEO pseudorange residuals at the truth antenna (receiver clock of each system removed)
  7. filter baselines on the training and validation parts of train.py: free INS, EKF with GNSS only,
     EKF with GNSS + LEO and, if outputs/masked_cla_network.pt exists, the network
"""
import dataclasses
import struct
import sys

import numpy as np
import torch

import settings as cfg
from data_cache import load_dataset
from earth_models import EARTH_ROTATION_VECTOR, ecef_to_llh, ecef_to_ned_matrix, gravity, rotation_matrix_to_vector, skew
from gnss_measurements import predict_pseudoranges
from ins_filter import propagate_ins, truth_state
from navigation_filter import run_filter
from read_dataset import IMR_HEADER_SIZE, find_dataset_folder, read_imu, read_rover_info

IMR_FIELDS = ('header', 'byte_order', 'version', 'delta_theta', 'delta_velocity', 'rate_hz', 'gyro_scale',
              'accel_scale', 'utc_or_gps_time', 'receiver_or_corrected_time', 'time_tag_bias', 'imu_name',
              'reserved', 'reserved', 'reserved', 'program', 'creation_time', 'creation_time', 'creation_time',
              'creation_time', 'creation_time', 'creation_time', 'lever_arm_valid', 'lever_x_mm', 'lever_y_mm',
              'lever_z_mm', 'reserved')
OFFSETS = (-18.0, -1.0, -0.1, -0.02, -0.01, 0.0, 0.01, 0.02, 0.1, 1.0, 18.0)   # s, IMU time tag shifts tried
FREE_INS_SPANS = (10, 30, 60, 100)                                            # fusion epochs


def ned(data, k, vector):
    return ecef_to_ned_matrix(*ecef_to_llh(data.truth_position[k])[:2]) @ vector


def stats(values):
    values = np.abs(np.asarray(values, dtype=float))
    return f'RMS {np.sqrt(np.mean(values ** 2)):.4g}  95% {np.percentile(values, 95):.4g}  max {np.max(values):.4g}'


def imr_header(folder, imu_type):
    print('\n1. IMR header and IMU samples')
    path = folder / f'{imu_type}.imr'
    header = path.read_bytes()[:IMR_HEADER_SIZE]
    endian = '<' if header[8] == 0 else '>'
    fields = dict(zip(IMR_FIELDS, struct.unpack(endian + '8scdiidddiid32s?BBB32s6h?iii354s', header)))
    for name in ('version', 'delta_theta', 'delta_velocity', 'rate_hz', 'gyro_scale', 'accel_scale',
                 'utc_or_gps_time', 'receiver_or_corrected_time', 'time_tag_bias', 'imu_name'):
        value = fields[name]
        print(f"   {name:28s} {value.rstrip(bytes(1)).decode(errors='replace') if isinstance(value, bytes) else value}")
    print('   read_dataset.read_imu assumes delta_theta = delta_velocity = 1 (increments, multiplied by the rate),'
          ' utc_or_gps_time = 2 (GPS; 1 = UTC would be 18 s off) and time_tag_bias in ms')
    if fields['delta_theta'] != 1 or fields['delta_velocity'] != 1:
        print('   !!! samples are not increments: read_imu multiplies them by the rate')
    if fields['utc_or_gps_time'] == 1:
        print('   !!! time tags are UTC: read_imu treats them as GPS time')


def truth_columns(folder, imu_type):
    print('\n2. Truth files (read_dataset.truth_row reads: 0 week, 1 seconds, 9:12 ECEF position, '
          '15:18 ECEF velocity, 21:24 heading/pitch/roll)')
    for path in sorted(folder.glob('*GroundTruth.txt')):
        with path.open(errors='replace') as f:
            lines = [next(f, '') for _ in range(60)]
        header = [line.rstrip() for line in lines if line.strip() and not line.split()[0].isdigit()]
        first = next((line.split() for line in lines if line.split() and line.split()[0].isdigit()), [])
        print(f'   {path.name}:')
        for line in header[-4:]:
            print(f'      | {line[:160]}')
        print('      ' + '  '.join(f'[{i}]{v}' for i, v in enumerate(first)))


def truth_consistency(data):
    print('\n3. Truth consistency')
    t = data.fusion_times
    derivative = (data.truth_position[2:] - data.truth_position[:-2]) / (t[2:] - t[:-2])[:, None]
    print(f'   |truth velocity - d(truth position)/dt| [m/s]: '
          f'{stats(np.linalg.norm(data.truth_velocity[1:-1] - derivative, axis=1))}   (expected < 0.05)')
    print(f'   speed [m/s]: min {np.linalg.norm(data.truth_velocity, axis=1).min():.2f}  '
          f'max {np.linalg.norm(data.truth_velocity, axis=1).max():.2f}')
    lever = np.einsum('kij,j->ki', data.truth_attitude, data.lever_arm)
    difference = np.array([ned(data, k, data.truth_antenna_position[k] - data.truth_position[k] - lever[k])
                           for k in range(len(t))])
    print(f'   lever arm (body) {np.round(data.lever_arm, 3)} m; antenna truth - (IMU truth + C l), NED [m]: '
          f'mean {np.round(difference.mean(axis=0), 3)}, {stats(np.linalg.norm(difference, axis=1))}'
          f'   (expected < 0.1)')


def one_second_ins(data, full_imu):
    print('\n4. INS over each fusion interval from the truth (velocity error at the end of the interval, m/s)')
    print('   expected: smallest at offset 0, RMS < 0.01 m/s for a navigation-grade IMU')
    results = []
    for offset in OFFSETS:
        shifted = dataclasses.replace(data, imu_times=full_imu[0] + offset, gyro=full_imu[1], accel=full_imu[2])
        if shifted.imu_times[0] > data.fusion_times[0] or shifted.imu_times[-1] <= data.fusion_times[-1]:
            print(f'   IMU time offset {offset:+7.2f} s: the IMU file does not cover the fusion epochs')
            continue
        velocity_error, attitude_error = [], []
        for k in range(len(data.fusion_times) - 1):
            state = propagate_ins(truth_state(shifted, k), shifted, data.fusion_times[k], data.fusion_times[k + 1])[0]
            velocity_error.append(np.linalg.norm(data.truth_velocity[k + 1] - state.velocity))
            attitude_error.append(np.linalg.norm(rotation_matrix_to_vector(data.truth_attitude[k + 1] @ state.attitude.T)))
        results.append((offset, np.sqrt(np.mean(np.square(velocity_error)))))
        print(f'   IMU time offset {offset:+7.2f} s: velocity {stats(velocity_error)}   '
              f'attitude [deg] {stats(np.rad2deg(attitude_error))}')
    best, best_rms = min(results, key=lambda r: r[1])
    if best_rms < 0.8 * dict(results).get(0.0, np.inf):
        print(f'   !!! the IMU time offset {best:+.2f} s fits better than 0')

    static = np.flatnonzero(np.linalg.norm(data.truth_velocity, axis=1) < 0.02)
    static = static[:np.argmax(np.append(np.diff(static) > 1, True)) + 1]          # first run of static epochs
    if len(static) >= 5:
        i = np.flatnonzero((full_imu[0] >= data.fusion_times[static[0]]) & (full_imu[0] <= data.fusion_times[static[-1]]))
        k = static[len(static) // 2]
        C, r = data.truth_attitude[k], data.truth_position[k]
        W = skew(EARTH_ROTATION_VECTOR)
        expected_force, expected_rate = C.T @ (-gravity(r) + W @ W @ r), C.T @ EARTH_ROTATION_VECTOR
        force, rate = full_imu[2][i].mean(axis=0), full_imu[1][i].mean(axis=0)
        angle = np.rad2deg(np.arccos(force @ expected_force / np.linalg.norm(force) / np.linalg.norm(expected_force)))
        print(f'   static epochs ({len(static)}): measured specific force {np.round(force, 4)}, from truth attitude '
              f'{np.round(expected_force, 4)} m/s^2, angle {angle:.3f} deg (expected < 0.05)')
        print(f'   gyro mean - Earth rate in body: {np.round(np.rad2deg(rate - expected_rate) * 3600.0, 2)} deg/h')
    else:
        print('   no static epochs (speed < 0.02 m/s) in this span')


def free_ins(data):
    print('\n5. Free INS from the truth (3-D position error at the end, m)')
    for span in FREE_INS_SPANS:
        errors = []
        for first in range(0, len(data.fusion_times) - span, max(span // 2, 1)):
            state = truth_state(data, first)
            for k in range(first + 1, first + span + 1):
                state = propagate_ins(state, data, data.fusion_times[k - 1], data.fusion_times[k])[0]
            errors.append(np.linalg.norm(data.truth_position[first + span] - state.position))
        if errors:
            print(f'   {span:4d} epochs ({data.fusion_times[span] - data.fusion_times[0]:.0f} s): {stats(errors)}')


def residuals(data, measurements):
    print('\n6. Pseudorange residuals at the truth antenna, receiver clock of GPS and BDS-3 removed (median), m')
    errors, counts = {'G': [], 'C': [], 'L': []}, {'G': [], 'C': [], 'L': []}
    for k, meas in enumerate(measurements):
        predicted, _, elevation, _ = predict_pseudoranges(data.truth_antenna_position[k], meas, data.fusion_times[k],
                                                          data.klobuchar_alpha, data.klobuchar_beta)
        error = meas.pseudoranges - predicted
        for system in errors:
            rows = np.flatnonzero((meas.systems == system) & (elevation > 0.0))
            counts[system].append(len(rows))
            if len(rows) and system != 'L':
                errors[system] += list(error[rows] - np.median(error[rows]))
            elif len(rows):
                errors[system] += list(error[rows])
    for system, name in (('G', 'GPS'), ('C', 'BDS-3'), ('L', 'LEO')):
        if errors[system]:
            e = np.array(errors[system])
            print(f'   {name:5s}: {stats(e)}  mean {e.mean():+.3f}  > 30 m: {100 * np.mean(np.abs(e) > 30):.1f} %  '
                  f'satellites per epoch: {np.mean(counts[system]):.1f} (min {min(counts[system])}, '
                  f'max {max(counts[system])})')
    print('   expected: GPS / BDS-3 a few m (more in the urban parts), no large mean; LEO about 1-3 m')


def baselines(data, measurements):
    print('\n7. Filter baselines on the training / validation parts of train.py, 3-D antenna RMSE [m]')
    last = len(data.fusion_times) - 1
    split = int(round(last * (1.0 - cfg.VALIDATION_FRACTION)))
    only_gnss = [m.subset(np.flatnonzero(m.systems != 'L')) for m in measurements]
    nothing = [m.subset(np.array([], dtype=int)) for m in measurements]
    network = None
    checkpoint_file = cfg.OUTPUT_FOLDER / 'masked_cla_network.pt'
    if checkpoint_file.exists():
        from masked_cla_network import MaskedCLANetwork
        checkpoint = torch.load(checkpoint_file)
        network = MaskedCLANetwork(checkpoint['max_measurements'])
        network.load_state_dict(checkpoint['state_dict'])
    for label, first, end in (('training', 0, split), ('validation', split, last), ('all', 0, last)):
        line = f'   {label:10s} epochs {first}..{end}:'
        for name, meas in (('free INS', nothing), ('EKF GNSS', only_gnss), ('EKF GNSS+LEO', measurements)):
            line += f'  {name} {run_filter(data, meas, None, first, end, fault_detection=False)["position_rmse_m"]:.2f}'
        if network is not None:
            line += f'  network {run_filter(data, measurements, network, first, end, False)["position_rmse_m"]:.2f}'
        print(line)


def main():
    split = sys.argv[1] if len(sys.argv) > 1 else 'train'
    folder = find_dataset_folder(cfg.TRAIN_FOLDER_NAME if split == 'train' else cfg.TEST_FOLDER_NAME)
    imu_type = read_rover_info(folder / 'README.xml')[0]
    data, orbits = load_dataset(split)
    measurements = orbits[cfg.LEO_TRAIN_ORBIT if split == 'train' else cfg.LEO_TEST_ORBITS[0]]
    print(f'{data.name}: {len(data.fusion_times)} fusion epochs, '
          f'{data.fusion_times[-1] - data.fusion_times[0]:.0f} s, '
          f'interval {np.median(np.diff(data.fusion_times)):.3f} s; IMU interval '
          f'{np.median(np.diff(data.imu_times)) * 1e3:.2f} ms (max gap {np.max(np.diff(data.imu_times)) * 1e3:.1f} ms)')
    imr_header(folder, imu_type)
    truth_columns(folder, imu_type)
    truth_consistency(data)
    one_second_ins(data, read_imu(folder / f'{imu_type}.imr', data.fusion_times[0]))
    free_ins(data)
    residuals(data, measurements)
    baselines(data, measurements)


if __name__ == '__main__':
    main()
