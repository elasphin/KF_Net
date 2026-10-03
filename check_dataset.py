"""Checks that the real dataset is read and synchronized correctly, and baselines for the training run.

    python check_dataset.py          (after train.py: uses the cache and leo_orbit_error_train.json)
    python check_dataset.py test     (the testing dataset; after test.py has written leo_orbit_error_test.json)

Everything is compared with the post-processed truth of the same dataset; the expected values are printed with
each result. Sections:
  1. IMR header (delta or rate samples, GPS or UTC time tags, time tag bias) and the IMU sample interval
  2. first rows of both truth files with their column numbers (the columns read by dataset.truth_row)
  3. truth velocity vs the time derivative of the truth position; antenna truth vs IMU truth + C lever arm
  4. INS over each 1 s fusion interval from the truth, for several IMU time offsets (attitude convention,
     mounting, IMU scale and time tags: a wrong one gives a large velocity error or a non-zero best offset)
  5. free INS from the truth over 10 / 30 / 60 / 100 s
  6. GPS / BDS-3 / LEO pseudorange residuals at the truth antenna (receiver clock of each system removed),
     also the mean residual of each GPS / BDS-3 satellite
  7. filter baselines on the whole dataset: free INS, EKF with GNSS only,
     EKF with GNSS + LEO and, if outputs/masked_cla_network.pt exists, the network
"""
import dataclasses
import struct
import sys

import numpy as np
import torch

import settings as cfg
from dataset import IMR_HEADER_SIZE, find_dataset_folder, load_dataset, read_imu, read_rover_info
from measurements import (EARTH_ROTATION_VECTOR, ecef_to_llh, ecef_to_ned_matrix, gravity, predict_pseudoranges,
                          rotation_matrix_to_vector, skew)
from navigation import propagate_ins, run_filter, truth_state
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
    print('   dataset.read_imu assumes delta_theta = delta_velocity = 1 (increments, multiplied by the rate),'
          ' utc_or_gps_time = 2 (GPS; 1 = UTC would be 18 s off) and time_tag_bias in ms')
    if fields['delta_theta'] != 1 or fields['delta_velocity'] != 1:
        print('   !!! samples are not increments: read_imu multiplies them by the rate')
    if fields['utc_or_gps_time'] == 1:
        print('   !!! time tags are UTC: read_imu treats them as GPS time')


def truth_columns(folder, imu_type):
    print('\n2. Truth files (dataset.truth_row reads: 0 week, 1 seconds, 9:12 ECEF position, '
          '15:18 ECEF velocity, 21:24 heading/pitch/roll)')
    print(f"   files: {', '.join(p.name for p in sorted(folder.glob('*GroundTruth.txt')))}")
    with (folder / f'{imu_type}_GroundTruth.txt').open(errors='replace') as f:
        lines = [next(f, '') for _ in range(60)]
    names = next((line.split() for line in lines if 'X-ECEF' in line), [])
    first = next((line.split() for line in lines if line.split() and line.split()[0].isdigit()), [])
    print(f'   {imu_type}_GroundTruth.txt, column [index] name = first value:')
    for i in range(0, len(first), 6):
        print('      ' + '  '.join(f'[{j}] {names[j] if j < len(names) else "?"} = {first[j]}'
                                    for j in range(i, min(i + 6, len(first)))))


def truth_consistency(data):
    print('\n3. Truth consistency')
    t = data.fusion_times
    derivative = (data.truth_position[2:] - data.truth_position[:-2]) / (t[2:] - t[:-2])[:, None]
    print(f'   |truth velocity - d(truth position)/dt| [m/s]: '
          f'{stats(np.linalg.norm(data.truth_velocity[1:-1] - derivative, axis=1))}   (expected < 0.05)')
    speed = np.linalg.norm(data.truth_velocity, axis=1)
    print(f'   speed [m/s]: min {speed.min():.2f}  max {speed.max():.2f}')
    if speed.max() < 0.1:
        print('   !!! the vehicle does not move in these epochs: the velocity columns, the IMU time offset and the '
              'dynamics are not tested (and training on them teaches the network nothing about driving); '
              'set a larger MAX_FUSION_EPOCHS')
    lever = np.einsum('kij,j->ki', data.truth_attitude, data.lever_arm)
    difference = np.array([ned(data, k, data.truth_antenna_position[k] - data.truth_position[k] - lever[k])
                           for k in range(len(t))])
    print(f'   lever arm (body) {np.round(data.lever_arm, 3)} m; antenna truth - (IMU truth + C l), NED [m]: '
          f'mean {np.round(difference.mean(axis=0), 3)}, {stats(np.linalg.norm(difference, axis=1))}'
          f'   (expected < 0.1)')


def one_second_ins(data, full_imu, mounting):
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
        difference = force - expected_force
        print(f'   measured - expected specific force: {np.round(difference / 9.80665 * 1e3, 2)} mg; a constant error '
              f'a gives a free-INS error of a t^2 / 2 = {0.5 * np.linalg.norm(difference) * 100.0 ** 2:.0f} m at 100 s. '
              f'It is an accelerometer bias or a tilt of {np.rad2deg(np.linalg.norm(difference) / 9.8):.3f} deg; '
              f'the filter starts with sigma {np.round(data.accel_bias_std / 9.80665 * 1e3, 3)} mg (bias) and '
              f'{np.rad2deg(cfg.INITIAL_ATTITUDE_STD):.3g} deg (attitude)')
        print(f'   README.xml SINS_RotAngle_IMU (mounting) {mounting} deg')
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
    errors, counts, satellites = {'G': [], 'C': [], 'L': []}, {'G': [], 'C': [], 'L': []}, {}
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
            if system != 'L' and len(rows):
                for i in rows:
                    satellites.setdefault(meas.sat_ids[i], []).append(
                        (error[i] - np.median(error[rows]), np.rad2deg(elevation[i]), meas.cn0[i]))
    for system, name in (('G', 'GPS'), ('C', 'BDS-3'), ('L', 'LEO')):
        if errors[system]:
            e = np.array(errors[system])
            print(f'   {name:5s}: {stats(e)}  mean {e.mean():+.3f}  > 30 m: {100 * np.mean(np.abs(e) > 30):.1f} %  '
                  f'satellites per epoch: {np.mean(counts[system]):.1f} (min {min(counts[system])}, '
                  f'max {max(counts[system])})')
    print('   expected: GPS / BDS-3 a few m (more in the urban parts), no large mean; LEO about 1-3 m')
    print('   per GPS / BDS-3 satellite: mean residual [m], mean elevation [deg], mean C/N0 [dB-Hz], epochs')
    for sat_id, values in sorted(satellites.items()):
        e, elevation, cn0 = np.array(values).T
        print(f'      {sat_id}  {e.mean():+7.2f}  {elevation.mean():5.1f}  {cn0.mean():5.1f}  {len(e)}'
              f"{'   !!!' if abs(e.mean()) > 10 else ''}")


def baselines(data, measurements):
    print('\n7. Filter baselines on the whole dataset, 3-D antenna RMSE [m]')
    last = len(data.fusion_times) - 1
    only_gnss = [m.subset(np.flatnonzero(m.systems != 'L')) for m in measurements]
    nothing = [m.subset(np.array([], dtype=int)) for m in measurements]
    network = None
    checkpoint_file = cfg.OUTPUT_FOLDER / 'masked_cla_network.pt'
    if checkpoint_file.exists():
        from navigation import MaskedCLANetwork
        checkpoint = torch.load(checkpoint_file)
        network = MaskedCLANetwork(checkpoint['max_measurements'])
        network.load_state_dict(checkpoint['state_dict'])
    line = f'   epochs 0..{last}:'
    for name, meas in (('free INS', nothing), ('EKF GNSS', only_gnss), ('EKF GNSS+LEO', measurements)):
        line += f'  {name} {run_filter(data, meas, None, 0, last, fault_detection=False)["position_rmse_m"]:.2f}'
    if network is not None:
        line += f'  network {run_filter(data, measurements, network, 0, last, False)["position_rmse_m"]:.2f}'
    print(line)


def main():
    split = sys.argv[1] if len(sys.argv) > 1 else 'train'
    folder = find_dataset_folder(cfg.TRAIN_FOLDER_NAME if split == 'train' else cfg.TEST_FOLDER_NAME)
    imu_type, mounting, _ = read_rover_info(folder / 'README.xml')
    data, orbits = load_dataset(split)
    measurements = orbits[cfg.LEO_TRAIN_ORBIT if split == 'train' else cfg.LEO_TEST_ORBITS[0]]
    print(f'{data.name}: {len(data.fusion_times)} fusion epochs, '
          f'{data.fusion_times[-1] - data.fusion_times[0]:.0f} s, '
          f'interval {np.median(np.diff(data.fusion_times)):.3f} s; IMU interval '
          f'{np.median(np.diff(data.imu_times)) * 1e3:.2f} ms (max gap {np.max(np.diff(data.imu_times)) * 1e3:.1f} ms)')
    imr_header(folder, imu_type)
    truth_columns(folder, imu_type)
    truth_consistency(data)
    one_second_ins(data, read_imu(folder / f'{imu_type}.imr', data.fusion_times[0]), mounting)
    free_ins(data)
    residuals(data, measurements)
    baselines(data, measurements)


if __name__ == '__main__':
    main()
