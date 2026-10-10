"""Show the training and test results as the paper does, plus a table of the run settings.

    python show_results.py   (after train.py and test.py; outputs = settings.OUTPUT_FOLDER of this run)

    One chart per file, without the report text (that is only in results_table.txt):
    outputs/results_loss.png                      training and validation loss per epoch (cf. paper Fig. 15;
                                                  validation: A21)
    outputs/results_rmse.png                      training and validation position RMSE per epoch
    outputs/results_trajectory_<orbit>.png        2-D trajectory (Fig. 18(a))
    outputs/results_error_<axis>_<orbit>.png      north, east or down error over time (Fig. 18(b))
    outputs/results_cdf_<axis>_<orbit>.png        CDF of the north, east or down error (Fig. 19)
    outputs/results_stanford_<method>_<direction>_<orbit>.png
                                                  horizontal or vertical Stanford diagram of one method (Fig. 20)
    outputs/results_orbits.png                    CDF of the 3-D error of each method with every LEO filter orbit (A26)
    outputs/results_table.txt                     the text report (not printed, not in the figures)
<orbit> is each LEO filter orbit of the test (settings.LEO_TEST_ORBITS: reference, tle, network), <axis> north, east
or down, <method> masked_cla_kalmannet or traditional_ekf, <direction> horizontal or vertical.

Table: samples of the training, validation and test; network architecture (layers, neurons, parameters, from
the tested model masked_cla_network.pt); preprocessing used and not used; loss and position RMSE of train,
validation and test at the last epoch and for the tested model; the test of every LEO orbit.

Reads outputs/training_info.json, training_history.json, test_summary.json, test_epochs.csv, test_loss.json and
the saved models. Nothing is trained: if test_loss.json (test.py) is missing or older than the models, the saved
models run on the testing dataset to write it.

    python show_results.py compare   ->  OUTPUT_ROOT/comparison.txt (also printed), OUTPUT_ROOT/comparison.png

Comparison of the training experiments (branches exp/...) found in settings.OUTPUT_ROOT. An experiment is a
folder with training_info.json: OUTPUT_ROOT/<experiment>/<loss>/lr_<rate>. Table: training and
validation samples and epochs (the experiments are comparable only if these are the same: a warning is printed
otherwise), final training RMSE, epoch and validation RMSE of the tested model (best validation loss, A21) and,
for each LEO orbit of the test, the 3-D RMSE of the network and of the EKF on the
testing dataset and the improvement. Figure: the test 3-D RMSE of the network of each experiment, one panel
per LEO orbit, with the EKF as reference.
"""
import csv
import json
import re
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LogNorm

import settings as cfg

TRAIN_COLOR = '#1baf7a'
VALIDATION_COLOR = '#8a4fd6'
METHOD_COLORS = {'masked_cla_kalmannet': '#2a78d6', 'traditional_ekf': '#eb6834'}
AXES = (('North', 'north_error_m'), ('East', 'east_error_m'), ('Down', 'down_error_m'))
REGIONS = ('NO', 'MI', 'HO', 'SU', 'SU&MI')

# ===== Comparison of the experiments (python show_results.py compare) =================================================
NETWORK_COLOR, EKF_COLOR = METHOD_COLORS['masked_cla_kalmannet'], METHOD_COLORS['traditional_ekf']
TEXT_COLOR, MUTED_COLOR, GRID_COLOR = '#0b0b0b', '#52514e', '#e1e0d9'

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import settings as cfg



def find_experiments(root):
    """{name: folder} of every experiment under root (name: <experiment>/<loss>/lr_<rate>), main first."""
    folders = {path.parent.relative_to(root).as_posix(): path.parent for path in root.rglob('training_info.json')}
    return dict(sorted(folders.items(), key=lambda item: (not item[0].startswith('main/'), item[0])))


def read_experiment(folder):
    info = json.loads((folder / 'training_info.json').read_text())
    summary_file, model_file = folder / 'test_summary.json', folder / 'masked_cla_network.pt'
    test = json.loads(summary_file.read_text()) if summary_file.exists() else None
    notes = []
    if info.get('epochs_run', 0) < info['max_epochs']:
        notes.append(f"training not finished ({info.get('epochs_run', 0)}/{info['max_epochs']})")
    if test is None:
        notes.append('not tested')
    elif model_file.exists() and summary_file.stat().st_mtime < model_file.stat().st_mtime:
        notes.append('test older than the model')
    return {'info': info, 'test': test, 'notes': notes}


def comparison_table(experiments, orbits):
    width = max(len(name) for name in experiments)
    lines = [f'EXPERIMENTS in {cfg.OUTPUT_ROOT}; RMSE in m, test = 3-D RMSE on the testing dataset']
    for key, label in (('training_samples', 'training samples'), ('validation_samples', 'validation samples'),
                       ('max_epochs', 'epochs')):
        values = {e['info'].get(key) for e in experiments.values()}
        if len(values) > 1:
            lines.append(f'WARNING: the experiments differ in {label} {sorted(values, key=str)}: not comparable')
    test_epochs = {e['test'][orbits[0]]['masked_cla_kalmannet']['epochs'] for e in experiments.values() if e['test']}
    if len(test_epochs) > 1:
        lines.append(f'WARNING: the experiments differ in test epochs {sorted(test_epochs)}: not comparable')
    header = f"{'experiment':<{width}}  samples  epochs  train RMSE  best epoch  valid RMSE"
    for orbit in orbits:
        header += f'  | test {orbit}: network      EKF  improvement'
    lines.append(header)
    for name, e in experiments.items():
        info = e['info']
        line = (f"{name:<{width}}  {info['training_samples']:7d}  {info.get('epochs_run', 0):6d}  "
                f"{info.get('final_train_position_rmse_m', float('nan')):10.3f}  {info.get('best_epoch', '-'):>10}  "
                f"{info.get('best_validation_position_rmse_m', float('nan')):10.3f}")
        for orbit in orbits:
            if e['test'] and orbit in e['test']:
                network = e['test'][orbit]['masked_cla_kalmannet']['rmse_3d_m']
                ekf = e['test'][orbit]['traditional_ekf']['rmse_3d_m']
                line += f"  | {'':{len(orbit) + 5}s}{network:8.2f} {ekf:8.2f} {100 * (ekf - network) / ekf:10.1f} %"
            else:
                line += f"  | {'':{len(orbit) + 5}s}{'-':>8s} {'-':>8s} {'-':>12s}"
        lines.append(line + ('   ' + '; '.join(e['notes']) if e['notes'] else ''))
    return '\n'.join(lines)


def comparison_figure(experiments, orbits, path):
    tested = {name: e for name, e in experiments.items() if e['test']}
    names = list(tested)[::-1]                                       # first experiment at the top
    fig, axes = plt.subplots(1, len(orbits), figsize=(7 * len(orbits), 1.6 + 0.5 * len(names)), squeeze=False)
    for ax, orbit in zip(axes[0], orbits):
        network = [tested[n]['test'][orbit]['masked_cla_kalmannet']['rmse_3d_m'] for n in names]
        ekf = [tested[n]['test'][orbit]['traditional_ekf']['rmse_3d_m'] for n in names]
        limit = min(max(network + ekf) * 1.15, 4.0 * max(ekf))       # a diverged network does not hide the others
        ax.barh(names, [min(v, limit) for v in network], height=0.7, color=NETWORK_COLOR, label='network')
        for y, (value, reference) in enumerate(zip(network, ekf)):
            if value > limit:                                        # inside the clipped bar
                ax.text(0.99 * limit, y, f'{value:.2f} (beyond the axis)', va='center', ha='right', fontsize=9,
                        color='white')
            else:                                                    # after the bar and the EKF mark
                ax.text(max(value, reference) + 0.02 * limit, y, f'{value:.2f}', va='center', ha='left',
                        fontsize=9, color=TEXT_COLOR)
        if max(ekf) - min(ekf) < 1e-6:                               # the same EKF in every experiment
            ax.axvline(ekf[0], color=EKF_COLOR, linewidth=2, linestyle='--', label=f'EKF {ekf[0]:.2f}')
        else:
            ax.scatter(ekf, names, marker='|', s=300, linewidths=2, color=EKF_COLOR, zorder=3,
                       label='EKF of the experiment')
        ax.set_xlim(0, limit * 1.12)
        ax.set_xlabel('Test 3-D RMSE [m]')
        ax.set_title(f'LEO orbit {orbit}', pad=30)
        ax.grid(axis='x', color=GRID_COLOR, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(colors=MUTED_COLOR)
        ax.legend(frameon=False, loc='lower left', bbox_to_anchor=(0.0, 1.0), ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def compare_experiments():
    folders = find_experiments(cfg.OUTPUT_ROOT)
    if not folders:
        raise FileNotFoundError(f'no experiment (training_info.json) in {cfg.OUTPUT_ROOT}')
    experiments = {name: read_experiment(folder) for name, folder in folders.items()}
    orbits = list(next((e['test'] for e in experiments.values() if e['test']), {}))
    text = comparison_table(experiments, orbits)
    print(text)
    (cfg.OUTPUT_ROOT / 'comparison.txt').write_text(text + '\n')
    if orbits:
        comparison_figure(experiments, orbits, cfg.OUTPUT_ROOT / 'comparison.png')
        print(f"saved comparison.txt and comparison.png in {cfg.OUTPUT_ROOT}")


if 'compare' in sys.argv[1:]:
    compare_experiments()
    sys.exit()

# ===== Results of this experiment =====================================================================================
folder = cfg.OUTPUT_FOLDER
info = json.loads((folder / 'training_info.json').read_text())
history = json.loads((folder / 'training_history.json').read_text())
test = json.loads((folder / 'test_summary.json').read_text())
with open(folder / 'test_epochs.csv') as f:
    rows = list(csv.DictReader(f))
orbits = list(test)
METHOD_STYLES = {'masked_cla_kalmannet': '-', 'traditional_ekf': '--'}
ORBIT_COLORS = {'reference': '#1baf7a', 'tle': '#d03b3b', 'network': '#2a78d6'}


def test_column(orbit, method, name):
    return np.array([float(r[name]) for r in rows if r['leo_orbit'] == orbit and r['method'] == method])


def finish(ax, legend=True):
    ax.grid(color='#e5e5e3', linewidth=0.8)
    ax.spines[['top', 'right']].set_visible(False)
    if legend:   # the same place in every figure: outside the axes, right of them, at the top (never over the data)
        ax.legend(frameon=False, loc='upper left', bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)


# --- Table (paper Table IV and Fig. 20 numbers, plus the run settings) -------------------
def test_lines(orbit):
    """Table IV, Fig. 20 numbers and FDE alarms of one LEO filter orbit."""
    network, ekf = test[orbit]['masked_cla_kalmannet'], test[orbit]['traditional_ekf']

    def improvement(key):
        return 100.0 * (1.0 - network[key] / ekf[key])

    return [
        f'TEST, LEO orbit {orbit}: position RMSE [m]',
        '                              network      EKF  improvement',
        *[f"  {name:<28s}{network[key]:7.2f}  {ekf[key]:7.2f}  {improvement(key):9.2f} %"
          for name, key in (('North', 'rmse_north_m'), ('East', 'rmse_east_m'), ('Down', 'rmse_down_m'),
                            ('3D', 'rmse_3d_m'))],
        f'TEST, LEO orbit {orbit}: Stanford [% of epochs]',
        *[f"  {direction} {region:<{27 - len(direction)}s}{network[key][region]:7.2f}  {ekf[key][region]:7.2f}"
          for direction, key in (('horizontal', 'stanford_horizontal_percent'),
                                 ('vertical', 'stanford_vertical_percent'))
          for region in REGIONS],
        f"  epochs with fault           {network['epochs_with_fault']:7d}  {ekf['epochs_with_fault']:7d}",
        f"  epochs with LEO fault       {network['epochs_with_leo_fault']:7d}  {ekf['epochs_with_leo_fault']:7d}",
    ]


# --- Network architecture, from the tested model (masked_cla_network.pt) --------------------
def architecture_lines():
    """Layers, neurons and parameters of the saved network (its weights, so the table is that of the trained model;
    pooling and dropout are not in the weights: from training_info.json)."""
    checkpoint = torch.load(folder / 'masked_cla_network.pt')
    weights, n_max = checkpoint['state_dict'], checkpoint['max_measurements']
    pool = re.search(r'max-pool (\d+)', info['network'])
    dropout = re.search(r'dropout ([\d.]+)', info['network'])
    pool = pool.group(1) if pool else cfg.POOL_KERNEL_SIZE
    dropout = dropout.group(1) if dropout else cfg.LSTM_DROPOUT

    def parameters(prefix, suffix=''):
        return sum(w.numel() for name, w in weights.items()
                   if name.startswith(prefix) and name.endswith(suffix) and 'loss_weight' not in name
                   and 'gain_row_scale' not in name)

    filters, _, kernel = weights['conv.weight'].shape
    lstm_layers = sorted(int(name.rsplit('_l', 1)[1]) for name in weights if name.startswith('lstm.weight_ih_l'))
    units = weights['lstm.weight_hh_l0'].shape[1]
    fc_units, gain_size = weights['fc_hidden.weight'].shape[0], weights['fc_output.weight'].shape[0]
    D = info['input_size']
    layers = [('masked Conv1D (Eq. 22)', f'{filters} filters, kernel {kernel}', f'{filters} x D', 'ReLU',
               parameters('conv')),
              ('max-pool (stride 1)', f'size {pool}', f'{filters} x D', '-', 0),
              *[(f'LSTM layer {i + 1} (Eq. 24-25)', f'{units} units', f'L x {units}', 'tanh, sigmoid',
                 parameters('lstm.', f'_l{i}')) for i in lstm_layers],
              ('attention W_h, b_h (Eq. 26)', f'{weights["attention_hidden.weight"].shape[0]}', f'L x {units}', 'tanh',
               parameters('attention_hidden')),
              ('attention v (Eq. 26-29)', '1 score per step', f'{units}', 'softmax', parameters('attention_vector')),
              ('FC hidden', f'{fc_units}', f'{fc_units}', 'ReLU', parameters('fc_hidden')),
              ('FC output: Kalman gain', f'{gain_size} = 15 x N_max', f'15 x {n_max}', 'linear',
               parameters('fc_output'))]
    rows, number = [], 0
    for name, units_text, output, activation, count in layers:
        number += count > 0                          # layers with weights are numbered, the max-pool is not
        rows.append(f"  {number if count else '-':>2}  {name:<28}{units_text:<22}{output:<12}{activation:<15}"
                    f"{count:>10,}")
    return [
        f'NETWORK ARCHITECTURE (tested model; input X_k, D = 36 + 2 N_max = {D}, N_max = {n_max}, '
        f'L = valid entries of X_k <= D)',
        f"  {'#':>2}  {'layer':<28}{'neurons / units':<22}{'output':<12}{'activation':<15}{'parameters':>10}",
        *rows,
        f'  layers with weights         {number} (Conv1D 1, LSTM {len(lstm_layers)}, attention 2, FC 2), plus max-pool',
        f'  LSTM dropout                {dropout} between the LSTM layers (training only)',
        f"  trainable parameters        {info['trainable_parameters']:,}",
    ]


def preprocessing_lines():
    """What is done to the network input and output and to the labels (A13, A16, A28)."""
    loss_weights = info.get('loss_weights_p_v_theta', [1.0])
    scale = info.get('gain_row_scale_p_v_theta_ba_bg')
    if scale and any(abs(v - 1.0) > 1e-12 for v in scale):
        scale_text = f"used: K = diag(s) K_net, s p / v / theta / b_a / b_g = {' / '.join(f'{v:.3g}' for v in scale)} (A28)"
    else:
        scale_text = 'not used'
    if len(loss_weights) > 1:
        weight_text = f"used: p / v / theta = {' / '.join(f'{w:.3g}' for w in loss_weights)} (LOSS = 'pva', A28)"
    else:
        weight_text = "not used: position labels only (LOSS = 'p', paper)"
    normalization = info.get('input_normalization')                # set by exp/input-norm-grad-clip only
    if normalization == 'zscore':
        normalization_text = 'used: z-score (mean and standard deviation of the training part)'
    elif normalization == 'l2':
        normalization_text = 'used: L2 (each feature group to unit norm, as KalmanNet)'
    else:
        normalization_text = 'not used: X_k (Eq. (15)-(16)) enters the network as it is (A13)'
    return [
        'PREPROCESSING',
        f'  input normalization         {normalization_text}',
        f"  zero padding and mask       used: X_k padded to D = {info['input_size']}, mask M_k (Eq. (17)), "
        f"at most N_max = {info['max_measurements']} measurements (A16)",
        f'  gain row scale (output)     {scale_text}',
        f'  loss weights (labels)       {weight_text}',
    ]


def test_loss():
    """test_loss.json of test.py; if it is missing or older than the models, it is written here: the saved models
    run on the testing dataset (no training). {} if that fails (e.g. no dataset here)."""
    path = folder / 'test_loss.json'
    models = [p for p in (folder / 'masked_cla_network.pt', folder / 'masked_cla_network_last.pt') if p.exists()]
    if path.exists() and all(path.stat().st_mtime >= p.stat().st_mtime for p in models):
        return json.loads(path.read_text())
    print(f'{path.name} missing or older than the models: the saved models run on the testing dataset '
          f'(no training)')
    try:
        from test import write_test_loss
        return write_test_loss()
    except Exception as error:                     # the other results are shown all the same
        print(f'WARNING: no test loss ({type(error).__name__}: {error}); run python test.py')
        return {}


def results_lines():
    """Loss and position RMSE of train, validation and test at the last epoch and for the tested model."""
    by_epoch = {h['epoch']: h for h in history}
    last, best = history[-1], by_epoch.get(info.get('best_epoch'), {})
    test_run = test_loss()
    na = float('nan')

    def number(value, form):
        return '-' if value != value else format(value, form)         # NaN: not available

    def row(name, samples, at_last, at_best):
        return (f"  {name:<12}{samples:>9}  {number(at_last[0], '.4g'):>12}{number(at_last[1], '.3f'):>12}    "
                f"{number(at_best[0], '.4g'):>12}{number(at_best[1], '.3f'):>12}")

    def test_values(model):
        values = test_run.get(model, {})
        return values.get('loss', na), values.get('position_rmse_m', na)

    lines = [
        f"RESULTS: loss (Eq. (30), MSE of the labels {info.get('loss', 'p')}) and position RMSE [m]",
        f"  {'':<12}{'':>9}  {'last epoch ' + str(last['epoch']):^24}    "
        f"{'tested model, epoch ' + str(info.get('best_epoch', '-')):^24}",
        f"  {'':<12}{'samples':>9}  {'loss':>12}{'RMSE [m]':>12}    {'loss':>12}{'RMSE [m]':>12}",
        row('train', info['training_samples'], (last['train_loss'], last['train_position_rmse_m']),
            (best.get('train_loss', na), best.get('train_position_rmse_m', na))),
        row('validation', info.get('validation_samples', '-'),
            (last.get('validation_loss', na), last.get('validation_position_rmse_m', na)),
            (best.get('validation_loss', na), best.get('validation_position_rmse_m', na))),
        row('test', test_run.get('samples', test[orbits[0]]['masked_cla_kalmannet']['epochs']),
            test_values('last_epoch_model'), test_values('tested_model')),
        '  train: training pass of the epoch (dropout on); validation and test: no dropout, no gradient, no FDE,',
        f"  LEO orbit {test_run.get('leo_orbit', info['leo_train_orbit'])} (test: test_loss.json; with FDE and "
        f"every LEO orbit: TEST below)",
        f"  traditional EKF validation RMSE {info.get('classical_ekf_validation_position_rmse_m', na):.3f} m",
    ]
    if test_run.get('last_epoch_model', {}).get('epoch') not in (None, last['epoch']):
        lines.append(f"  WARNING: the last model of test_loss.json is of epoch {test_run['last_epoch_model']['epoch']}")
    return lines


lines = [
    'DATA (samples = fusion epochs, one per GNSS epoch)',
    f"  training samples            {info['training_samples']} ({info.get('dataset', '-')}, fusion epochs "
    f"{info.get('training_fusion_epochs', '-')})",
    f"  validation samples (A21)    {info.get('validation_samples', '-')} ({info.get('validation_dataset', '-')}, "
    f"fusion epochs {info.get('validation_fusion_epochs', '-')})",
    f"  test samples                {test[orbits[0]]['masked_cla_kalmannet']['epochs']} ({cfg.TEST_FOLDER_NAME})",
    f"  LEO orbit, training         {info['leo_train_orbit']}",
    f"  max measurements N_max      {info['max_measurements']}",
    *architecture_lines(),
    *preprocessing_lines(),
    'TRAINING',
    f"  loss labels (A28)           {info.get('loss', 'p')}",
    f"  learning rate               {info['learning_rate']}",
    f"  optimization                {info['optimization']}",
    f"  epochs run / max            {info['epochs_run']} / {info['max_epochs']}",
    f"  tested model                epoch {info.get('best_epoch', '-')} (best validation loss, A21)",
    f"  L2 weight                   {info['l2_weight']}",
    f"  training time               {info['training_time_s'] / 3600:.2f} h",
    *results_lines(),
    'TEST by LEO orbit (A26)   LEO range RMS  3D RMSE network / EKF  LEO fault epochs network / EKF',
    *[f"  {orbit:<24s}{test[orbit]['leo_range_error']['rms_m']:10.1f} m"
      f"{test[orbit]['masked_cla_kalmannet']['rmse_3d_m']:12.2f} / {test[orbit]['traditional_ekf']['rmse_3d_m']:.2f} m"
      f"{test[orbit]['masked_cla_kalmannet']['epochs_with_leo_fault']:17d} / "
      f"{test[orbit]['traditional_ekf']['epochs_with_leo_fault']}"
      for orbit in orbits],
    *[line for orbit in orbits for line in test_lines(orbit)],
]
(folder / 'results_table.txt').write_text('\n'.join(lines) + '\n')     # the text report: only in this file

# ===== Figures: one chart per file, no report text ========================================
saved = []


def save(fig, name):
    fig.tight_layout()
    fig.savefig(folder / name, dpi=150, bbox_inches='tight')     # the legend outside the axes is kept
    plt.close(fig)
    saved.append(name)


# --- Training: loss and position RMSE per epoch (cf. paper Fig. 15) ---------------------
epochs = [h['epoch'] for h in history]
for key, label, name in (('loss', 'Loss, Eq. (32)', 'results_loss.png'),
                         ('position_rmse_m', 'Position RMSE [m]', 'results_rmse.png')):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(epochs, [h[f'train_{key}'] for h in history], color=TRAIN_COLOR, linewidth=2, label='train')
    if 'validation_loss' in history[0]:
        ax.plot(epochs, [h[f'validation_{key}'] for h in history], color=VALIDATION_COLOR, linewidth=2,
                label='validation')
        ax.axvline(info['best_epoch'], color=VALIDATION_COLOR, linewidth=1, linestyle='--',
                   label=f"tested model (epoch {info['best_epoch']})")
    ax.set(xlabel='Epoch', ylabel=label, title=f'{label} per epoch')
    if key == 'loss':
        ax.set_yscale('log')
    finish(ax)
    save(fig, name)

# --- Regions of the Stanford diagrams (cf. paper Fig. 20) --------------------------------
al = cfg.ALERT_LIMIT
limit = 4.0 / 3.0 * al
region_shapes = {                                   # (polygon, fill color, label position)
    'MI': ([(0, 0), (al, 0), (al, al)], '#fab219', (0.62 * al, 0.3 * al)),
    'HO': ([(al, 0), (limit, 0), (limit, al), (al, al)], '#d03b3b', (1.17 * al, 0.5 * al)),
    'SU': ([(0, al), (al, al), (limit, limit), (0, limit)], '#ec835a', (0.4 * al, 1.17 * al)),
    'SU&MI': ([(al, al), (limit, al), (limit, limit)], '#d03b3b', (1.2 * al, 1.08 * al)),
}
for orbit in orbits:
    # --- 2-D trajectory (cf. paper Fig. 18(a)) ---------------------------------------------
    truth_north = test_column(orbit, 'traditional_ekf', 'truth_north_m')
    truth_east = test_column(orbit, 'traditional_ekf', 'truth_east_m')
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(truth_east, truth_north, color='#0b0b0b', linewidth=2, label='ground truth')
    for method, color in METHOD_COLORS.items():
        ax.plot(truth_east + test_column(orbit, method, 'east_error_m'),
                truth_north + test_column(orbit, method, 'north_error_m'), color=color, linewidth=1.2, label=method)
    ax.set(xlabel='East [m]', ylabel='North [m]', aspect='equal', title=f'2-D trajectory, LEO orbit {orbit} (Fig. 18(a))')
    finish(ax)
    save(fig, f'results_trajectory_{orbit}.png')

    for name, column in AXES:
        # --- Error over time (cf. paper Fig. 18(b)) ---------------------------------------
        fig, ax = plt.subplots(figsize=(10, 4.5))
        for method, color in METHOD_COLORS.items():
            seconds = test_column(orbit, method, 'time_gpst_s') - test_column(orbit, method, 'time_gpst_s')[0]
            ax.plot(seconds, test_column(orbit, method, column), color=color, linewidth=1.2, label=method)
        ax.set(xlabel='Time [s]', ylabel=f'{name} error [m]', title=f'{name} error, LEO orbit {orbit} (Fig. 18(b))')
        finish(ax)
        save(fig, f'results_error_{name.lower()}_{orbit}.png')

        # --- CDF of the error (cf. paper Fig. 19) -----------------------------------------
        fig, ax = plt.subplots(figsize=(7, 5))
        for method, color in METHOD_COLORS.items():
            error = np.sort(np.abs(test_column(orbit, method, column)))
            ax.plot(error, np.arange(1, len(error) + 1) / len(error), color=color, linewidth=2, label=method)
        ax.set(xlabel=f'{name} error [m]', ylabel='Cumulative probability',
               title=f'{name} error CDF, LEO orbit {orbit} (Fig. 19)')
        finish(ax)
        save(fig, f'results_cdf_{name.lower()}_{orbit}.png')

    # --- Stanford diagrams (cf. paper Fig. 20) -------------------------------------------
    for method in METHOD_COLORS:
        summary = test[orbit][method]
        north, east, down = (test_column(orbit, method, column) for _, column in AXES)
        for direction, error, pl_name in (('horizontal', np.hypot(north, east), 'horizontal_pl_m'),
                                          ('vertical', np.abs(down), 'vertical_pl_m')):
            fig, ax = plt.subplots(figsize=(7, 6))
            pl = test_column(orbit, method, pl_name)
            percent = summary[f'stanford_{direction}_percent']
            for region, (polygon, color, (x, y)) in region_shapes.items():
                ax.fill(*zip(*polygon), color=color, alpha=0.25, linewidth=0)
                ax.text(x, y, f'{region}\n{percent[region]:.3f} %', ha='center', va='center', fontsize=9)
            ax.text(0.2 * al, 0.8 * al, f"NO\n{percent['NO']:.3f} %", ha='center', va='center', fontsize=9)
            counts, x_edges, y_edges = np.histogram2d(error, pl, bins=80, range=[[0, limit], [0, limit]])
            x_bin = np.clip(np.searchsorted(x_edges, error) - 1, 0, 79)
            y_bin = np.clip(np.searchsorted(y_edges, pl) - 1, 0, 79)
            points = ax.scatter(error, pl, c=np.maximum(counts[x_bin, y_bin], 1), cmap='Blues', norm=LogNorm(vmin=1),
                                s=6)
            ax.plot([0, limit], [0, limit], color='#0b0b0b', linewidth=1)
            ax.set(xlim=(0, limit), ylim=(0, limit), xlabel='Position error [m]', ylabel='Protection level [m]',
                   title=f'{method}, LEO orbit {orbit}: {direction} (AL = {al:g} m, Fig. 20)')
            fig.colorbar(points, ax=ax, label='epochs per cell')
            finish(ax, legend=False)
            save(fig, f'results_stanford_{method}_{direction}_{orbit}.png')

# --- 3-D error CDF with every LEO filter orbit (A26) -------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
for orbit in orbits:
    for method, style in METHOD_STYLES.items():
        error = np.sort(np.linalg.norm([test_column(orbit, method, column) for _, column in AXES], axis=0))
        ax.plot(error, np.arange(1, len(error) + 1) / len(error), color=ORBIT_COLORS.get(orbit, '#52514e'),
                linestyle=style, linewidth=2, label=f'{method}, LEO orbit {orbit}')
ax.set(xscale='log', xlabel='3-D position error [m]', ylabel='Cumulative probability',
       title='3-D error CDF by LEO filter orbit')
finish(ax)
save(fig, 'results_orbits.png')
print(f'saved results_table.txt (text report) and {len(saved)} figures in {folder}:')
print('\n'.join(f'  {name}' for name in saved))
