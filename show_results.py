"""Show the training and test results as the paper does, plus a table of the run settings.

    python show_results.py   (after train.py and test.py, of settings.LEARNING_RATE)

    outputs/results_training.png          loss and position RMSE per epoch (cf. paper Fig. 15) and the table
    outputs/results_errors_<orbit>.png    2-D trajectory and north/east/down errors over time (Fig. 18)
    outputs/results_cdf_<orbit>.png       CDF of the north/east/down errors (Fig. 19)
    outputs/results_stanford_<orbit>.png  horizontal and vertical Stanford diagram of each method (Fig. 20)
    outputs/results_orbits.png            CDF of the 3-D error of each method with every LEO filter orbit (A26)
    outputs/results_table.txt             the table (also printed)
<orbit> is each LEO filter orbit of the test (settings.LEO_TEST_ORBITS: reference, tle, network).

Reads outputs/training_info.json, training_history.json, test_summary.json, test_epochs.csv.

    python show_results.py compare   ->  OUTPUT_ROOT/comparison.txt (also printed), OUTPUT_ROOT/comparison.png

Comparison of the training experiments (branches exp/...) found in settings.OUTPUT_ROOT. An experiment is a
folder with training_info.json: OUTPUT_ROOT/<experiment> (and lr-sweep/lr_<rate>). Table: training samples
and epochs (the experiments are comparable only if these are the same: a warning is printed otherwise),
final training RMSE and, for each LEO orbit of the test, the 3-D RMSE of the network and of the EKF on the
testing dataset and the improvement. Figure: the test 3-D RMSE of the network of each experiment, one panel
per LEO orbit, with the EKF as reference.
"""
import csv
import json
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

import settings as cfg

TRAIN_COLOR = '#1baf7a'
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
    """{name: folder} of every experiment under root, main (training as the paper) first."""
    folders = {path.parent.relative_to(root).as_posix(): path.parent for path in root.rglob('training_info.json')}
    return dict(sorted(folders.items(), key=lambda item: (item[0] not in ('main', 'paper'), item[0] != 'main', item[0])))


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
    for key, label in (('training_samples', 'training samples'), ('max_epochs', 'epochs')):
        values = {e['info'].get(key) for e in experiments.values()}
        if len(values) > 1:
            lines.append(f'WARNING: the experiments differ in {label} {sorted(values, key=str)}: not comparable')
    test_epochs = {e['test'][orbits[0]]['masked_cla_kalmannet']['epochs'] for e in experiments.values() if e['test']}
    if len(test_epochs) > 1:
        lines.append(f'WARNING: the experiments differ in test epochs {sorted(test_epochs)}: not comparable')
    header = f"{'experiment':<{width}}  samples  epochs  train RMSE"
    for orbit in orbits:
        header += f'  | test {orbit}: network      EKF  improvement'
    lines.append(header)
    for name, e in experiments.items():
        info = e['info']
        line = (f"{name:<{width}}  {info['training_samples']:7d}  {info.get('epochs_run', 0):6d}  "
                f"{info.get('final_train_position_rmse_m', float('nan')):10.3f}")
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
    if legend:
        ax.legend(frameon=False)


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


lines = [
    'DATA',
    f"  training samples (epochs)   {info['training_samples']}",
    f"  test samples                {test[orbits[0]]['masked_cla_kalmannet']['epochs']}",
    f"  LEO orbit, training         {info['leo_train_orbit']}",
    f"  max measurements N_max      {info['max_measurements']}",
    'NETWORK',
    f"  input size                  {info['input_size']}",
    '  layers', *[f'    {layer}' for layer in info['network'].split(' -> ')],
    f"  trainable parameters        {info['trainable_parameters']:,}",
    'TRAINING',
    f"  learning rate               {info['learning_rate']}",
    f"  optimization                {info['optimization']}",
    f"  epochs run / max            {info['epochs_run']} / {info['max_epochs']}",
    f"  final train loss            {info['final_train_loss']:.4g}",
    f"  final train RMSE            {info['final_train_position_rmse_m']:.3f} m",
    f"  L2 weight                   {info['l2_weight']}",
    f"  training time               {info['training_time_s'] / 3600:.2f} h",
    'TEST by LEO orbit (A26)   LEO range RMS  3D RMSE network / EKF  LEO fault epochs network / EKF',
    *[f"  {orbit:<24s}{test[orbit]['leo_range_error']['rms_m']:10.1f} m"
      f"{test[orbit]['masked_cla_kalmannet']['rmse_3d_m']:12.2f} / {test[orbit]['traditional_ekf']['rmse_3d_m']:.2f} m"
      f"{test[orbit]['masked_cla_kalmannet']['epochs_with_leo_fault']:17d} / "
      f"{test[orbit]['traditional_ekf']['epochs_with_leo_fault']}"
      for orbit in orbits],
    *[line for orbit in orbits for line in test_lines(orbit)],
]
table = '\n'.join(lines)
print(table)
(folder / 'results_table.txt').write_text(table + '\n')

# --- Training (cf. paper Fig. 15) ------------------------------------------------------
epochs = [h['epoch'] for h in history]
fig, axes = plt.subplots(1, 3, figsize=(20, 8), gridspec_kw={'width_ratios': [1, 1, 0.9]})
for ax, key, label in ((axes[0], 'loss', 'Loss, Eq. (32)'), (axes[1], 'position_rmse_m', 'Position RMSE [m]')):
    ax.plot(epochs, [h[f'train_{key}'] for h in history], color=TRAIN_COLOR, linewidth=2, label='train')
    ax.set(xlabel='Epoch', ylabel=label, title=label)
    finish(ax)
axes[0].set_yscale('log')
axes[2].axis('off')
axes[2].text(0.0, 1.0, table, family='monospace', fontsize=8, va='top')
fig.tight_layout()
fig.savefig(folder / 'results_training.png', dpi=150)
plt.close(fig)

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
    # --- Trajectory and errors over time (cf. paper Fig. 18) -------------------------------
    fig, axes = plt.subplot_mosaic([['trajectory', 'North'], ['trajectory', 'East'], ['trajectory', 'Down']],
                                   figsize=(16, 9))
    truth_north = test_column(orbit, 'traditional_ekf', 'truth_north_m')
    truth_east = test_column(orbit, 'traditional_ekf', 'truth_east_m')
    axes['trajectory'].plot(truth_east, truth_north, color='#0b0b0b', linewidth=2, label='ground truth')
    for method, color in METHOD_COLORS.items():
        seconds = test_column(orbit, method, 'time_gpst_s') - test_column(orbit, method, 'time_gpst_s')[0]
        axes['trajectory'].plot(truth_east + test_column(orbit, method, 'east_error_m'),
                                truth_north + test_column(orbit, method, 'north_error_m'), color=color,
                                linewidth=1.2, label=method)
        for name, column in AXES:
            axes[name].plot(seconds, test_column(orbit, method, column), color=color, linewidth=1.2, label=method)
    axes['trajectory'].set(xlabel='East [m]', ylabel='North [m]', aspect='equal',
                           title=f'2-D trajectory, LEO orbit {orbit} (Fig. 18(a))')
    for name, _ in AXES:
        axes[name].set(xlabel='Time [s]', ylabel=f'{name} error [m]', title=f'{name} error (Fig. 18(b))')
    for ax in axes.values():
        finish(ax)
    fig.tight_layout()
    fig.savefig(folder / f'results_errors_{orbit}.png', dpi=150)
    plt.close(fig)

    # --- CDF of the errors (cf. paper Fig. 19) ----------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, (name, column) in zip(axes, AXES):
        for method, color in METHOD_COLORS.items():
            error = np.sort(np.abs(test_column(orbit, method, column)))
            ax.plot(error, np.arange(1, len(error) + 1) / len(error), color=color, linewidth=2, label=method)
        ax.set(xlabel=f'{name} error [m]', ylabel='Cumulative probability',
               title=f'{name} error CDF, LEO orbit {orbit} (Fig. 19)')
        finish(ax)
    fig.tight_layout()
    fig.savefig(folder / f'results_cdf_{orbit}.png', dpi=150)
    plt.close(fig)

    # --- Stanford diagrams (cf. paper Fig. 20) -------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 11))
    for row, method in enumerate(METHOD_COLORS):
        summary = test[orbit][method]
        north, east, down = (test_column(orbit, method, column) for _, column in AXES)
        for col, (direction, error, pl_name) in enumerate((('horizontal', np.hypot(north, east), 'horizontal_pl_m'),
                                                            ('vertical', np.abs(down), 'vertical_pl_m'))):
            ax, pl = axes[row, col], test_column(orbit, method, pl_name)
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
    fig.tight_layout()
    fig.savefig(folder / f'results_stanford_{orbit}.png', dpi=150)
    plt.close(fig)

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
fig.tight_layout()
fig.savefig(folder / 'results_orbits.png', dpi=150)
plt.close(fig)
print(f'saved results_training.png, results_orbits.png and results_errors/cdf/stanford_<orbit>.png in {folder}')
