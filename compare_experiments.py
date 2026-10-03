"""Comparison of the training experiments (branches exp/...) found in settings.OUTPUT_ROOT.

    python compare_experiments.py   ->  OUTPUT_ROOT/comparison.txt (also printed), OUTPUT_ROOT/comparison.png

An experiment is a folder with training_info.json: OUTPUT_ROOT/<experiment> (and lr-sweep/lr_<rate>).
Table: training samples and epochs (the experiments are comparable only if these are the same: a warning
is printed otherwise), final training RMSE and, for each LEO orbit of the test, the 3-D RMSE of the network
and of the EKF on the testing dataset and the improvement. Figure: the test 3-D RMSE of the network of each
experiment, one panel per LEO orbit, with the EKF as reference.
"""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import settings as cfg

NETWORK_COLOR, EKF_COLOR = '#2a78d6', '#eb6834'          # as show_results.py
TEXT_COLOR, MUTED_COLOR, GRID_COLOR = '#0b0b0b', '#52514e', '#e1e0d9'


def find_experiments(root):
    """{name: folder} of every experiment under root, exp/paper first."""
    folders = {path.parent.relative_to(root).as_posix(): path.parent for path in root.rglob('training_info.json')}
    return dict(sorted(folders.items(), key=lambda item: (item[0] != 'paper', item[0])))


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


def table(experiments, orbits):
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


def figure(experiments, orbits, path):
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


def main():
    folders = find_experiments(cfg.OUTPUT_ROOT)
    if not folders:
        raise FileNotFoundError(f'no experiment (training_info.json) in {cfg.OUTPUT_ROOT}')
    experiments = {name: read_experiment(folder) for name, folder in folders.items()}
    orbits = list(next((e['test'] for e in experiments.values() if e['test']), {}))
    text = table(experiments, orbits)
    print(text)
    (cfg.OUTPUT_ROOT / 'comparison.txt').write_text(text + '\n')
    if orbits:
        figure(experiments, orbits, cfg.OUTPUT_ROOT / 'comparison.png')
        print(f"saved comparison.txt and comparison.png in {cfg.OUTPUT_ROOT}")


if __name__ == '__main__':
    main()
