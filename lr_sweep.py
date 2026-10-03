"""Learning-rate study of paper Fig. 15 (branch exp/lr-sweep): train and test with every learning rate of
settings.LEARNING_RATES, then compare them.

    python lr_sweep.py        train.py and test.py with --lr for each learning rate, then the comparison;
                              a learning rate whose training (all epochs) and test are already done is skipped
    python lr_sweep.py plot   only the comparison of the runs found

Each run writes to OUTPUT_ROOT/lr-sweep/lr_<rate>/ (python show_results.py --lr <rate> draws its figures).
The comparison, in OUTPUT_ROOT/lr-sweep/:
    results_lr_sweep.png   training position RMSE (cf. paper Fig. 15) and loss per epoch of every learning rate
    results_lr_sweep.txt   per learning rate: epochs, final and best training RMSE, test 3-D RMSE of the network
                           and the EKF for each LEO orbit (also printed)
"""
import json
import subprocess
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import settings as cfg

RATE_COLORS = ('#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4')   # categorical slots 1-5, in rate order


def run_folder(rate):
    return cfg.EXPERIMENT_FOLDER / f'lr_{rate:g}'


def training_done(folder):
    info_file = folder / 'training_info.json'
    if not info_file.exists():
        return False
    info = json.loads(info_file.read_text())
    return info.get('epochs_run') == info['max_epochs']


def test_done(folder):
    summary, checkpoint = folder / 'test_summary.json', folder / 'masked_cla_network.pt'
    return summary.exists() and summary.stat().st_mtime >= checkpoint.stat().st_mtime


def run_all():
    for rate in cfg.LEARNING_RATES:
        folder = run_folder(rate)
        for script, done in (('train.py', training_done), ('test.py', test_done)):
            if done(folder):
                print(f'learning rate {rate:g}: {script} already done')
                continue
            print(f'learning rate {rate:g}: {script}', flush=True)
            subprocess.run([sys.executable, script, '--lr', f'{rate:g}'], cwd=cfg.PROJECT_FOLDER, check=True)


def compare():
    runs = {rate: run_folder(rate) for rate in cfg.LEARNING_RATES
            if (run_folder(rate) / 'training_history.json').exists()}
    if not runs:
        raise FileNotFoundError(f'no run of {cfg.LEARNING_RATES} in {cfg.EXPERIMENT_FOLDER}; run lr_sweep.py first')
    histories = {rate: json.loads((folder / 'training_history.json').read_text()) for rate, folder in runs.items()}
    tests = {rate: json.loads((folder / 'test_summary.json').read_text()) for rate, folder in runs.items()
             if (folder / 'test_summary.json').exists()}
    orbits = list(next(iter(tests.values()))) if tests else []

    lines = ['LEARNING RATES (paper Fig. 15); RMSE in m',
             'rate      epochs  final train  best train (epoch)'
             + ''.join(f'  test {orbit}: network / EKF' for orbit in orbits)]
    for rate, history in histories.items():
        rmse = [h['train_position_rmse_m'] for h in history]
        best = min(range(len(rmse)), key=rmse.__getitem__)
        line = f'{rate:<9g} {len(history):6d}  {rmse[-1]:11.3f}  {rmse[best]:10.3f} ({history[best]["epoch"]:4d})'
        for orbit in orbits:
            if rate in tests:
                network, ekf = tests[rate][orbit]['masked_cla_kalmannet'], tests[rate][orbit]['traditional_ekf']
                line += f"  {network['rmse_3d_m']:{len(orbit) + 16}.2f} / {ekf['rmse_3d_m']:.2f}"
            else:
                line += f"  {'not tested':>{len(orbit) + 23}s}"
        lines.append(line)
    table = '\n'.join(lines)
    print(table)
    (cfg.EXPERIMENT_FOLDER / 'results_lr_sweep.txt').write_text(table + '\n')

    fig, axes = plt.subplots(1, 3, figsize=(20, 7), gridspec_kw={'width_ratios': [1, 1, 0.9]})
    for ax, key, label in ((axes[0], 'train_position_rmse_m', 'Training position RMSE [m]'),
                           (axes[1], 'train_loss', 'Training loss, Eq. (32)')):
        for color, rate in zip(RATE_COLORS, cfg.LEARNING_RATES):       # the color follows the rate, not the rank
            if rate in histories:
                ax.plot([h['epoch'] for h in histories[rate]], [h[key] for h in histories[rate]], color=color,
                        linewidth=1.5, label=f'learning rate {rate:g}')
        ax.set(xlabel='Epoch', ylabel=label, title=label)
        ax.grid(color='#e1e0d9', linewidth=0.8)
        ax.spines[['top', 'right']].set_visible(False)
        ax.legend(frameon=False)
    axes[1].set_yscale('log')
    axes[2].axis('off')
    axes[2].text(0.0, 1.0, table, family='monospace', fontsize=8, va='top')
    fig.tight_layout()
    fig.savefig(cfg.EXPERIMENT_FOLDER / 'results_lr_sweep.png', dpi=150)
    plt.close(fig)
    print(f'saved results_lr_sweep.png and results_lr_sweep.txt in {cfg.EXPERIMENT_FOLDER}')


if __name__ == '__main__':
    if sys.argv[1:] != ['plot']:
        run_all()
    compare()
